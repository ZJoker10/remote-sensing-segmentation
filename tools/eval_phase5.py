"""Evaluate the phase-5 two-head checkpoint on full LoveDA validation tiles.

Why this exists: `train_multihead.py` scores 512-px random crops from a 200-tile
subset. That is fine for model selection but it is not the number to report, so
this script runs the real inference path (`rs_core`, the same one the app uses)
over whole 1024x1024 tiles and reports per-head IoU.

Two deliberate choices:

* The IoU sample is a **fixed stride over all validation tiles**, not a filtered
  subset. Filtering for "tiles that contain water" would inflate the score.
* The contact sheet is the opposite: it picks the tiles with the most water and
  building so a human can see failure modes. It is labelled as such, and its
  numbers are not the reported ones.

Usage:
    venv\\Scripts\\python.exe tools/eval_phase5.py --limit 60
    venv\\Scripts\\python.exe tools/eval_phase5.py --limit 60 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from phase5_multihead.loveda import (  # noqa: E402
    T_BUILDING, T_IGNORE, T_WATER, find_tiles, to_compact,
)
from rs_core.inference import predict_probs  # noqa: E402
from rs_core.model import load_model      # noqa: E402

HEAD_NAMES = ("building", "water")
PALETTE = {  # RGB
    "rgb":  None,
    # Same background grey in both columns: otherwise "grey" vs "black" reads as
    # a difference between truth and prediction when it is only a legend artifact.
    "gt":   {T_BUILDING: (235, 60, 60), T_WATER: (40, 120, 235), 0: (60, 60, 60)},
    "pred": {1: (235, 60, 60), 2: (40, 120, 235), 0: (60, 60, 60)},
}


def colourise(compact: np.ndarray, table: dict) -> np.ndarray:
    """Map a class-id image to RGB. Ignores stay black."""
    out = np.zeros((*compact.shape, 3), dtype=np.uint8)
    for cls, rgb in table.items():
        out[compact == cls] = rgb
    return out


def iou_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Val")
    ap.add_argument("--limit", type=int, default=60,
                    help="tiles to score (fixed stride across the whole split)")
    ap.add_argument("--stride", type=int, default=None,
                    help="0 means 'every tile'. Default: spread --limit over the split")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--panel", type=int, default=4,
                    help="tiles to put in the visual contact sheet")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--out", default="./outputs/phase5_eval")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(cfg["phase5"]["checkpoint"])
    if not ckpt_path.is_file():
        raise SystemExit(f"no phase-5 checkpoint at {ckpt_path} — train it first")
    model = load_model(5, cfg, device).eval()

    pairs = find_tiles(ROOT / "data" / "LoveDA", args.split)
    if not pairs:
        raise SystemExit(f"no {args.split} tiles found")
    stride = args.stride or max(1, len(pairs) // args.limit)
    sample = pairs[::stride]
    print(f"{args.split}: {len(pairs)} tiles total -> scoring {len(sample)} "
          f"(every {stride}th, unfiltered)")
    print(f"Device: {device} | threshold {args.threshold}")

    counts = {name: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for name in HEAD_NAMES}
    per_tile: list[tuple[float, Path, np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray]] = []

    for i, (img_path, mask_path) in enumerate(sample, 1):
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gt = to_compact(np.array(Image.open(mask_path)))
        valid = gt != T_IGNORE

        probs = predict_probs(model, rgb, device)          # (2, H, W)
        pred = np.zeros(gt.shape, dtype=np.uint8)
        pred[probs[0] > args.threshold] = T_BUILDING
        pred[probs[1] > args.threshold] = T_WATER

        for name, cls in zip(HEAD_NAMES, (T_BUILDING, T_WATER)):
            truth = (gt == cls) & valid
            guess = (pred == cls) & valid
            counts[name]["tp"] += float((truth & guess).sum())
            counts[name]["fp"] += float((~truth & guess).sum())
            counts[name]["fn"] += float((truth & ~guess).sum())

        content = float((gt == T_BUILDING).sum() + (gt == T_WATER).sum())
        per_tile.append((content, img_path, rgb, gt, pred, probs))
        if i % 10 == 0:
            print(f"  scored {i}/{len(sample)}", flush=True)

    print("\n=== full-tile results (unfiltered sample) ===")
    print(f"{'head':10s} {'IoU':>7s} {'precision':>10s} {'recall':>8s} {'F1':>7s}")
    ious = []
    for name in HEAD_NAMES:
        c = counts[name]
        iou = iou_from_counts(c["tp"], c["fp"], c["fn"])
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if prec + rec else float("nan")
        ious.append(iou)
        print(f"{name:10s} {iou:7.4f} {prec:10.4f} {rec:8.4f} {f1:7.4f}")
    print(f"{'mean':10s} {np.nanmean(ious):7.4f}")
    print(f"\nval tiles scored: {len(sample)} | "
          f"valid px: {sum(c['tp'] + c['fn'] for c in counts.values()):.0f}")

    # ── Contact sheet: the highest-content tiles, for eyeballing only ─────────
    per_tile.sort(key=lambda t: -t[0])
    chosen = per_tile[:args.panel]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell = 256
    sheet = Image.new("RGB", (cell * 3, cell * len(chosen) + 22), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), "left: RGB | middle: ground truth | right: prediction "
                      f"(threshold {args.threshold})", fill=(220, 220, 220))
    for row, (_, img_path, rgb, gt, pred, _) in enumerate(chosen):
        y = 22 + row * cell
        gt_rgb = colourise(gt, PALETTE["gt"])
        pr_rgb = colourise(pred, PALETTE["pred"])
        for col, panel in enumerate((rgb, gt_rgb, pr_rgb)):
            im = Image.fromarray(panel).resize((cell, cell), Image.NEAREST)
            sheet.paste(im, (col * cell, y))
        draw.text((6, y + 4), f"{img_path.parent.parent.name}/{img_path.stem}",
                  fill=(255, 255, 255))
    sheet_path = out_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"\ncontact sheet (highest-content tiles, NOT the scored sample): {sheet_path}")
    print("  legend: red = building, blue = water, dark grey = background")
    return 0


if __name__ == "__main__":
    sys.exit(main())
