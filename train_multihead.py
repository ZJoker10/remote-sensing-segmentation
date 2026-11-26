"""Train the two-head building + water model on LoveDA.

Why a separate script: the four original phases are single-class building
segmentation on WHU. Water needs a different dataset and a per-class loss mask,
which is a different training contract, not a config flag.

Heads are simply the two output channels of ResNetUNet(num_classes=2):
    channel 0 -> building logit
    channel 1 -> water logit
Each is trained with masked BCE + Dice; LoveDA's nodata pixels are excluded from
both instead of being labelled "background".

Usage:
    venv\\Scripts\\python.exe train_multihead.py --epochs 12 --tag phase5_loveda
    venv\\Scripts\\python.exe train_multihead.py --epochs 2 --limit-train 200 --val-limit 40
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Redirected output on Windows defaults to cp1252 and cannot encode the banners.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from phase1_baseline.resnet_unet import ResNetUNet          # noqa: E402
from phase5_multihead.loveda import (                        # noqa: E402
    CLASS_NAMES,
    LoveDABinary,
    LoveDATiles,
    class_balance,
    verify_mapping,
)
from rs_core.inference import predict_probs                  # noqa: E402

CLASS_ORDER = CLASS_NAMES          # ("building", "water")


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def masked_bce(logits: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)


def masked_dice(logits: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits) * valid
    t = target * valid
    inter = (p * t).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return 1.0 - ((2 * inter + 1.0) / (denom + 1.0)).mean()


def multi_head_loss(logits, targets: dict[str, torch.Tensor],
                    valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    parts = {}
    total = logits.new_zeros(())
    for idx, name in enumerate(CLASS_ORDER):
        target = targets[name]
        # A batch (or a crop) may contain no pixels of a class at all; masked
        # BCE is still well defined, and Dice is skipped when nothing is valid.
        bce = masked_bce(logits[:, idx], target, valid)
        dice = masked_dice(logits[:, idx], target, valid) if valid.sum() > 0 else bce * 0
        parts[name] = float(bce.detach() + dice.detach())
        total = total + bce + dice
    return total / len(CLASS_ORDER), parts


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class ClassAccumulator:
    """Pixel-accumulated IoU/F1 per class, ignoring nodata."""

    def __init__(self):
        self.tp = np.zeros(len(CLASS_ORDER), dtype=np.int64)
        self.fp = np.zeros(len(CLASS_ORDER), dtype=np.int64)
        self.fn = np.zeros(len(CLASS_ORDER), dtype=np.int64)

    def add(self, probs: np.ndarray, target: np.ndarray) -> None:
        """probs: (C, H, W) in [0, 1]; target: (H, W) compact ids."""
        valid = target != 255
        for i in range(len(CLASS_ORDER)):
            pred = (probs[i] > 0.5) & valid
            true = (target == i + 1) & valid
            self.tp[i] += int((pred & true).sum())
            self.fp[i] += int((pred & ~true & valid).sum())
            self.fn[i] += int((~pred & true).sum())

    def report(self) -> dict[str, float]:
        out = {}
        ious = []
        for i, name in enumerate(CLASS_ORDER):
            union = self.tp[i] + self.fp[i] + self.fn[i]
            iou = self.tp[i] / union if union else float("nan")
            denom = 2 * self.tp[i] + self.fp[i] + self.fn[i]
            f1 = 2 * self.tp[i] / denom if denom else float("nan")
            out[f"{name}_iou"] = float(iou)
            out[f"{name}_f1"] = float(f1)
            if not np.isnan(iou):
                ious.append(iou)
        out["mean_iou"] = float(np.mean(ious)) if ious else float("nan")
        return out


@torch.no_grad()
def evaluate(model, dataset: LoveDATiles, device, batch_tiles: int = 4) -> dict:
    """Per-class metrics over whole tiles, using the app's tiled inference path."""
    model.eval()
    acc = ClassAccumulator()
    for i in range(len(dataset)):
        item = dataset[i]
        probs = predict_probs(model, item["image"], device, batch_size=batch_tiles)
        acc.add(probs, item["target"])
    return acc.report()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="./data/LoveDA")
    ap.add_argument("--train-split", default="Train")
    ap.add_argument("--val-split", default="Val")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="phase5_loveda")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto = CUDA when available. cpu leaves the GPU free "
                         "for another training job (Windows ignores "
                         "CUDA_VISIBLE_DEVICES).")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = Path(args.root)

    # Gate: prove the LoveDA label mapping against the imagery before training.
    # Reading the raw ids off by one trains the building head on background and
    # the water head on roads, and nothing in the loss curve would reveal it.
    mapping, failures = verify_mapping(root, args.train_split)
    if mapping:
        print("LoveDA label mapping check:")
        for i in sorted(mapping["share"]):
            print(f"  raw {i}: {mapping['share'][i]:6.2f} % of px | "
                  f"mean RGB {mapping['mean_rgb'][i]} | "
                  f"luminance {mapping['luminance'][i]:5.1f}")
        print(f"  building = raw 2 ({mapping['share'].get(2, 0):.2f} %), "
              f"water = raw 4 ({mapping['share'].get(4, 0):.2f} %)")
    if failures:
        print("\nLABEL MAPPING REJECTED:")
        for f in failures:
            print(f"  - {f}")
        return 2

    balance = class_balance(root, args.train_split)
    print("LoveDA class balance (sampled train pixels): "
          + ", ".join(f"{k}={v}%" for k, v in balance.items()))

    train_ds = LoveDABinary(root, args.train_split, crop=args.crop,
                            train=True, limit=args.limit_train)
    val_ds = LoveDATiles(root, args.val_split, limit=args.val_limit)
    print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"),
                          drop_last=len(train_ds) > args.batch)

    model = ResNetUNet(num_classes=len(CLASS_ORDER),
                       pretrained=args.pretrained).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: ResNet34-UNet with {len(CLASS_ORDER)} heads | {total:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(len(train_dl), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.15,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    run_dir = ROOT / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(vars(args), sort_keys=False))
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text(
            "epoch,lr,train_loss,building_loss,water_loss,"
            "val_building_iou,val_water_iou,val_mean_iou,"
            "val_building_f1,val_water_f1,seconds\n"
        )
    print(f"Run directory: {run_dir}")

    best_mean = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "building": 0.0, "water": 0.0}
        for batch in train_dl:
            images = batch["image"].to(device, non_blocking=True)
            targets = {name: batch[name].to(device, non_blocking=True)
                       for name in CLASS_ORDER}
            valid = batch["valid"].to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss, parts = multi_head_loss(logits, targets, valid)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["loss"] += float(loss.detach())
            running["building"] += parts["building"]
            running["water"] += parts["water"]

        n = max(len(train_dl), 1)
        metrics = evaluate(model, val_ds, device)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) "
              f"loss={running['loss'] / n:.4f} "
              f"(b={running['building'] / n:.3f} w={running['water'] / n:.3f}) | "
              f"val building IoU={metrics['building_iou']:.4f} "
              f"water IoU={metrics['water_iou']:.4f} "
              f"mean={metrics['mean_iou']:.4f}")

        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{lr_now:.6g},{running['loss'] / n:.4f},"
                f"{running['building'] / n:.4f},{running['water'] / n:.4f},"
                f"{metrics['building_iou']:.4f},{metrics['water_iou']:.4f},"
                f"{metrics['mean_iou']:.4f},"
                f"{metrics['building_f1']:.4f},{metrics['water_f1']:.4f},"
                f"{elapsed:.0f}\n"
            )

        if metrics["mean_iou"] > best_mean:
            best_mean = metrics["mean_iou"]
            payload = {"model": model.state_dict(), "head_order": list(CLASS_ORDER),
                       "epoch": epoch, "metrics": metrics, "args": vars(args)}
            # Write-then-rename for both copies: the app loads the `checkpoints/`
            # one directly, so a kill mid-save must not be able to truncate it.
            for dest in (run_dir / "best.pth",
                         ROOT / "checkpoints" / "phase5_multihead_best.pth"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".pth.tmp")
                torch.save(payload, tmp)
                os.replace(tmp, dest)
            print(f"  ✓ Best model saved (mean IoU={best_mean:.4f}) -> "
                  f"phase5_multihead_best.pth")

    print(f"Finished. Best val mean IoU = {best_mean:.4f}")
    if best_mean < 0.5:
        print("NOTE: this is below a usable threshold — report it honestly rather "
              "than shipping the checkpoint as production quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
