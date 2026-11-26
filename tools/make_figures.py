"""Build the README figures from a single WHU test tile.

The README shows a 2x2 panel: input, ground truth, predicted probability and the
overlay the app produces. This script regenerates that panel so the figure in the
README is reproducible rather than a screenshot of unknown provenance.

It does not run the model itself. Run `predict.py` first, then point this at the
directory it wrote:

    venv\\Scripts\\python.exe predict.py --phase 1 ^
        --input data\\WHU\\test\\image\\106.tif --output results\\demo --overlay
    venv\\Scripts\\python.exe tools\\make_figures.py --tile 106 --pred results\\demo

Output: docs/building_example.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PANEL = 512
PAD = 12
LABEL_H = 44
BG = (18, 18, 20)
FG = (225, 225, 225)
BUILDING = (0, 255, 100)          # matches rs_core.visualize
TRUTH = (255, 214, 0)

# Pillow's default bitmap font is ~11 px, which is unreadable at GitHub's render
# width. Try the usual system fonts, fall back to the default if none exist.
FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def load_font(size: int = 22) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def to_panel(img: np.ndarray) -> Image.Image:
    """Any array -> RGB panel of exactly PANEL x PANEL."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return Image.fromarray(img.astype(np.uint8)).convert("RGB").resize(
        (PANEL, PANEL), Image.NEAREST
    )


def colour_truth(mask: np.ndarray) -> np.ndarray:
    """Binary ground truth -> dark background with yellow buildings."""
    out = np.full((*mask.shape, 3), 30, dtype=np.uint8)
    out[mask] = TRUTH
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="106", help="WHU test tile stem")
    ap.add_argument("--pred", default="results/demo", help="predict.py output dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="docs/building_example.png")
    args = ap.parse_args()

    img_path = ROOT / "data" / "WHU" / args.split / "image" / f"{args.tile}.tif"
    lbl_path = ROOT / "data" / "WHU" / args.split / "label" / f"{args.tile}.tif"
    pred_dir = ROOT / args.pred
    overlay_path = pred_dir / f"{args.tile}_overlay.png"
    prob_path = pred_dir / f"{args.tile}_prob.png"

    for p in (img_path, lbl_path, overlay_path, prob_path):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    rgb = np.array(Image.open(img_path).convert("RGB"))
    truth = np.array(Image.open(lbl_path)) > 0
    overlay = np.array(Image.open(overlay_path).convert("RGB"))
    prob = np.array(Image.open(prob_path).convert("L"))

    panels = [
        (to_panel(rgb), "Input RGB (WHU, 0.30 m/px)"),
        (to_panel(colour_truth(truth)), "Ground truth"),
        (to_panel(prob), "Predicted probability"),
        (to_panel(overlay), "Overlay (phase 1, IoU 0.884)"),
    ]

    w = PANEL * 2 + PAD * 3
    h = (PANEL + LABEL_H) * 2 + PAD * 3
    sheet = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(sheet)
    font = load_font(22)

    for i, (panel, caption) in enumerate(panels):
        col, row = i % 2, i // 2
        x = PAD + col * (PANEL + PAD)
        y = PAD + row * (PANEL + LABEL_H + PAD)
        sheet.paste(panel, (x, y))
        draw.text((x + 2, y + PANEL + 10), caption, fill=FG, font=font)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, optimize=True)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KiB)")
    print(f"  tile {args.tile}: building coverage "
          f"{100 * truth.mean():.1f}% (ground truth)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
