"""Verify the extracted LoveDA tree: pairing, geometry, label ids, class balance.

Read-only. Prints measured counts; no assumptions are hard-coded beyond the
documented LoveDA label ids (0 background, 1 building, 2 road, 3 water,
4 barren, 5 forest, 6 agriculture, 255 nodata).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "LoveDA"

sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLASSES = {
    0: "ignore", 1: "background", 2: "building", 3: "road",
    4: "water", 5: "barren", 6: "forest", 7: "agricultural",
}

problems: list[str] = []
counts = Counter()
pixel_counts = Counter()
sizes = Counter()
sample = 400          # masks sampled for the pixel statistics


def check(ok: bool, msg: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        problems.append(msg)


for split in ("Train", "Val"):
    print(f"\n=== {split} ===")
    for area in ("Urban", "Rural"):
        img_dir = BASE / split / area / "images_png"
        mask_dir = BASE / split / area / "masks_png"
        if not img_dir.is_dir() or not mask_dir.is_dir():
            check(False, f"{split}/{area}: missing directory")
            continue

        imgs = sorted(img_dir.glob("*.png"))
        masks = sorted(mask_dir.glob("*.png"))
        names_i = {p.name for p in imgs}
        names_m = {p.name for p in masks}

        check(len(imgs) > 0, f"{split}/{area}: {len(imgs)} images, {len(masks)} masks")
        check(names_i == names_m,
              f"{split}/{area}: image and mask filenames match exactly")

        # Geometry of every image/mask pair; label ids over a sample of masks.
        for i, path in enumerate(imgs):
            try:
                with Image.open(path) as im:
                    sizes[(split, area, im.size, im.mode)] += 1
            except Exception as exc:                      # noqa: BLE001
                check(False, f"{split}/{area}/{path.name}: unreadable image ({exc})")
                break

            if i >= sample:
                continue
            mask_path = mask_dir / path.name
            if not mask_path.is_file():
                continue
            with Image.open(mask_path) as m:
                arr = np.array(m)
            for value, count in zip(*np.unique(arr, return_counts=True)):
                pixel_counts[(split, int(value))] += int(count)

        counts[(split, area)] = len(imgs)

print("\n=== geometry ===")
for (split, area, size, mode), n in sorted(sizes.items()):
    check(size == (1024, 1024) and mode in ("RGB", "RGBA"),
          f"{split}/{area}: {n} images are {size[0]}x{size[1]} {mode}")

print("\n=== label values (sampled masks) ===")
seen_ids = sorted({v for (_, v) in pixel_counts})
unknown = [v for v in seen_ids if v not in CLASSES]
check(not unknown, f"all label ids are known ({seen_ids}); unknown={unknown}")

# Prove the production mapping against the imagery, not against documentation.
print("\n=== LoveDA label mapping (verified against image pixels) ===")
from phase5_multihead.loveda import verify_mapping      # noqa: E402

report, failures = verify_mapping(BASE, "Train", sample=40)
for i in sorted(report.get("share", {})):
    print(f"  raw {i} {CLASSES.get(i, '?'):12s} {report['share'][i]:6.2f} % | "
          f"RGB {report['mean_rgb'][i]} | luminance {report['luminance'][i]:5.1f}")
check(not failures, "raw-id mapping corroborated by the pixels"
      + ("" if not failures else ": " + "; ".join(failures)))
check(report.get("share", {}).get(2, 0) > 1.0,
      f"raw 2 (building) is a substantial class ({report.get('share', {}).get(2)} %)")
check(report.get("share", {}).get(4, 0) > 0.5,
      f"raw 4 (water) is present ({report.get('share', {}).get(4)} %)")


for split in ("Train", "Val"):
    total = sum(c for (s, _), c in pixel_counts.items() if s == split)
    if not total:
        continue
    print(f"\n  {split} pixel shares ({total:,} px sampled):")
    for value in seen_ids:
        c = pixel_counts.get((split, value), 0)
        print(f"    {value:3d} {CLASSES.get(value, '?'):12s} {c / total * 100:6.2f} %")

print("\n=== totals ===")
train_total = counts[("Train", "Urban")] + counts[("Train", "Rural")]
val_total = counts[("Val", "Urban")] + counts[("Val", "Rural")]
print(f"  Train: {train_total} tiles "
      f"(Urban {counts[('Train', 'Urban')]}, Rural {counts[('Train', 'Rural')]})")
print(f"  Val:   {val_total} tiles "
      f"(Urban {counts[('Val', 'Urban')]}, Rural {counts[('Val', 'Rural')]})")

check(train_total > 2000, f"Train has {train_total} tiles (expected ~2522)")
check(val_total > 1500, f"Val has {val_total} tiles (expected ~1669)")
check(not (BASE / "Test").exists() or not any((BASE / "Test").rglob("*.png")),
      "no Test split present (LoveDA test labels are withheld, so it must not be used)")

# Both target classes must actually be present in the training split.
train_px = sum(c for (s, _), c in pixel_counts.items() if s == "Train")
for value, label in ((2, "building"), (4, "water")):
    share = pixel_counts.get(("Train", value), 0) / max(train_px, 1) * 100
    check(share > 0.5, f"Train contains {label} pixels ({share:.2f} % of sampled px)")

print(f"\n{'ALL CHECKS PASSED' if not problems else f'{len(problems)} PROBLEM(S)'}")
sys.exit(1 if problems else 0)
