"""LoveDA -> binary building/water targets for the two-head model.

LoveDA ships 1024x1024 PNG tiles at 0.3 m GSD with 7 land-cover classes plus an
ignore index. This module keeps only what the project needs and, critically,
keeps track of what is *unknown* instead of silently calling it background.

The raw ids are NOT the class order people assume. From the official dataset
definition (`mmseg/datasets/loveda.py`): "In segmentation map annotation for
LoveDA, 0 is the ignore index. ``reduce_zero_label`` should be set to True."
With zero-label reduction the classes are

    raw 0  ignore        (pure black padding in the imagery)
    raw 1  background
    raw 2  building      <- the building head
    raw 3  road
    raw 4  water         <- the water head
    raw 5  barren
    raw 6  forest
    raw 7  agricultural

Reading the ids as 0=background, 1=building, 3=water trains the building head on
*background* and the water head on *roads*, and is the exact mistake this module
made before it was checked against the pixels.

The dataset therefore emits a single uint8 target where

    0   = background      (valid negative for both heads)
    1   = building        (positive for head 0, negative for head 1)
    2   = water           (positive for head 1, negative for head 0)
    255 = ignore          (excluded from both heads' loss)

Augmentation runs on image + that one target channel, so the three masks can
never drift out of alignment.

Layout after unzipping the official archives:

    <root>/Train/Urban/images_png/*.png
    <root>/Train/Urban/masks_png/*.png
    <root>/Train/Rural/...
    <root>/Val/Urban/...   <root>/Val/Rural/...
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# LoveDA raw class ids (see module docstring; verified against image pixels and
# against mmseg's LoveDADataset, which documents raw 0 as the ignore index).
L_IGNORE, L_BUILDING, L_WATER = 0, 2, 4

# Every raw id that is a genuine negative for both heads. Anything outside this
# set and the two positives would be an unexpected label and is therefore
# treated as ignore rather than silently counted as background.
L_NEGATIVES = (1, 3, 5, 6, 7)     # background, road, barren, forest, agricultural

# Our compact target values
T_BACKGROUND, T_BUILDING, T_WATER, T_IGNORE = 0, 1, 2, 255

CLASS_NAMES = ("building", "water")


def to_compact(raw: np.ndarray) -> np.ndarray:
    """LoveDA class ids -> {0 background, 1 building, 2 water, 255 ignore}.

    An id that is neither a known negative nor a positive is mapped to ignore
    (not to background), so a surprise label can never be trained as "not
    building and not water".
    """
    out = np.full(raw.shape, T_IGNORE, dtype=np.uint8)
    for value in L_NEGATIVES:
        out[raw == value] = T_BACKGROUND
    out[raw == L_BUILDING] = T_BUILDING
    out[raw == L_WATER] = T_WATER
    out[raw == L_IGNORE] = T_IGNORE
    return out


def to_heads(compact: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact target -> (building, water, valid) float masks for the loss."""
    valid = compact != T_IGNORE
    building = (compact == T_BUILDING).astype(np.float32)
    water = (compact == T_WATER).astype(np.float32)
    return building, water, valid.astype(np.float32)


def find_tiles(root: Path, split: str) -> list[tuple[Path, Path]]:
    """All (image, mask) pairs for a split, across the Urban/Rural subfolders."""
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"LoveDA split not found: {split_dir}\n"
            f"Expected {split_dir}/Urban and {split_dir}/Rural after unzipping."
        )
    pairs = []
    for area in sorted(split_dir.iterdir()):
        img_dir = area / "images_png"
        mask_dir = area / "masks_png"
        if not (img_dir.is_dir() and mask_dir.is_dir()):
            continue
        for img in sorted(img_dir.glob("*.png")):
            mask = mask_dir / img.name
            if mask.is_file():
                pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs under {split_dir}")
    return pairs


def train_transforms(crop: int) -> A.Compose:
    return A.Compose([
        A.RandomCrop(crop, crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Mild photometric jitter only: LoveDA is 0.3 m aerial RGB and the water
        # head must survive specular glint without learning "bright = water".
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class LoveDABinary(Dataset):
    """One tile at a time: image tensor + (building, water, valid) masks."""

    def __init__(self, root: str | Path, split: str = "Train",
                 crop: int = 512, train: bool = True,
                 limit: int | None = None):
        self.root = Path(root)
        self.crop = crop
        self.train = train
        self.pairs = find_tiles(self.root, split)
        if limit:
            self.pairs = self.pairs[:limit]
        self.tf = train_transforms(crop) if train else eval_transforms()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        compact = to_compact(np.array(Image.open(mask_path), dtype=np.uint8))

        if not self.train and min(image.shape[:2]) < self.crop:
            raise ValueError(f"{img_path.name} is smaller than the {self.crop}px crop")

        out = self.tf(image=image, mask=compact)
        image_t = out["image"]
        compact_t = out["mask"].numpy() if torch.is_tensor(out["mask"]) else out["mask"]
        compact_t = np.asarray(compact_t, dtype=np.uint8)

        building, water, valid = to_heads(compact_t)
        return {
            "image": image_t,
            "building": torch.from_numpy(building),
            "water": torch.from_numpy(water),
            "valid": torch.from_numpy(valid),
            "name": img_path.name,
        }


class LoveDATiles(Dataset):
    """Full tile as raw uint8 HWC + compact mask. Used for tiled evaluation.

    Deliberately unnormalised: the evaluation path feeds these straight into
    `rs_core.inference.predict_probs`, which is the exact code the app uses, so
    reported numbers and deployed behaviour cannot diverge.
    """

    def __init__(self, root: str | Path, split: str = "Val", limit: int | None = None):
        self.pairs = find_tiles(Path(root), split)
        if limit:
            self.pairs = self.pairs[:limit]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        compact = to_compact(np.array(Image.open(mask_path), dtype=np.uint8))
        return {"image": image, "target": compact, "name": img_path.name}


def class_balance(root: str | Path, split: str = "Train", sample: int = 300) -> dict:
    """Pixel share per compact class, for loss weights and honest reporting."""
    pairs = find_tiles(Path(root), split)[:sample]
    counts = np.zeros(4, dtype=np.int64)   # bg, building, water, ignore
    for _, mask_path in pairs:
        compact = to_compact(np.array(Image.open(mask_path), dtype=np.uint8))
        counts[0] += int((compact == T_BACKGROUND).sum())
        counts[1] += int((compact == T_BUILDING).sum())
        counts[2] += int((compact == T_WATER).sum())
        counts[3] += int((compact == T_IGNORE).sum())
    total = max(int(counts.sum()), 1)
    names = ("background", "building", "water", "ignore")
    return {n: round(100.0 * c / total, 3) for n, c in zip(names, counts)}


def verify_mapping(root: str | Path, split: str = "Train",
                   sample: int = 40) -> tuple[dict, list[str]]:
    """Check the raw-id mapping against the imagery itself.

    Documentation is not evidence. Raw 0 must land on the pure-black padding of
    the image, buildings must be brighter than water, and water must be the
    blue-leaning class. Reading the ids off by one (building=1, water=3) puts the
    building head on background and the water head on roads, and every one of
    these checks fails when that happens.

    Returns (measurements, failures). `failures` is empty when the mapping is
    corroborated by the pixels.
    """
    pairs = find_tiles(Path(root), split)
    # Spread the sample across the whole split: find_tiles() returns Rural before
    # Urban, so taking the first N tiles would measure only one landscape type.
    if len(pairs) > sample:
        stride = len(pairs) / sample
        pairs = [pairs[int(i * stride)] for i in range(sample)]
    failures: list[str] = []

    acc = np.zeros((8, 3), dtype=np.float64)      # raw id -> summed RGB
    npx = np.zeros(8, dtype=np.int64)

    for img_path, mask_path in pairs:
        raw = np.array(Image.open(mask_path))
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64)
        for value in np.unique(raw):
            v = int(value)
            if not 0 <= v <= 7:
                failures.append(f"unexpected raw label id {v} in {mask_path.name}")
                continue
            sel = raw == value
            npx[v] += int(sel.sum())
            acc[v] += rgb[sel].sum(axis=0)

    total = int(npx.sum())
    if total == 0:
        return {}, ["no labelled pixels found"]

    mean = np.divide(acc, npx[:, None], out=np.zeros_like(acc),
                     where=npx[:, None] > 0)
    lum = 0.299 * mean[:, 0] + 0.587 * mean[:, 1] + 0.114 * mean[:, 2]
    share = 100.0 * npx / total

    report = {
        "tiles": len(pairs),
        "share": {i: round(float(share[i]), 2) for i in range(8) if npx[i]},
        "mean_rgb": {i: tuple(round(float(c), 1) for c in mean[i])
                     for i in range(8) if npx[i]},
        "luminance": {i: round(float(lum[i]), 1) for i in range(8) if npx[i]},
    }

    # 1. Raw 0 is the ignore index and must be the black padding of the imagery.
    if npx[0]:
        if lum[0] > 12:
            failures.append(
                f"raw 0 is documented as ignore/padding but its image pixels have "
                f"luminance {lum[0]:.1f}; it is terrain, not padding")
    else:
        failures.append("raw 0 never appears; expected it on black padding")

    # 2. Building (raw 2) and water (raw 4) must both be present and plausible.
    for value, name, lo, hi in ((L_BUILDING, "building", 1.0, 35.0),
                                (L_WATER, "water", 0.5, 25.0)):
        s = float(share[value])
        if not lo <= s <= hi:
            failures.append(
                f"raw {value} ({name}) covers {s:.2f} % of pixels, outside the "
                f"plausible {lo}-{hi} % band — check the label mapping")

    # 3. The two heads must not be the same thing. Buildings are bright, low
    #    saturation surfaces; water is dark and blue-leaning.
    if npx[L_BUILDING] and npx[L_WATER]:
        b, w = lum[L_BUILDING], lum[L_WATER]
        if not b > w + 15:
            failures.append(
                f"building luminance {b:.1f} is not clearly brighter than water "
                f"{w:.1f}; ids {L_BUILDING}/{L_WATER} probably do not mean "
                f"building/water")
        b_lead = mean[L_WATER, 2] - (mean[L_WATER, 0] + mean[L_WATER, 1]) / 2
        if b_lead <= 0:
            failures.append(
                f"water (raw {L_WATER}) has no blue lead ({b_lead:+.1f}); it is "
                f"not water")

    return report, failures

