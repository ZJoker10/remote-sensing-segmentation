"""Rendering helpers: overlay, mask, probability heatmap."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _colorize(rgb: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    if mask.any():
        for c, val in enumerate(color):
            out[:, :, c] = np.where(
                mask, out[:, :, c] * (1 - alpha) + val * alpha, out[:, :, c]
            )
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    classes: dict[str, dict],
) -> np.ndarray:
    """Paint each class colour over the source image."""
    out = rgb
    for name, mask in masks.items():
        meta = classes.get(name)
        if meta is None or mask is None:
            continue
        out = _colorize(out, mask, meta["color"], meta["alpha"])
    return out


def mask_png(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")


def prob_png(prob: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8), mode="L")
