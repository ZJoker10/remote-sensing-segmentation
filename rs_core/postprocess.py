"""Mask cleanup, instance extraction and per-building statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def clean_mask(
    mask: np.ndarray,
    min_area_px: int = 20,
    fill_holes: bool = True,
) -> np.ndarray:
    """Remove specks and optionally close interior holes (roof courtyards)."""
    mask = mask.astype(bool)
    if min_area_px > 0:
        labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n:
            sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
            keep = np.isin(labeled, np.flatnonzero(sizes >= min_area_px) + 1)
            mask = keep
    if fill_holes and mask.any():
        mask = ndimage.binary_fill_holes(mask)
    return mask


def analyse(
    prob: np.ndarray,
    threshold: float = 0.5,
    gsd_m: float | None = None,
    min_area_m2: float = 20.0,
    max_area_ratio: float = 0.9,
) -> dict[str, Any]:
    """Turn a probability map into a cleaned mask plus summary statistics.

    Args:
        prob:          (H, W) float probability in [0, 1]
        threshold:     decision threshold
        gsd_m:         ground sample distance in metres/pixel (for m^2 units)
        min_area_m2:   drop instances smaller than this
        max_area_ratio: drop instances covering more than this share of the scene
    """
    px_area_m2 = (gsd_m ** 2) if gsd_m else None
    min_area_px = int(min_area_m2 / px_area_m2) if px_area_m2 else 20

    raw = prob > threshold
    mask = clean_mask(raw, min_area_px=min_area_px)

    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    total_px = int(mask.sum())
    scene_px = mask.size
    instances = []
    for idx in range(1, n + 1):
        comp = labeled == idx
        area_px = int(comp.sum())
        if area_px == 0:
            continue
        if area_px / scene_px > max_area_ratio:
            continue
        ys, xs = np.nonzero(comp)
        instances.append(
            {
                "id": idx,
                "area_px": area_px,
                "area_m2": round(area_px * px_area_m2, 1) if px_area_m2 else None,
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "mean_conf": round(float(prob[comp].mean()), 3),
                "max_conf": round(float(prob[comp].max()), 3),
            }
        )

    return {
        "mask": mask,
        "instances": instances,
        "stats": {
            "instances": len(instances),
            "coverage_pct": round(100.0 * total_px / scene_px, 2),
            "total_area_m2": round(total_px * px_area_m2, 1) if px_area_m2 else None,
            "mean_confidence": round(float(prob[mask].mean()), 3) if total_px else 0.0,
            "gsd_m": gsd_m,
            "px_area_m2": round(px_area_m2, 4) if px_area_m2 else None,
            # Height / volume arrive with the nDSM head (plan section 5).
            "height_m": None,
            "volume_m3": None,
        },
    }
