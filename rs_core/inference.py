"""Tiled inference for arbitrarily large aerial images.

The models consume exactly TILE x TILE pixels, so larger scenes are processed as
overlapping windows and stitched in probability space with a raised-cosine weight
so seams are invisible.

Works for any number of output channels: a single-class checkpoint returns
(C=1) and the two-head building/water model returns (C=2) in the order
CLASS_ORDER below.
"""

from __future__ import annotations

import numpy as np
import torch

TILE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Channel order produced by multi-class checkpoints.
CLASS_ORDER = ("building", "water")


def _to_tensor(tiles: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) normalised float tensor."""
    x = tiles.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    return x.to(device, non_blocking=True)


def _window_starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def _blend_window(tile: int) -> np.ndarray:
    """Raised-cosine 2-D window, floored so no pixel gets zero weight."""
    w = np.hanning(tile).astype(np.float32)
    w = np.maximum(w, 1e-3)
    return np.outer(w, w)


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    tile: int = TILE,
    overlap: int = 64,
    batch_size: int = 4,
) -> np.ndarray:
    """Return a float32 (C, H, W) probability map for one RGB uint8 image.

    Args:
        model:       segmentation model in eval mode, output (B, C, H, W) logits
        image_rgb:   (H, W, 3) uint8
        tile:        network input size
        overlap:     pixels shared between neighbouring windows
        batch_size:  windows per forward pass
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {image_rgb.shape}")
    if tile != TILE:
        raise ValueError(f"models are trained at {TILE}px; got tile={tile}")

    h, w = image_rgb.shape[:2]
    stride = tile - overlap

    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h or pad_w:
        image_rgb = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    h_pad, w_pad = image_rgb.shape[:2]

    ys = _window_starts(h_pad, tile, stride)
    xs = _window_starts(w_pad, tile, stride)
    window = _blend_window(tile)
    coords = [(y, x) for y in ys for x in xs]

    prob = weight = None
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        tiles = np.stack([image_rgb[y:y + tile, x:x + tile] for y, x in chunk])
        logits = model(_to_tensor(tiles, device))
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        if prob is None:
            channels = probs.shape[1]
            prob = np.zeros((channels, h_pad, w_pad), dtype=np.float32)
            weight = np.zeros((h_pad, w_pad), dtype=np.float32)
        for (y, x), p in zip(chunk, probs):
            prob[:, y:y + tile, x:x + tile] += p * window
            weight[y:y + tile, x:x + tile] += window

    prob /= np.maximum(weight, 1e-6)[None]
    return prob[:, :h, :w]


@torch.no_grad()
def predict_prob(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    tile: int = TILE,
    overlap: int = 64,
    batch_size: int = 4,
) -> np.ndarray:
    """Single-class convenience wrapper -> (H, W) probability."""
    probs = predict_probs(model, image_rgb, device, tile, overlap, batch_size)
    if probs.shape[0] != 1:
        raise ValueError(
            f"model predicts {probs.shape[0]} channels; use predict_probs() instead"
        )
    return probs[0]
