"""Sanity gate for the SatMAE++ encoder.

Question this answers: does the frozen encoder carry *spatial* information?
Before the pos_embed fix it did not (position table was all zeros), which is why
phases 2-4 underperformed a plain ResNet34 U-Net.

Two metrics per pyramid level:
  spatial_std  : mean per-channel std across spatial positions (higher = more structure)
  pos_cosine   : mean pairwise cosine similarity between feature vectors at
                 different spatial positions (lower = more location-aware)

Usage:
    python tools/probe_encoder.py --n 2
    python tools/probe_encoder.py --n 2 --mode drop   # reproduce the old behaviour
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2_satmae.satmae_encoder import (  # noqa: E402
    SatMAEPlusEncoder,
    interpolate_pos_embed,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_images(img_dir: Path, n: int, image_size: int = 512) -> torch.Tensor:
    paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    )[:n]
    if not paths:
        raise FileNotFoundError(f"no images in {img_dir}")
    batch = []
    for p in paths:
        arr = np.array(Image.open(p).convert("RGB").resize(
            (image_size, image_size), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        batch.append(torch.from_numpy(arr).permute(2, 0, 1))
    print(f"[probe] using {[p.name for p in paths]}")
    return torch.stack(batch)


def build_encoder(mode: str, ckpt: Path, device: torch.device) -> SatMAEPlusEncoder:
    encoder = SatMAEPlusEncoder(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        extract_layers=[5, 11, 17, 23], freeze=True,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    cleaned = {}
    for k, v in state.items():
        k = k[len("model."):] if k.startswith("model.") else k
        if k.startswith("decoder"):
            continue
        cleaned[k] = v

    if mode == "interp":
        if cleaned["pos_embed"].shape != encoder.state_dict()["pos_embed"].shape:
            cleaned["pos_embed"] = interpolate_pos_embed(cleaned["pos_embed"], encoder.grid_size)
    elif mode == "drop":
        cleaned.pop("pos_embed", None)          # legacy (buggy) behaviour
    else:
        raise ValueError(mode)

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    print(f"[probe] mode={mode} missing={len(missing)} unexpected={len(unexpected)} "
          f"pos_abs_sum={float(encoder.pos_embed.abs().sum()):.3f}")
    return encoder.to(device).eval()


@torch.no_grad()
def measure(encoder: SatMAEPlusEncoder, batch: torch.Tensor) -> None:
    feats = encoder(batch)
    print(f"  grid={encoder.grid_size}x{encoder.grid_size}  "
          f"{feats[0].shape[0]} images  {feats[0].shape[1]} channels")
    for lvl, f in enumerate(feats):
        flat = f.flatten(2)                                  # (B, C, N)
        spatial_std = f.std(dim=(2, 3)).mean().item()
        unit = torch.nn.functional.normalize(flat, dim=1)    # (B, C, N)
        sim = unit.transpose(1, 2) @ unit                    # (B, N, N)
        n = sim.shape[-1]
        diag = sim.diagonal(dim1=1, dim2=2).sum(dim=1)       # (B,)
        off_diag = ((sim.sum(dim=(1, 2)) - diag) / (n * (n - 1))).mean()
        print(f"  level {lvl} (block {encoder.extract_layers[lvl]:>2}): "
              f"spatial_std={spatial_std:.5f}  pos_cosine={off_diag.item():.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./weights/satmae_pp.pth")
    ap.add_argument("--img_dir", default="./data/WHU/val/image")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--mode", choices=["interp", "drop", "both"], default="both")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}")
    batch = load_images(Path(args.img_dir), args.n).to(device)

    modes = ["drop", "interp"] if args.mode == "both" else [args.mode]
    for mode in modes:
        encoder = build_encoder(mode, Path(args.ckpt), device)
        measure(encoder, batch)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
