"""Linear probe on frozen SatMAE++ features: is the encoder useful for dense prediction?

A 1x1 conv (linear classifier) is trained on *frozen* level-3 features at 32x32
patch resolution, then evaluated against ground-truth masks downsampled to the
same grid. This isolates the encoder from the decoder: if the encoder is good,
even a linear head gets a usable IoU.

Compares the two pos_embed policies:
    drop   : legacy behaviour (position table deleted -> all zeros)
    interp : fixed behaviour (14x14 -> 32x32 bicubic interpolation)

Usage:
    python tools/probe_linear.py --mode both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase2_satmae.satmae_encoder import (  # noqa: E402
    SatMAEPlusEncoder,
    interpolate_pos_embed,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_pairs(img_dir: Path, mask_dir: Path, n: int) -> list[tuple[Path, Path]]:
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg"})[:n]
    return [(p, mask_dir / p.name) for p in imgs]


def load_batch(pairs, image_size: int = 512):
    images, masks = [], []
    for img_p, mask_p in pairs:
        img = np.array(Image.open(img_p).convert("RGB").resize(
            (image_size, image_size), Image.BILINEAR), dtype=np.float32) / 255.0
        img = (img - MEAN) / STD
        m = np.array(Image.open(mask_p).convert("L"), dtype=np.float32) / 255.0
        m = np.array(Image.fromarray(m).resize((image_size // 16, image_size // 16),
                                               Image.BOX))
        images.append(torch.from_numpy(img).permute(2, 0, 1))
        masks.append(torch.from_numpy((m > 0.5).astype(np.float32)))
    return torch.stack(images), torch.stack(masks)


def build_encoder(mode: str, ckpt: Path, device) -> SatMAEPlusEncoder:
    enc = SatMAEPlusEncoder(img_size=512, patch_size=16, embed_dim=1024, depth=24,
                            num_heads=16, extract_layers=[5, 11, 17, 23], freeze=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    cleaned = {}
    for k, v in state.items():
        k = k[len("model."):] if k.startswith("model.") else k
        if k.startswith("decoder"):
            continue
        cleaned[k] = v
    if mode == "interp":
        if cleaned["pos_embed"].shape != enc.state_dict()["pos_embed"].shape:
            cleaned["pos_embed"] = interpolate_pos_embed(cleaned["pos_embed"], enc.grid_size)
    else:
        cleaned.pop("pos_embed", None)
    enc.load_state_dict(cleaned, strict=False)
    return enc.to(device).eval()


@torch.no_grad()
def extract(enc, pairs, device, batch_size=6, level=3):
    feats, masks = [], []
    for i in range(0, len(pairs), batch_size):
        imgs, ms = load_batch(pairs[i:i + batch_size])
        f = enc(imgs.to(device))[level]
        feats.append(f.cpu())
        masks.append(ms)
    return torch.cat(feats), torch.cat(masks)


def linear_probe(train_f, train_m, val_f, val_m, device, steps=400, lr=1e-3):
    head = nn.Conv2d(train_f.shape[1], 1, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    tf, tm = train_f.to(device), train_m.to(device)
    vf, vm = val_f.to(device), val_m.to(device)
    n = tf.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (min(32, n),), device=device)
        logits = head(tf[idx]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, tm[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = (torch.sigmoid(head(vf).squeeze(1)) > 0.5).float()
        inter = (p * vm).sum()
        union = ((p + vm) > 0).sum()
        iou = (inter / union.clamp(min=1)).item()
        f1 = (2 * inter / (p.sum() + vm.sum()).clamp(min=1)).item()
    return iou, f1, loss.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./weights/satmae_pp.pth")
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--n_val", type=int, default=40)
    ap.add_argument("--mode", choices=["drop", "interp", "both"], default="both")
    ap.add_argument("--level", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("./data/WHU")
    train = list_pairs(root / "train/image", root / "train/label", args.n_train)
    val = list_pairs(root / "val/image", root / "val/label", args.n_val)
    print(f"[probe] device={device} train={len(train)} val={len(val)} level={args.level}")

    for mode in (["drop", "interp"] if args.mode == "both" else [args.mode]):
        enc = build_encoder(mode, Path(args.ckpt), device)
        tr_f, tr_m = extract(enc, train, device, level=args.level)
        va_f, va_m = extract(enc, val, device, level=args.level)
        print(f"[probe] mode={mode} pos_abs_sum={float(enc.pos_embed.abs().sum()):.1f} "
              f"feature_std={tr_f.std().item():.3f}")
        iou, f1, loss = linear_probe(tr_f, tr_m, va_f, va_m, device)
        print(f"  >>> LINEAR-PROBE val IoU={iou:.4f}  F1={f1:.4f}  (final loss {loss:.4f})")
        del enc, tr_f, tr_m, va_f, va_m
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
