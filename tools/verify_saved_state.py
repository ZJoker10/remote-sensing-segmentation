"""Verify the saved state is resumable and the LoveDA mapping is correct.

Run this before resuming work. It checks:
  * runs/phase3_fixed/best.pth loads and reports the epoch/metrics it holds
  * the LoveDA compact-target mapping puts building on raw 2 and water on raw 4
  * a real LoveDA tile converts to the expected target distribution
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from phase5_multihead.loveda import (  # noqa: E402
    L_BUILDING, L_IGNORE, L_WATER,
    T_BACKGROUND, T_BUILDING, T_IGNORE, T_WATER,
    find_tiles, to_compact,
)

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


print("=== LoveDA raw-id mapping ===")
check((L_IGNORE, L_BUILDING, L_WATER) == (0, 2, 4),
      f"raw ids are ignore=0, building=2, water=4 (got "
      f"{L_IGNORE}, {L_BUILDING}, {L_WATER})")

raw = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint8)
compact = to_compact(raw)
expected = np.array([
    [T_IGNORE, T_BACKGROUND, T_BUILDING, T_BACKGROUND],
    [T_WATER, T_BACKGROUND, T_BACKGROUND, T_BACKGROUND],
], dtype=np.uint8)
check(np.array_equal(compact, expected),
      "to_compact maps raw 0->ignore, 1/3/5/6/7->background, 2->building, 4->water")

# An unexpected id must become ignore, never background.
check(to_compact(np.array([[200]], dtype=np.uint8))[0, 0] == T_IGNORE,
      "an unknown raw id becomes ignore, not background")

print("\n=== mapping on a real LoveDA tile ===")
pairs = find_tiles(ROOT / "data" / "LoveDA", "Train")
if pairs:
    img_path, mask_path = pairs[len(pairs) // 2]
    raw = np.array(Image.open(mask_path))
    compact = to_compact(raw)
    total = compact.size
    shares = {name: 100.0 * int((compact == value).sum()) / total
              for name, value in (("building", T_BUILDING),
                                  ("water", T_WATER),
                                  ("ignore", T_IGNORE))}
    print(f"  {mask_path.parent.parent.name}/{mask_path.name}: "
          + ", ".join(f"{k}={v:.2f}%" for k, v in shares.items()))
    check(0.0 <= shares["building"] <= 60.0, "building share is plausible")
    check(0.0 <= shares["water"] <= 60.0, "water share is plausible")
else:
    check(False, "LoveDA Train tiles found")

print("\n=== phase 3 checkpoint (paused retrain) ===")
ckpt_path = ROOT / "runs" / "phase3_fixed" / "best.pth"
if ckpt_path.is_file():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    val_iou = ckpt.get("val_iou")
    print(f"  epoch={epoch} | val IoU={val_iou}")
    check(epoch is not None, "checkpoint records its epoch")
    check(bool(val_iou) and val_iou > 0.7,
          f"paused checkpoint is already usable (val IoU={val_iou})")
    check("optimizer" in ckpt,
          "optimizer state is stored (weights resume; the cosine schedule does not)")

    model_state = ckpt.get("model", ckpt)
    encoder = model_state.get("encoder.pos_embed")
    if encoder is not None:
        s = float(encoder.abs().sum())
        print(f"  encoder.pos_embed abs sum = {s:.4f} (0 would mean still broken)")
        check(s > 0, "pos_embed is non-zero: the interpolation fix survived")

    # The decisive test: the saved weights must fit a freshly built phase-3
    # model. A strict mismatch here would only surface ~7 hours into a resume.
    import yaml
    from phase3_swin_unet.swin_unet_decoder import SatMAESwinUNet

    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    p2, p3 = cfg["phase2"], cfg["phase3"]
    model = SatMAESwinUNet(
        img_size       = cfg["data"]["image_size"],
        patch_size     = p2["patch_size"],
        num_classes    = 1,
        window_size    = p3["swin_window_size"],
        swin_depths    = p3["swin_depths"],
        swin_heads     = p3["swin_num_heads"],
        satmae_ckpt    = p2.get("satmae_checkpoint"),
        freeze_encoder = p2["freeze_encoder"],
    )
    result = model.load_state_dict(model_state, strict=True)
    print(f"  strict load: missing={len(result.missing_keys)} "
          f"unexpected={len(result.unexpected_keys)}")
    check(not result.missing_keys, "no missing keys")
    check(not result.unexpected_keys, "no unexpected keys")
else:
    check(False, "runs/phase3_fixed/best.pth exists")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} PROBLEM(S)'}")
sys.exit(1 if failures else 0)
