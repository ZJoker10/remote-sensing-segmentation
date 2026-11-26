"""Score any WHU building checkpoint on the held-out test split.

`train.py` only evaluates the test split at the very end of a run, so a run that
was stopped early has validation numbers but no test number. This closes that
gap without retraining, and it is the only way to compare `runs/phase3_fixed/`
(the retrained, pos_embed-fixed model) against the original
`checkpoints/phase3_best.pth` on equal terms.

Usage:
    venv\\Scripts\\python.exe tools\\eval_whu_test.py --phase 3 --checkpoint runs/phase3_fixed/best.pth
    venv\\Scripts\\python.exe tools\\eval_whu_test.py --phase 3 --checkpoint checkpoints/phase3_best.pth --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils.dataset import build_dataloaders  # noqa: E402
from utils.metrics import build_loss           # noqa: E402


def load_train_module():
    """train.py defines main() but does not run it, so importing is safe."""
    spec = importlib.util.spec_from_file_location("train_mod", ROOT / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=3)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    train_mod = load_train_module()

    print(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "epoch" in ckpt:
        print(f"  epoch {ckpt.get('epoch')} | stored val IoU {ckpt.get('val_iou')}")

    model = train_mod.build_model(args.phase, cfg)
    model.load_state_dict(ckpt["model"], strict=True)   # fails loudly on any drift
    model.to(device).eval()

    loaders = build_dataloaders(cfg)
    test_loader = loaders["test"]
    total_tiles = len(test_loader.dataset)
    print(f"Device: {device} | test tiles: {total_tiles} "
          f"| batches: {len(test_loader)}")

    results = train_mod.evaluate(model, test_loader, build_loss(cfg), device)
    print("\n=== test split (threshold 0.5) ===")
    for key in ("iou", "f1", "precision", "recall", "loss"):
        print(f"  {key:10s} {results[key]:.4f}")
    print(f"\n{tokens_note(total_tiles)}")
    return 0


def tokens_note(tiles: int) -> str:
    return (f"scored all {tiles} test tiles — same protocol as the numbers in "
            "README.md, so these are directly comparable")


if __name__ == "__main__":
    sys.exit(main())
