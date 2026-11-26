"""Model loading and class configuration.

The architecture factory lives in `predict.py` (single source of truth), so a
future change there — e.g. adding water/height heads — reaches the app for free.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Class registry. Water and height become available when a checkpoint with the
# corresponding heads exists; the app reads `available` at runtime.
CLASSES: dict[str, dict[str, Any]] = {
    "building": {
        "label": "Building",
        "color": (0, 255, 100),
        "alpha": 0.40,
    },
    "water": {
        "label": "Water",
        "color": (0, 140, 255),
        "alpha": 0.45,
    },
}

# Height is computed from shadow geometry (rs_core.height), not from a trained
# head, so it needs no checkpoint — only a GSD and the sun's position.
HEIGHT_AVAILABLE = True
HEIGHT_METHOD = "shadow_geometry"
HEIGHT_NOTE = ("estimate from shadow length x GSD x tan(sun elevation); "
               "needs a GSD and a timestamp/coordinates; not survey grade")


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_model(phase: int, cfg: dict, device):
    """Instantiate + restore a trained segmentation model."""
    from predict import load_model as _load

    return _load(phase, cfg, device)


# Checkpoint diagnostics are computed once per process: /health is polled by the
# UI and loading a 1.3 GB checkpoint on every request would be far too slow.
_DIAG_CACHE: dict[int, dict] = {}


def checkpoint_diagnostics(phase: int, cfg: dict) -> dict:
    """Cheaply inspect a checkpoint for the known `pos_embed` failure mode.

    Phases 2-4 were trained with the ViT position table deleted (all zeros),
    which makes the encoder position-blind. Those checkpoints still run, but
    their output is degraded, so the app must say so instead of quietly
    listing them next to healthy models.

    Uses mmap so only the few pages holding `pos_embed` are read.
    """
    if phase in _DIAG_CACHE:
        return _DIAG_CACHE[phase]

    import torch

    path = Path(cfg.get(f"phase{phase}", {}).get("checkpoint", ""))
    info: dict[str, Any] = {"degraded": False, "reason": None, "pos_embed_abs_sum": None}
    if not path.is_file():
        _DIAG_CACHE[phase] = info
        return info

    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = state.get("model", state) if isinstance(state, dict) else state
        pos = None
        for key in ("encoder.pos_embed", "pos_embed"):
            if isinstance(state, dict) and key in state:
                pos = state[key]
                break
        if pos is not None:
            total = float(pos.float().abs().sum())
            info["pos_embed_abs_sum"] = total
            if total == 0.0:
                info["degraded"] = True
                info["reason"] = ("ViT position table is all zeros (pos_embed bug); "
                                  "predictions are degraded — retrain with the fix")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the app
        info["reason"] = f"could not inspect checkpoint: {type(exc).__name__}: {exc}"

    _DIAG_CACHE[phase] = info
    return info


def available_phases(cfg: dict) -> dict[int, str]:
    """Phases whose checkpoint exists, with a short description."""
    base = {
        1: "Phase 1 — ResNet34 U-Net, buildings (fastest, best building IoU)",
        2: "Phase 2 — SatMAE++ + CNN decoder, buildings",
        3: "Phase 3 — SatMAE++ + Swin decoder, buildings",
        4: "Phase 4 — Phase 3 + pseudo-labels, buildings",
        5: "Phase 5 — ResNet34 U-Net two heads, buildings + water (LoveDA)",
    }
    out = {}
    for phase, label in base.items():
        ckpt = Path(cfg.get(f"phase{phase}", {}).get("checkpoint", ""))
        if not ckpt.is_file():
            continue
        diag = checkpoint_diagnostics(phase, cfg)
        out[phase] = f"{label} [DEGRADED — pos_embed zeros]" if diag["degraded"] else label
    return out


def class_status(cfg: dict) -> dict[str, dict[str, Any]]:
    """Which classes the available checkpoints can actually predict."""
    phases = available_phases(cfg)
    out = {name: dict(meta) for name, meta in CLASSES.items()}
    out["building"]["available"] = any(p in phases for p in (1, 2, 3, 4, 5))
    out["water"]["available"] = 5 in phases
    if 5 not in phases:
        out["water"]["reason"] = (
            "no two-head checkpoint yet — train it with train_multihead.py"
        )
    return out
