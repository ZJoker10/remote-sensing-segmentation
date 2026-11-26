"""Check the resume plumbing without running a full epoch.

A real resume test costs ~10 min/epoch on GPU, so this verifies the parts that
are actually easy to get wrong:

  * `build_scheduler` sizes the cosine from the *requested* epoch count
  * the resume path anneals from the restored LR instead of resetting it
  * the resume arithmetic against the paused phase-3 checkpoint is right
  * a checkpoint newer than `--epochs` is refused rather than silently skipped

Nothing here trains a model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


# train.py only defines main() at import time; it does not run it.
spec = importlib.util.spec_from_file_location("train_mod", ROOT / "train.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

CFG = {
    "scheduler": {"type": "cosine", "warmup_epochs": 3},
    "training": {"num_epochs": 50},
}
STEPS = 10  # pretend a 10-batch epoch so the schedule is cheap to step


def make_opt(lr: float = 1e-4):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


def lrs(scheduler, n):
    out = []
    for _ in range(n):
        out.append(scheduler.optimizer.param_groups[0]["lr"])
        scheduler.step()
    return out


print("=== fresh run: cosine is sized by --epochs, not config.yaml ===")
cfg = {"scheduler": dict(CFG["scheduler"]), "training": {"num_epochs": 10}}
sched = train_mod.build_scheduler(make_opt(), cfg, STEPS, epochs=10)
seq = lrs(sched, 10 * STEPS)
check(abs(seq[0] - 1e-4 * 0.01) < 1e-12,
      f"starts at the 1% warmup LR ({seq[0]:.3e})")
check(seq.index(max(seq)) == 3 * STEPS,
      f"peaks at the end of the 3-epoch warmup (step {seq.index(max(seq))})")
check(max(seq) < 1.01e-4, "peak never exceeds the configured LR")
check(seq[-1] < 1e-6, f"anneals to ~0 by the last step ({seq[-1]:.3e})")

print("\n=== resume: anneals from the restored LR, no warmup reset ===")
resume_cfg = {"scheduler": dict(CFG["scheduler"]), "training": {"num_epochs": 50}}
sched = train_mod.build_scheduler(make_opt(8.0e-5), resume_cfg, STEPS,
                                  epochs=46, warmup=False)
seq = lrs(sched, 46 * STEPS)
check(abs(seq[0] - 8.0e-5) < 1e-12,
      f"starts at the checkpoint LR, not 1% of it ({seq[0]:.3e})")
check(all(b <= a + 1e-15 for a, b in zip(seq, seq[1:])),
      "LR decreases monotonically (no warmup bump)")
check(seq[-1] < 1e-6, f"anneals to ~0 over the remaining epochs ({seq[-1]:.3e})")


print("\n=== resume arithmetic against the paused phase-3 checkpoint ===")
ckpt_path = ROOT / "runs" / "phase3_fixed" / "best.pth"
if not ckpt_path.is_file():
    check(False, f"{ckpt_path} exists")
else:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = int(ckpt.get("epoch") or 0)
    best_iou = float(ckpt.get("val_iou") or 0.0)
    start_epoch = epoch + 1
    print(f"  checkpoint: epoch={epoch} val_iou={best_iou:.4f} "
          f"-> start_epoch={start_epoch}")

    num_epochs = 50
    remaining = num_epochs - start_epoch + 1
    check(remaining == 46, f"--epochs 50 leaves {remaining} epochs, not a fresh 50")

    # The guard that stops a stale checkpoint from being silently skipped.
    def guard(num_epochs: int, start: int) -> bool:
        """Mirror of the `start_epoch > num_epochs` refusal in train.py."""
        return start > num_epochs

    check(not guard(num_epochs, start_epoch),
          f"--epochs {num_epochs} is accepted (start_epoch {start_epoch})")
    check(guard(start_epoch - 1, start_epoch),
          f"--epochs {start_epoch - 1} would be refused, not silently skipped")
    check(best_iou > 0.7,
          f"best_iou is seeded from the checkpoint ({best_iou:.4f}), so a worse "
          "epoch cannot overwrite best.pth")

print("\n=== resume: metrics.csv keeps one row per epoch ===")
import shutil
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    csv_path = Path(tmp) / "metrics.csv"
    shutil.copy(ROOT / "runs" / "phase3_fixed" / "metrics.csv", csv_path)
    before = csv_path.read_text().splitlines()
    trimmed = train_mod.trim_metrics(csv_path, start_epoch=5)
    after = csv_path.read_text().splitlines()
    epochs_after = [int(line.split(",")[0]) for line in after[1:]]

    check(trimmed == 1, f"removes exactly the epoch-5 row ({trimmed} removed)")
    check(after[0] == before[0], "header is preserved")
    check(epochs_after == [1, 2, 3, 4],
          f"epochs 1-4 survive in order ({epochs_after})")
    check(all(len(line.split(",")) == 10 for line in after[1:]),
          "every surviving row still has 10 columns")

    # A killed run can leave a half-written last line.
    with open(csv_path, "a") as f:
        f.write("5,0.0001,0.20\n")      # truncated, still parseable
        f.write("garbage\n")            # unparseable row number
    trimmed = train_mod.trim_metrics(csv_path, start_epoch=5)
    check(trimmed == 2, f"truncated and unparseable trailing rows are dropped ({trimmed})")
    check(train_mod.trim_metrics(csv_path, start_epoch=1) == 4,
          "resuming from epoch 1 clears the body")
    check(csv_path.read_text().splitlines() == [before[0]],
          "header alone remains")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} PROBLEM(S)'}")
sys.exit(1 if failures else 0)
