# HANDOFF — current state and how to continue

**State as of 18 Nov 2026.** Delivered: water works (0.7064 IoU, 20 epochs),
buildings use phase 1 (0.884 test IoU), height works where shadows exist.
The phase 3 retrain was stopped at its completed 5 epochs (val 0.7642, test
0.7291) and remains resumable.

Nothing is running. Nothing needs to be stopped.

Everything below was verified with the scripts named next to it — this is not a
plan, it is the state on disk.

---

## 1. The one command to continue

```bat
cd "Z:\AI & Robtics Masters\N.A\Neural Networks\Project\remote_sensing"
venv\Scripts\python.exe train.py --phase 3 --resume runs\phase3_fixed\best.pth --epochs 50 --tag phase3_fixed > runs\phase3_fixed.log 2>&1
```

`--epochs 50` is the **total** epoch count, not the number still to do. The
checkpoint is at epoch 4, so this runs epochs 5–50 (46 epochs, ~7 h 20 min at
the measured 9.6 min/epoch).

If you want fewer, `--epochs` must still be **greater than 4** or the run
refuses to start (it will not silently skip a checkpoint it cannot continue).

Phase 3 was deliberately left at its **completed 5 epochs** — that is why it is
paused rather than unfinished. Its measured test IoU is **0.7291** (all 2416 test
tiles, `tools/eval_whu_test.py`), i.e. below the broken 50-epoch model's 0.7967.
Finishing the 46 epochs is what would actually test the `pos_embed` fix.

## 2. What is saved

| Artifact | State |
|---|---|
| `runs/phase3_fixed/best.pth` | 1.5 GB, **epoch 4, val IoU 0.7642**, test IoU 0.7291 |
| `runs/phase3_fixed/metrics.csv` | epochs 1–5 |
| `runs/phase3_fixed/config.yaml` | exact config of the run |
| `runs/phase3_fixed.log` | full stdout including epoch 5 |

Epoch history: 0.6769 → 0.7211 → 0.7419 → **0.7642** → 0.7580.

Epoch 5 was worse than epoch 4, so `best.pth` is epoch 4 and epoch 5 exists only
in the CSV/log. On resume the CSV's epoch-5 row is **replaced**, because that
epoch gets re-run — the old value stays readable in `phase3_fixed.log`.

Verified by `tools/verify_saved_state.py` (all checks pass):
- checkpoint loads into a freshly built phase-3 model with **strict=True**,
  0 missing / 0 unexpected keys — the expensive thing to discover 7 hours in
- `encoder.pos_embed` abs-sum = 586968.19, i.e. the bicubic interpolation fix
  is baked into these weights (the original checkpoints had it all zeros)

## 3. How resume behaves (read this before judging the numbers)

* Weights **and** optimizer state (AdamW moments, step counts) are restored.
* The **scheduler position is not stored** in the checkpoint. Resume therefore
  does a **warm restart**: a fresh cosine anneals from the checkpoint's current
  LR (8.02e-5) down to ~0 across the 46 remaining epochs, with **no** 1%
  warmup — resetting the LR to 1% and climbing back would throw away the
  progress the run had already made.
* Consequence: this is not bit-identical to an uninterrupted 50-epoch run. The
  LR curve differs after epoch 5. Report it as "resumed", not as "a single
  50-epoch run".
* `best_iou` starts at 0.7642, so a worse epoch cannot overwrite `best.pth`.
* Checkpoint writes are now write-to-`.tmp`-then-`os.replace`, so killing the
  process mid-save can no longer truncate `best.pth`.

Checked by `tools/test_resume.py` (all checks pass): the cosine is sized by
`--epochs` rather than `config.yaml`, the resume path starts at the restored LR
and decreases monotonically to ~0, no warmup bump.

## 4. Phase 5 (water) — trained, 20 epochs

`checkpoints/phase5_multihead_best.pth` is the **epoch-17** model (best validation
mean IoU 0.6844). `checkpoints/phase5_multihead_5ep.pth` keeps the earlier
5-epoch model for comparison.

**Measured** by `tools/eval_phase5.py` — full 1024² tiles, 62 of 1669 Val tiles
(every 27th, unfiltered), 10.05 M valid px, threshold 0.5, same tiles both rows:

| Run | Building IoU | Water IoU | Mean |
|---|---|---|---|
| 5 epochs | 0.5944 | 0.7023 | 0.6484 |
| **20 epochs** | **0.6597** | **0.7064** | **0.6830** |

Building precision/recall at 20 epochs: 0.7418 / 0.8562. Water: 0.8635 / 0.7952.

**Honest reading:** 15 more epochs bought +0.065 building IoU and +0.004 water.
Water had converged by epoch 5 — it is 7.4 % of pixels — while buildings, at 2.7 %,
kept improving. Training further is unlikely to move water much; if more is
wanted, the lever is better data or a stronger encoder, not more epochs.

Epoch curve (crop validation, 200 tiles): 0.453 → 0.617 (5) → 0.684 (17) → 0.678 (20).

`train_multihead.py` has **no `--resume`**, so any further training is a fresh run:

```bat
venv\Scripts\python.exe train_multihead.py --epochs 20 --tag phase5_loveda20
venv\Scripts\python.exe tools\eval_phase5.py --limit 60
```

Note: a fresh run resets `best_mean` to -1, so its epoch 1 checkpoint will briefly
overwrite `phase5_multihead_best.pth` with a worse model before improving. Back the
current one up before re-running, as was done here.

Remaining failure mode (`outputs/phase5_eval/contact_sheet.png`): dark deep-water
tiles are over-segmented and shorelines are blocky.

Do **not** compare these numbers to the WHU building IoUs — different sensor,
GSD and label source. Phase-5 building 0.6597 is not a replacement for phase 1's
0.884.

Reminder of the mapping (this was a real bug — the original code trained the
building head on background and the water head on roads):

| raw LoveDA id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| meaning | ignore | background | **building** | road | **water** | barren | forest | agricultural |

## 5. Run this first, next session

```bat
venv\Scripts\python.exe tools\verify_saved_state.py   :: mapping + checkpoint integrity + strict load
venv\Scripts\python.exe tools\check_loveda.py         :: dataset pairing / ids / split counts
venv\Scripts\python.exe tools\test_resume.py          :: schedule + resume arithmetic
venv\Scripts\python.exe tools\test_height.py          :: solar / shadow-height maths
venv\Scripts\python.exe tools\smoke_app.py            :: app end-to-end on CPU
venv\Scripts\python.exe tools\eval_phase5.py --limit 60  :: phase-5 IoU on full tiles
venv\Scripts\python.exe tools\eval_whu_test.py --phase 3 --checkpoint runs\phase3_fixed\best.pth
```

All passed as of wrap-up. If any fails, fix it before spending GPU hours —
each one guards something that would otherwise waste the whole run.

## 6. Where things stand overall

| Thing | Status |
|---|---|
| Review of the original 4 phases | done, with measured numbers |
| `pos_embed` all-zeros bug | fixed in code; phase 3 retrain shows val 0.7642 / test 0.7291 at 5 epochs |
| GSD 16x area bug (0.075 → 0.30 m/px) | fixed |
| Building height + volume (shadow geometry) | works; refuses without shadows |
| Water (phase 5, LoveDA) | **done — 0.7064 water IoU, 20 epochs** |
| Upload app (`start_app.bat`) | works, CPU smoke test green, water selectable |
| Phase 3 retrain | **stopped at its completed 5 epochs**, resumable |

## 7. Honest caveats to keep when reporting

* `checkpoints/phase3_best.pth` (the old one) still has `pos_embed` all zeros.
  The app flags it `[DEGRADED]` instead of hiding it.
* The retrained phase 3 scores **0.7291 test IoU at 5 epochs**, *below* the broken
  model's 0.7967 across ~50 epochs. That is a training-budget difference, not
  evidence the fix hurt — but do **not** claim the fix improved phase 3 end-to-end
  until the 46 remaining epochs are trained and re-scored.
* Height is an **estimate from shadow length**, not LiDAR. Most WHU tiles have
  no usable shadow and are refused with an explicit reason; that is the correct
  behaviour, not a failure.
* No git repo here. Nothing is committed; the Z: drive holds the only copy.
