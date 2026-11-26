# train.py
# Unified training entry point for all 4 phases.
#
# Usage:
#   python train.py --phase 1               # ResNet U-Net baseline
#   python train.py --phase 2               # SatMAE++ encoder + simple decoder
#   python train.py --phase 3               # Full SatMAE++ + Swin-Unet
#   python train.py --phase 4               # Phase 3 + SAM pseudo-labels
#   python train.py --phase 4 --gen_pseudo  # Generate pseudo-labels first

import os
import sys
import time
import yaml
import argparse
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Redirected output defaults to the Windows legacy code page (cp1252), which
# cannot encode characters like the check marks and box-drawing banners used
# below. Without this, `python train.py > run.log` crashes mid-training.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # non-standard stream, leave it alone
        pass

# ── Project imports ───────────────────────────────────────────────────────────
from utils.dataset import build_dataloaders
from utils.metrics import SegmentationMetrics, build_loss


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(phase: int, cfg: dict) -> nn.Module:
    """Instantiate the correct model for the given phase."""

    if phase == 1:
        from phase1_baseline.resnet_unet import ResNetUNet
        return ResNetUNet(
            num_classes = 1,
            pretrained  = cfg["phase1"]["encoder_pretrained"],
        )

    elif phase == 2:
        # SatMAE++ encoder + lightweight CNN decoder (thin U-Net head)
        from phase2_satmae.satmae_encoder import SatMAEPlusEncoder, FeaturePyramidAdapter
        from phase1_baseline.resnet_unet import UpBlock, DoubleConv
        import torch.nn.functional as F

        class SatMAESimpleDecoder(nn.Module):
            """Phase 2 model: SatMAE++ encoder + basic CNN decoder (no Swin).
            Shows that even a simple decoder benefits from strong pre-training."""

            def __init__(self, cfg):
                super().__init__()
                p2 = cfg["phase2"]
                self.encoder = SatMAEPlusEncoder(
                    img_size       = cfg["data"]["image_size"],
                    patch_size     = p2["patch_size"],
                    embed_dim      = p2["embed_dim"],
                    depth          = 24, num_heads=16,
                    extract_layers = [5, 11, 17, 23],
                    freeze         = p2["freeze_encoder"],
                )
                if p2.get("satmae_checkpoint"):
                    self.encoder.load_satmae_checkpoint(p2["satmae_checkpoint"])

                self.adapter = FeaturePyramidAdapter(p2["embed_dim"], [128, 256, 512, 1024])

                self.up3 = UpBlock(1024, 512, 512)
                self.up2 = UpBlock(512,  256, 256)
                self.up1 = UpBlock(256,  128, 128)
                self.head = nn.Conv2d(128, 1, 1)

            def forward(self, x):
                H, W = x.shape[2:]
                feats = self.encoder(x)
                feats = self.adapter(feats)
                d = self.up3(feats[3], feats[2])
                d = self.up2(d,        feats[1])
                d = self.up1(d,        feats[0])
                d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
                return self.head(d)

        return SatMAESimpleDecoder(cfg)

    elif phase in (3, 4):
        from phase3_swin_unet.swin_unet_decoder import SatMAESwinUNet
        p2 = cfg["phase2"]
        p3 = cfg["phase3"]
        return SatMAESwinUNet(
            img_size        = cfg["data"]["image_size"],
            patch_size      = p2["patch_size"],
            num_classes     = 1,
            window_size     = p3["swin_window_size"],
            swin_depths     = p3["swin_depths"],
            swin_heads      = p3["swin_num_heads"],
            satmae_ckpt     = p2.get("satmae_checkpoint"),
            freeze_encoder  = p2["freeze_encoder"],
        )

    else:
        raise ValueError(f"Unknown phase: {phase}")


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer factory
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict, phase: int):
    opt_cfg = cfg["optimizer"]
    lr      = opt_cfg["lr"]
    wd      = opt_cfg["weight_decay"]

    # Phase 3/4: use differential LR for encoder vs decoder.
    # Only SatMAESwinUNet exposes an encoder/decoder split; phase 2 must not
    # silently pretend it did (the old try/except swallowed the mismatch).
    if phase >= 3 and hasattr(model, "encoder"):
        from phase3_swin_unet.swin_unet_decoder import get_param_groups
        enc_scale = cfg["phase2"]["encoder_lr_scale"]
        param_groups = get_param_groups(model, lr, enc_scale)
        frozen = sum(p.numel() for p in model.encoder.parameters()
                     if not p.requires_grad) / 1e6
        print(f"  Differential LR: encoder x{enc_scale} "
              f"({frozen:.1f}M frozen encoder params)")
    else:
        param_groups = model.parameters()

    if opt_cfg["type"] == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif opt_cfg["type"] == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")


def trim_metrics(path, start_epoch: int) -> int:
    """Drop rows for epochs >= start_epoch from a metrics.csv, keeping the header.

    Used on resume: those epochs are about to be re-run, and a log holding two
    different results for the same epoch is worse than no log.
    Returns the number of rows removed.
    """
    path = Path(path)
    if not path.is_file():
        return 0
    lines  = path.read_text().splitlines()
    header = lines[0] if lines else ""
    kept, removed = [], 0
    for line in lines[1:]:
        try:
            epoch_in_row = int(line.split(",")[0])
        except (ValueError, IndexError):
            removed += 1        # half-written row from a killed run
            continue
        if epoch_in_row < start_epoch:
            kept.append(line)
        else:
            removed += 1
    if removed:
        path.write_text("\n".join([header, *kept]) + "\n")
    return removed


def build_scheduler(optimizer, cfg: dict, num_steps: int,
                    epochs: int | None = None, warmup: bool = True):
    """Cosine schedule sized for `epochs` (defaults to the config value).

    `warmup=False` is the resume path: the LR anneals from whatever value the
    restored optimizer already holds instead of being reset to 1% and ramped
    back up, which would throw away the schedule position the run had reached.
    """
    sched_cfg    = cfg["scheduler"]
    epochs       = epochs or cfg["training"]["num_epochs"]
    total_steps  = epochs * num_steps

    if sched_cfg["type"] != "cosine":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    if not warmup:
        return CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    warmup_steps = int(sched_cfg["warmup_epochs"] * num_steps)
    # Always leave at least one step for the cosine half of the schedule.
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    metrics    = SegmentationMetrics()

    for batch in tqdm(loader, desc="  train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device,  non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        metrics.update(logits.detach(), masks)

    avg_loss = total_loss / len(loader)
    results  = metrics.compute()
    results["loss"] = round(avg_loss, 4)
    return results


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics    = SegmentationMetrics()

    for batch in tqdm(loader, desc="  val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device,  non_blocking=True)

        with autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item()
        metrics.update(logits, masks)

    avg_loss = total_loss / len(loader)
    results  = metrics.compute()
    results["loss"] = round(avg_loss, 4)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      type=int, default=1,
                        help="Pipeline phase: 1 | 2 | 3 | 4")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--gen_pseudo", action="store_true",
                        help="(Phase 4) Run SAM pseudo-label generation first")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override training.num_epochs (useful for short runs)")
    parser.add_argument("--tag",        default=None,
                        help="Name for runs/<tag>/. Defaults to phase<N>_<timestamp>")
    parser.add_argument("--set",        action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value, e.g. --set phase4.pseudo_label_ratio=0.0")
    parser.add_argument("--resume",     default=None, metavar="CKPT",
                        help="Continue from a saved best.pth (e.g. runs/phase3_fixed/best.pth). "
                             "Weights and optimizer state are restored and the cosine "
                             "schedule gets a warm restart over the remaining epochs.")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Ad-hoc config overrides (keeps config.yaml authoritative by default) ──
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)

    set_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Phase 4: generate pseudo-labels if requested ─────────────────────────
    pseudo_dir = None
    if args.phase == 4:
        pseudo_dir = cfg["phase4"]["pseudo_label_dir"]
        if args.gen_pseudo:
            from phase4_sam_pseudolabel.sam_pipeline import (
                SAMPipelineConfig, SAMPseudoLabelGenerator
            )
            p4 = cfg["phase4"]
            sam_cfg = SAMPipelineConfig(
                model_type          = p4["sam_model_type"],
                checkpoint          = p4["sam_checkpoint"],
                device              = cfg["training"]["device"],
                pred_iou_threshold  = p4["iou_threshold"],
                unlabeled_dir       = cfg["data"]["root"] + "/" + cfg["data"].get("unlabeled_dir", "unlabeled"),
                output_dir          = pseudo_dir,
            )
            SAMPseudoLabelGenerator(sam_cfg).process_dataset()

    # ── Dataloaders ───────────────────────────────────────────────────────────
    loaders = build_dataloaders(cfg, pseudo_label_dir=pseudo_dir)
    print(f"Train batches: {len(loaders['train'])} | "
          f"Val batches: {len(loaders['val'])}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.phase, cfg).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Phase {args.phase} | Total params: {total:.1f}M | "
          f"Trainable: {trainable:.1f}M")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg, args.phase)
    scaler    = GradScaler(device=device.type, enabled=cfg["training"]["mixed_precision"])

    # `--epochs` must drive the schedule, not just the loop: sizing the cosine
    # from config.yaml while running fewer epochs leaves the LR high at the end.
    num_epochs = args.epochs or cfg["training"]["num_epochs"]
    cfg["training"]["num_epochs"] = num_epochs

    # ── Checkpointing + run directory ─────────────────────────────────────────
    run_name  = args.tag or f"phase{args.phase}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir   = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pth"
    best_iou  = 0.0
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"--resume: checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        best_iou    = float(ckpt.get("val_iou") or 0.0)
        start_epoch = int(ckpt.get("epoch") or 0) + 1
        print(f"Resumed {resume_path} at epoch {ckpt.get('epoch')} "
              f"(val IoU {best_iou:.4f}); continuing from epoch {start_epoch}")
        if start_epoch > num_epochs:
            raise SystemExit(
                f"--resume: checkpoint is at epoch {start_epoch - 1} but "
                f"--epochs is {num_epochs}; raise --epochs to continue")

    # The cosine schedule is sized for the epochs that are actually left. On
    # resume this is a warm restart from the checkpoint's current LR, which is
    # deliberate: the original schedule position was not saved, so pretending to
    # continue it exactly would be worse than restarting it honestly.
    remaining = num_epochs - start_epoch + 1
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]),
                                epochs=remaining, warmup=args.resume is None)

    # Snapshot the exact config used, so the run is reproducible on its own.
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text(
            "epoch,lr,train_loss,train_iou,train_f1,"
            "val_loss,val_iou,val_f1,val_precision,val_recall\n"
        )
    elif args.resume:
        trimmed = trim_metrics(metrics_path, start_epoch)
        if trimmed:
            print(f"Trimmed {trimmed} metrics row(s) from epoch "
                  f"{start_epoch} onward (about to be re-run)")
    print(f"Run directory: {run_dir}")

    # ── Training loop ─────────────────────────────────────────────────────────
    if start_epoch > 1:
        print(f"\nResuming Phase {args.phase} training at epoch {start_epoch} "
              f"of {num_epochs} ({remaining} epoch(s) left, best val IoU so far "
              f"{best_iou:.4f})...\n")
    else:
        print(f"\nStarting Phase {args.phase} training for {num_epochs} epochs...\n")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_results = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, criterion, scaler, device
        )
        val_results = evaluate(model, loaders["val"], criterion, device)

        print(f"  [Train] Loss={train_results['loss']:.4f} | "
              f"IoU={train_results['iou']:.4f} | F1={train_results['f1']:.4f}")
        print(f"  [Val]   Loss={val_results['loss']:.4f} | "
              f"IoU={val_results['iou']:.4f} | F1={val_results['f1']:.4f}")

        # Append a machine-readable row so results never live only in a terminal dump.
        # Log the *largest* group lr: group 0 is the frozen encoder, whose lr is
        # meaningless and misleading when reading the CSV later.
        lr_logged = max(g["lr"] for g in optimizer.param_groups)
        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{lr_logged:.6g},"
                f"{train_results['loss']},{train_results['iou']},{train_results['f1']},"
                f"{val_results['loss']},{val_results['iou']},{val_results['f1']},"
                f"{val_results['precision']},{val_results['recall']}\n"
            )

        # Save best checkpoint
        if val_results["iou"] > best_iou:
            best_iou = val_results["iou"]
            # Write-then-rename: a kill mid-save would otherwise leave a
            # truncated best.pth and lose every epoch before it.
            tmp_path = ckpt_path.with_suffix(".pth.tmp")
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_iou":     best_iou,
                "cfg":         cfg,
            }, tmp_path)
            os.replace(tmp_path, ckpt_path)
            print(f"  ✓ Best model saved (IoU={best_iou:.4f})")

    print(f"\nPhase {args.phase} complete. Best Val IoU: {best_iou:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_results = evaluate(model, loaders["test"], criterion, device)
    print(f"\n[Test] IoU={test_results['iou']:.4f} | "
          f"F1={test_results['f1']:.4f} | "
          f"Precision={test_results['precision']:.4f} | "
          f"Recall={test_results['recall']:.4f}")


if __name__ == "__main__":
    main()