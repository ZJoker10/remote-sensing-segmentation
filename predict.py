# predict.py
# Run a trained model on new satellite images and save the predicted masks.
#
# Usage:
#   python predict.py --phase 1 --input data/test_images/ --output results/
#   python predict.py --phase 3 --input my_image.tif --output results/
#   python predict.py --phase 3 --input my_image.tif --overlay  # save color overlay

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(phase: int, cfg: dict, device: torch.device):
    """Load the correct model architecture and restore weights from checkpoint."""

    if phase == 1:
        from phase1_baseline.resnet_unet import ResNetUNet
        model = ResNetUNet(num_classes=1, pretrained=False)
        ckpt_path = cfg["phase1"]["checkpoint"]

    elif phase == 2:
        # Reuse the same simple decoder from train.py phase 2
        sys.path.insert(0, str(Path(__file__).parent))
        from phase2_satmae.satmae_encoder import SatMAEPlusEncoder, FeaturePyramidAdapter
        from phase1_baseline.resnet_unet import UpBlock

        class SatMAESimpleDecoder(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                p2 = cfg["phase2"]
                self.encoder = SatMAEPlusEncoder(
                    img_size=cfg["data"]["image_size"], patch_size=p2["patch_size"],
                    embed_dim=p2["embed_dim"], depth=24, num_heads=16,
                    extract_layers=[5, 11, 17, 23], freeze=True,
                )
                self.adapter = FeaturePyramidAdapter(p2["embed_dim"], [128, 256, 512, 1024])
                self.up3 = UpBlock(1024, 512, 512)
                self.up2 = UpBlock(512,  256, 256)
                self.up1 = UpBlock(256,  128, 128)
                self.head = torch.nn.Conv2d(128, 1, 1)

            def forward(self, x):
                H, W = x.shape[2:]
                feats = self.encoder(x)
                feats = self.adapter(feats)
                d = self.up3(feats[3], feats[2])
                d = self.up2(d, feats[1])
                d = self.up1(d, feats[0])
                d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
                return self.head(d)

        model = SatMAESimpleDecoder(cfg)
        ckpt_path = cfg["phase2"]["checkpoint"]

    elif phase in (3, 4):
        from phase3_swin_unet.swin_unet_decoder import SatMAESwinUNet
        p2, p3 = cfg["phase2"], cfg["phase3"]
        model = SatMAESwinUNet(
            img_size=cfg["data"]["image_size"], patch_size=p2["patch_size"],
            num_classes=1, window_size=p3["swin_window_size"],
            swin_depths=p3["swin_depths"], swin_heads=p3["swin_num_heads"],
            satmae_ckpt=None, freeze_encoder=p2["freeze_encoder"],
        )
        ckpt_path = cfg[f"phase{phase}"]["checkpoint"]

    elif phase == 5:
        # Two-head building + water model on LoveDA (train_multihead.py).
        # Channel order is fixed by rs_core.inference.CLASS_ORDER.
        from phase1_baseline.resnet_unet import ResNetUNet
        model = ResNetUNet(num_classes=2, pretrained=False)
        ckpt_path = cfg["phase5"]["checkpoint"]

    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Load saved weights
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Make sure you have run: python train.py --phase {phase}"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"Loaded Phase {phase} model from {ckpt_path}")
    if "val_iou" in ckpt:
        print(f"  Trained for {ckpt['epoch']} epochs | Best Val IoU: {ckpt['val_iou']:.4f}")
    else:
        # Multi-head checkpoints report per-class metrics instead of one IoU.
        metrics = ckpt.get("metrics", {})
        summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                            if isinstance(v, float))
        print(f"  Trained for {ckpt.get('epoch', '?')} epochs"
              + (f" | {summary}" if summary else ""))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Single image inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(
    model,
    image_path: str,
    image_size: int,
    device: torch.device,
    threshold: float = 0.5,
):
    """
    Run model on one image, using the same tiled inference as the app.

    Returns:
        original_image: numpy (H, W, 3)
        results:        dict  class_name -> {"mask": (H,W) uint8 {0,1},
                                             "prob": (H,W) float [0,1]}

    Single-head phases (1-4) return one entry under "building"; the phase-5
    two-head model returns "building" and "water".

    `image_size` is accepted for backwards compatibility and ignored. This used
    to resize the whole scene to 512 px, which blurred thin features and made
    these predictions disagree with the app and with `tools/eval_phase5.py`.
    `rs_core.inference` tiles at 512 px with Hann blending instead, so the
    numbers here now match the reported ones.
    """
    from rs_core.inference import predict_probs

    original = np.array(Image.open(image_path).convert("RGB"))
    probs = predict_probs(model, original, device)      # (C, H, W), full res

    names = class_names_for(probs.shape[0])
    results: dict[str, dict[str, np.ndarray]] = {}
    for ch, name in enumerate(names):
        prob_map = probs[ch].astype(np.float32)
        results[name] = {
            "mask": (prob_map > threshold).astype(np.uint8),
            "prob": prob_map,
        }

    return original, results


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_mask(binary_mask: np.ndarray, save_path: str):
    """Save binary mask as a black/white PNG (255 = building, 0 = background)."""
    Image.fromarray(binary_mask * 255).save(save_path)


def save_overlay(original: np.ndarray, binary_mask: np.ndarray, save_path: str,
                 color=(0, 255, 100), alpha=0.4):
    """
    Save the original image with predicted buildings highlighted in green.
    Useful for visually checking prediction quality.
    """
    overlay = original.copy().astype(np.float32)

    # Color the predicted building pixels
    for c, val in enumerate(color):
        overlay[:, :, c] = np.where(
            binary_mask == 1,
            overlay[:, :, c] * (1 - alpha) + val * alpha,
            overlay[:, :, c]
        )

    Image.fromarray(overlay.astype(np.uint8)).save(save_path)


def save_probability_map(prob_map: np.ndarray, save_path: str):
    """Save the raw probability map as a grayscale heatmap."""
    heatmap = (prob_map * 255).astype(np.uint8)
    Image.fromarray(heatmap).save(save_path)


# Channel order is fixed by rs_core.inference.CLASS_ORDER; keep them in step.
HEAD_NAMES = ("building", "water")

# Overlay colours, matching the app (rs_core.visualize.CLASSES).
HEAD_COLOURS = {"building": (0, 255, 100), "water": (0, 140, 255)}


def class_names_for(n_channels: int) -> tuple[str, ...]:
    """Map a model's output-channel count to class names."""
    if n_channels < 1:
        raise ValueError(f"model returned {n_channels} channels")
    return HEAD_NAMES[:n_channels]


def save_results(original: np.ndarray, results: dict, output_dir: Path,
                 stem: str, save_overlay_flag: bool = False,
                 overlay_dir: Path | None = None) -> None:
    """Write mask / probability / overlay files for every predicted class.

    Single-class phases keep the original `{stem}_mask.png` naming so existing
    scripts and notes still work. Two-head models get `{stem}_building_mask.png`
    and `{stem}_water_mask.png`.
    """
    overlay_dir = overlay_dir or output_dir
    multi = len(results) > 1
    for name, res in results.items():
        suffix = f"_{name}" if multi else ""
        save_mask(res["mask"], str(output_dir / f"{stem}{suffix}_mask.png"))
        save_probability_map(res["prob"],
                             str(output_dir / f"{stem}{suffix}_prob.png"))
        if save_overlay_flag:
            save_overlay(original, res["mask"],
                         str(overlay_dir / f"{stem}{suffix}_overlay.png"),
                         color=HEAD_COLOURS.get(name, (0, 255, 100)))


def coverage_report(results: dict) -> str:
    """Human-readable coverage line(s), one per class."""
    return "\n".join(
        f"  {name}: {res['mask'].mean() * 100:.1f}% of image"
        for name, res in results.items()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_folder(model, input_dir, output_dir, image_size, device,
                   threshold=0.5, save_overlay_flag=False):
    """Run prediction on every image in a folder."""

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    masks_dir  = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if save_overlay_flag:
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)

    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_paths = [p for p in sorted(input_dir.iterdir())
                   if p.suffix.lower() in extensions]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Running inference on {len(image_paths)} images...")

    for img_path in image_paths:
        original, results = predict_image(
            model, str(img_path), image_size, device, threshold
        )

        save_results(original, results, masks_dir, img_path.stem,
                     save_overlay_flag,
                     overlay_dir=overlay_dir if save_overlay_flag else None)

        print(f"  ✓ {img_path.name}")
        print(coverage_report(results))

    print(f"\nDone. Results saved to: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite image segmentation inference")
    parser.add_argument("--phase",     type=int,   default=1,
                        help="Which trained model to use: 1 | 2 | 3 | 4 | 5")
    parser.add_argument("--input",     required=True,
                        help="Path to a single image OR a folder of images")
    parser.add_argument("--output",    default="./results",
                        help="Folder to save predictions")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for binary mask (default 0.5)")
    parser.add_argument("--overlay",   action="store_true",
                        help="Also save color overlay images")
    parser.add_argument("--device",    choices=["auto", "cuda", "cpu"],
                        default="auto",
                        help="auto (default) uses CUDA when available")
    parser.add_argument("--config",    default="configs/config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = load_model(args.phase, cfg, device)

    # Run prediction
    input_path = Path(args.input)

    if input_path.is_dir():
        predict_folder(
            model, input_path, args.output,
            cfg["data"]["image_size"], device,
            args.threshold, args.overlay,
        )
    elif input_path.is_file():
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        original, results = predict_image(
            model, str(input_path), cfg["data"]["image_size"],
            device, args.threshold,
        )

        stem = input_path.stem
        save_results(original, results, output_dir, stem, args.overlay)

        print(f"\nPrediction saved to: {output_dir}")
        print("Coverage:")
        print(coverage_report(results))
    else:
        print(f"Error: {args.input} is not a valid file or folder")