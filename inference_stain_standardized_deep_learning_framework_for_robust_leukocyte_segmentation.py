#!/usr/bin/env python3
"""
Inference script for:
Stain-Standardized Deep Learning Framework for Robust Leukocyte Segmentation

Pipeline:
1) Load RGB images from input_dir
2) Resize -> COLORIZATION_IMAGE_SIZE, convert to LAB
3) Build 3xL input for the colorization model (VGG16 expects 3 channels)
4) Predict (a*, b*) and reconstruct standardized RGB image
5) Resize -> SEGMENTATION_IMAGE_SIZE and run segmentation model
6) Save outputs (mask + overlay)

Assumptions (adapt if your saved models differ):
- colorization_model(input) expects shape (B, Hc, Wc, 3) float32, where channels are [L, L, L]
- colorization_model output is shape (B, Hc, Wc, 2) float32 representing (a*, b*)
- segmentation_model expects a list of inputs in the order used below (multi-input model)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from skimage import color

# Your project utilities (must be available in PYTHONPATH)
from leukocyte_segmentation_module import compute_channels_and_masks_batch


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def read_rgb_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def resize_rgb(img_rgb: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    # size_hw = (H, W)
    h, w = size_hw
    return cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def rgb_uint8_to_float01(img_rgb_u8: np.ndarray) -> np.ndarray:
    return img_rgb_u8.astype(np.float32) / 255.0


def float01_to_uint8(img_rgb_f: np.ndarray) -> np.ndarray:
    x = np.clip(img_rgb_f, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def build_L3_input(rgb_float01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    rgb_float01: (B, H, W, 3) in [0,1]
    Returns:
      L3: (B, H, W, 3) where each channel is L
      L:  (B, H, W, 1) original luminance
    """
    lab = color.rgb2lab(rgb_float01)  # L in [0,100], a/b roughly [-128,128]
    L = lab[..., 0:1].astype(np.float32)  # (B,H,W,1)
    L3 = np.repeat(L, 3, axis=-1)         # (B,H,W,3)
    return L3, L


def reconstruct_rgb_from_L_and_ab(L: np.ndarray, ab_pred: np.ndarray) -> np.ndarray:
    """
    L:      (B, H, W, 1)
    ab_pred:(B, H, W, 2)
    Returns standardized RGB float in [0,1], shape (B,H,W,3)
    """
    L = L[..., 0].astype(np.float32)               # (B,H,W)
    a = ab_pred[..., 0].astype(np.float32)         # (B,H,W)
    b = ab_pred[..., 1].astype(np.float32)         # (B,H,W)
    lab = np.stack([L, a, b], axis=-1)             # (B,H,W,3)
    rgb = color.lab2rgb(lab)                       # (B,H,W,3) float64 in [0,1]
    return rgb.astype(np.float32)


def onehot_to_rgb_mask(onehot: np.ndarray) -> np.ndarray:
    """
    onehot: (B,H,W,C)
    Returns RGB visualization (B,H,W,3) uint8
    Color mapping can be adjusted to your paper conventions.
    """
    # Example mapping (background=0, class1=1, class2=2):
    # 0 -> black, 1 -> yellow, 2 -> green
    color_dict = {
        0: np.array([0, 0, 0], dtype=np.uint8),
        1: np.array([200, 200, 0], dtype=np.uint8),
        2: np.array([0, 200, 0], dtype=np.uint8),
    }

    argmax = np.argmax(onehot, axis=-1)  # (B,H,W)
    B, H, W = argmax.shape
    out = np.zeros((B, H, W, 3), dtype=np.uint8)
    for k, col in color_dict.items():
        out[argmax == k] = col
    return out


def overlay_mask(rgb_u8: np.ndarray, mask_rgb_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    rgb_u8: (H,W,3), mask_rgb_u8: (H,W,3)
    Returns overlay uint8
    """
    rgb_f = rgb_u8.astype(np.float32)
    m_f = mask_rgb_u8.astype(np.float32)
    out = (1.0 - alpha) * rgb_f + alpha * m_f
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------
# Segmentation prediction
# ----------------------------
def predict_segmentation(
    segmentation_model: tf.keras.Model,
    input_rgb_batch_float01: np.ndarray,
    n_classes: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    input_rgb_batch_float01: (B,H,W,3) in [0,1]
    Returns:
      pred_onehot: (B,H,W,n_classes) one-hot
      pred_rgb_vis: (B,H,W,3) uint8 visualization
    """
    # compute_channels_and_masks_batch should accept float RGB in [0,1] (or adapt if yours expects 0-255)
    nuc_ch, leu_ch, masks = compute_channels_and_masks_batch(input_rgb_batch_float01)

    # Assemble inputs in the SAME ORDER used at training time
    X_inputs_for_model = [
        input_rgb_batch_float01.astype(np.float32),  # 0) RGB
        nuc_ch,                                      # 1) nuc_channels
        masks["sat_otsu"],                            # 2) sat_otsu
        masks["green_min"],                           # 3) green_min
        masks["mag_iso"],                             # 4) mag_iso
        leu_ch,                                       # 5) leuko_channels
        masks["yellow_li"],                           # 6) yellow_li
        masks["v_otsu"],                              # 7) v_otsu
        masks["v_yen"],                               # 8) v_yen
    ]

    logits_or_probs = segmentation_model.predict(X_inputs_for_model, batch_size=batch_size, verbose=0)
    pred_labels = np.argmax(logits_or_probs, axis=-1)  # (B,H,W)
    pred_onehot = tf.keras.utils.to_categorical(pred_labels, num_classes=n_classes).astype(np.uint8)
    pred_rgb_vis = onehot_to_rgb_mask(pred_onehot)
    return pred_onehot, pred_rgb_vis


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference for stain-standardized leukocyte segmentation (colorization + segmentation)."
    )
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    p.add_argument("--colorization_model", type=str, required=True, help="Path to colorization .h5 model (full model).")
    p.add_argument("--segmentation_model", type=str, required=True, help="Path to segmentation .h5 model (full model).")

    p.add_argument("--colorization_size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    p.add_argument("--segmentation_size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))

    p.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    p.add_argument("--n_classes", type=int, default=3, help="Number of segmentation classes.")
    p.add_argument("--save_overlay", action="store_true", help="If set, also save overlay images.")
    p.add_argument("--overlay_alpha", type=float, default=0.45, help="Alpha for overlay blending.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    out_std_dir = output_dir / "standardized_rgb"
    out_mask_dir = output_dir / "mask_rgb"
    out_overlay_dir = output_dir / "overlay"
    ensure_dir(out_std_dir)
    ensure_dir(out_mask_dir)
    if args.save_overlay:
        ensure_dir(out_overlay_dir)

    # Load models (full models)
    colorization_model = tf.keras.models.load_model(args.colorization_model, compile=False)
    segmentation_model = tf.keras.models.load_model(args.segmentation_model, compile=False)

    files = list_images(input_dir)
    if not files:
        raise RuntimeError(f"No images found in: {input_dir}")

    Hc, Wc = args.colorization_size
    Hs, Ws = args.segmentation_size

    # Process in batches
    bs = max(1, int(args.batch_size))
    for start in range(0, len(files), bs):
        batch_files = files[start : start + bs]

        # 1) Read and resize for colorization
        rgb_u8_list = []
        for f in batch_files:
            rgb_u8 = read_rgb_image(f)
            rgb_u8 = resize_rgb(rgb_u8, (Hc, Wc))
            rgb_u8_list.append(rgb_u8)

        rgb_u8_batch = np.stack(rgb_u8_list, axis=0)                      # (B,Hc,Wc,3) uint8
        rgb_f_batch = rgb_uint8_to_float01(rgb_u8_batch)                  # (B,Hc,Wc,3) float01

        # 2) Build 3xL input for colorization
        L3, L = build_L3_input(rgb_f_batch)                                # L3: (B,Hc,Wc,3), L:(B,Hc,Wc,1)

        # 3) Predict (a*, b*) and reconstruct standardized RGB
        ab_pred = colorization_model.predict(L3, batch_size=bs, verbose=0) # (B,Hc,Wc,2) expected
        rgb_std_f = reconstruct_rgb_from_L_and_ab(L, ab_pred)              # (B,Hc,Wc,3) float01
        rgb_std_u8 = float01_to_uint8(rgb_std_f)                           # (B,Hc,Wc,3) uint8

        # 4) Resize standardized image for segmentation and convert to float01
        rgb_seg_u8 = np.stack([resize_rgb(im, (Hs, Ws)) for im in rgb_std_u8], axis=0)  # (B,Hs,Ws,3)
        rgb_seg_f = rgb_uint8_to_float01(rgb_seg_u8)                                     # (B,Hs,Ws,3)

        # 5) Segmentation prediction
        _, mask_rgb_u8_batch = predict_segmentation(
            segmentation_model=segmentation_model,
            input_rgb_batch_float01=rgb_seg_f,
            n_classes=int(args.n_classes),
            batch_size=bs,
        )

        # 6) Save results
        for i, f in enumerate(batch_files):
            stem = f.stem

            # Save standardized RGB at colorization resolution (Hc,Wc)
            std_path = out_std_dir / f"{stem}_std.png"
            cv2.imwrite(str(std_path), cv2.cvtColor(rgb_std_u8[i], cv2.COLOR_RGB2BGR))

            # Save mask visualization at segmentation resolution (Hs,Ws)
            mask_path = out_mask_dir / f"{stem}_mask.png"
            cv2.imwrite(str(mask_path), cv2.cvtColor(mask_rgb_u8_batch[i], cv2.COLOR_RGB2BGR))

            # Optional overlay
            if args.save_overlay:
                ov = overlay_mask(rgb_seg_u8[i], mask_rgb_u8_batch[i], alpha=float(args.overlay_alpha))
                ov_path = out_overlay_dir / f"{stem}_overlay.png"
                cv2.imwrite(str(ov_path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    # Make TF a bit quieter (optional)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
