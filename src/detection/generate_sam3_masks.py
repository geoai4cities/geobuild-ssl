#!/usr/bin/env python3
"""

Usage:
    python src/detection/generate_sam3_masks.py
"""

import os, sys, csv, time, glob
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRICULUM_ROOT = PROJECT_ROOT / "data" / "curriculum"
DEPTH_ROOT      = PROJECT_ROOT / "data" / "curriculum_depths"
MASK_OUT_DIR    = PROJECT_ROOT / "data" / "annotations" / "sam3_building_masks"
CSV_OUT         = PROJECT_ROOT / "data" / "annotations" / "sam3_building_instances.csv"

TEXT_PROMPT          = "building"
DEPTH_THRESHOLD      = 30.0
MAX_IMAGE_SIZE       = 1024
MASK_THRESHOLD       = 0.35
INSTANCE_THRESHOLD   = 0.25


def load_depth(depth_root, rel_path):
    """Load depth map (.npy or .png) for a curriculum image."""
    for ext in [".npy", ".png"]:
        dp = depth_root / (rel_path.with_suffix(ext))
        if dp.exists():
            if ext == ".npy":
                return np.load(str(dp)).astype(np.float32)
            else:
                raw = cv2.imread(str(dp), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    return raw.astype(np.float32) / 255.0 * 35.0
    return None


def compute_crop_depth(depth, x1, y1, x2, y2):
    if depth is None:
        return None
    patch = depth[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    valid = patch[patch > 0.2]
    return float(np.median(valid)) if valid.size > 0 else None


def main():
    from transformers import Sam3Processor, Sam3Model

    MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SAM3 model...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
    model.eval()
    print(f"[OK] SAM3 loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    # Collect all curriculum images
    all_images = sorted(CURRICULUM_ROOT.rglob("*.jpg"))
    print(f"Found {len(all_images)} curriculum images")

    csv_file = open(CSV_OUT, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "source_image", "instance_id", "x1", "y1", "x2", "y2",
        "mask_area_pixels", "mean_depth", "mask_file"
    ])

    total_instances = 0
    images_with_dets = 0
    t0 = time.time()

    for idx, img_path in enumerate(all_images):
        try:
            # Load and resize image
            rgb_img = Image.open(img_path).convert("RGB")
            w, h = rgb_img.size
            if max(w, h) > MAX_IMAGE_SIZE:
                scale = MAX_IMAGE_SIZE / max(w, h)
                rgb_img = rgb_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            rgb = np.array(rgb_img)
            rh, rw = rgb.shape[:2]

            # Load depth
            rel = img_path.relative_to(CURRICULUM_ROOT)
            depth = load_depth(DEPTH_ROOT, rel)
            if depth is not None:
                depth = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_LINEAR)

            # Run SAM3 with text prompt "building"
            inputs = processor(
                images=rgb_img,
                text=TEXT_PROMPT,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=INSTANCE_THRESHOLD,
                mask_threshold=MASK_THRESHOLD,
                target_sizes=[rgb.shape[:2]]
            )[0]

            if len(results["masks"]) == 0:
                # Save empty mask
                mask_file = str(rel.with_suffix(".npy"))
                out_path = MASK_OUT_DIR / mask_file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(out_path), np.zeros((rh, rw), dtype=np.uint8))
                continue

            images_with_dets += 1

            # Process each instance mask
            combined_mask = np.zeros((rh, rw), dtype=np.uint8)
            img_instances = 0

            for inst_idx, mask_tensor in enumerate(results["masks"]):
                mask = mask_tensor.cpu().numpy().astype(np.uint8)

                # Get bbox from mask
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

                # Mask area
                mask_area = int(mask.sum())
                if mask_area < 100:  # too small
                    continue

                # Depth filter
                mean_d = compute_crop_depth(depth, x1, y1, x2, y2)
                if mean_d is not None and mean_d > DEPTH_THRESHOLD:
                    continue

                # Add to combined mask with INSTANCE ID (not binary)
                # Each building gets unique ID: 1, 2, 3, ...
                combined_mask[mask > 0] = img_instances + 1
                img_instances += 1
                total_instances += 1

                # Scale bbox back to original image coords
                if max(w, h) > MAX_IMAGE_SIZE:
                    inv_scale = max(w, h) / MAX_IMAGE_SIZE
                    ox1 = int(x1 * inv_scale)
                    oy1 = int(y1 * inv_scale)
                    ox2 = int(x2 * inv_scale)
                    oy2 = int(y2 * inv_scale)
                else:
                    ox1, oy1, ox2, oy2 = x1, y1, x2, y2

                instance_id = f"{rel.stem}_bld_{inst_idx:03d}"
                writer.writerow([
                    str(img_path), instance_id, ox1, oy1, ox2, oy2,
                    mask_area, round(mean_d, 2) if mean_d else 0,
                    str(rel.with_suffix(".npy"))
                ])

            # Save combined binary mask
            mask_file = str(rel.with_suffix(".npy"))
            out_path = MASK_OUT_DIR / mask_file
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), combined_mask)

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(all_images) - idx - 1) / rate
            print(f"  [{idx+1}/{len(all_images)}] instances={total_instances} "
                  f"({rate:.1f} img/s, ETA {eta/60:.1f}min)")

    csv_file.close()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"SAM3 MASK GENERATION COMPLETE")
    print(f"  Images processed : {len(all_images)}")
    print(f"  Images with dets : {images_with_dets}")
    print(f"  Total instances  : {total_instances}")
    print(f"  Avg per image    : {total_instances/max(images_with_dets,1):.1f}")
    print(f"  Time             : {elapsed/60:.1f} min")
    print(f"  Masks            : {MASK_OUT_DIR}")
    print(f"  CSV              : {CSV_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
