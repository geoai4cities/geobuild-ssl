"""

Usage:
    python src/annotation/sam3_detect.py \
        --images data/curriculum/ \
        --depth  data/curriculum_depths/ \
        --output data/annotations/sam3_detections.csv \
        --depth_threshold 30.0 \
        --resume

"""

import argparse, hashlib, os, re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from transformers import AutoProcessor, SAM2Model
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("[WARN] transformers not available — SAM3 disabled, will use bbox heuristics")

try:
    from ultralytics import SAM
    ULTRALYTICS_SAM = True
except ImportError:
    ULTRALYTICS_SAM = False


SPLITS  = ["train", "val", "test"]
DIFFS   = ["easy", "medium", "hard"]
CLASSES = ["residential", "commercial", "industrial", "institutional"]


# ── Depth helpers ───────────────────────────────────────────────────────────

def load_depth(depth_root: Path, split: str, diff: str, img_stem: str) -> np.ndarray | None:
    """Try to load depth map (.npy or .png) for a given image stem."""
    depth_dir = depth_root / split / diff / "depth"
    for ext in [".npy", ".png"]:
        p = depth_dir / f"{img_stem}{ext}"
        if p.exists():
            if ext == ".npy":
                return np.load(p).astype(np.float32)
            else:
                raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    return raw.astype(np.float32) / 255.0 * 35.0  # normalise to meters
    return None


def compute_crop_depth(depth: np.ndarray, x1: int, y1: int,
                       x2: int, y2: int) -> float | None:
    if depth is None:
        return None
    patch = depth[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    valid = patch[patch > 0.2]
    return float(np.median(valid)) if valid.size > 0 else None


# ── Geometry helpers ─────────────────────────────────────────────────────────

def extract_yaw(filename: str) -> float | None:
    """Extract yaw angle from filename like 1_yaw_090p00_pitch_000p00.jpg"""
    m = re.search(r"yaw_(\d{3})p(\d{2})", filename)
    if m:
        deg = int(m.group(1))
        dec = int(m.group(2))
        return deg + dec / 100
    return None


def bbox_to_difficulty(yaw: float | None) -> str:
    if yaw is None:
        return "medium"
    easy   = {67.5, 90.0, 112.5, 247.5, 270.0, 292.5}
    hard   = {0.0, 157.5, 180.0, 202.5}
    if yaw in easy:   return "easy"
    if yaw in hard:   return "hard"
    return "medium"


def parse_country_city(img_path: Path, root: Path) -> tuple[str, str]:
    """Infer country/city from path structure: .../split/diff/country/city/img.jpg"""
    try:
        rel = img_path.relative_to(root)
        parts = rel.parts  # (split, diff, [country, city,] img)
        if len(parts) >= 4:
            return parts[2], parts[3]
        elif len(parts) >= 3:
            return parts[2], "unknown"
    except ValueError:
        pass
    return "unknown", "unknown"


# ── SAM3 detection ───────────────────────────────────────────────────────────

class BuildingDetector:
    """Detects buildings using SAM3 + text prompt or fallback heuristics."""

    def __init__(self, method: str = "auto"):
        self.method = method
        self.model  = None

        if method == "auto":
            if ULTRALYTICS_SAM:
                self._init_ultralytics()
            elif SAM3_AVAILABLE:
                self._init_hf()
            else:
                print("[INFO] No SAM available — using edge-based fallback detector")
                self.method = "fallback"
        elif method == "ultralytics":
            self._init_ultralytics()
        elif method == "hf":
            self._init_hf()
        else:
            self.method = "fallback"

    def _init_ultralytics(self):
        print("[INFO] Using Ultralytics SAM2 model")
        self.model  = SAM("sam2_l.pt")
        self.method = "ultralytics"

    def _init_hf(self):
        print("[INFO] Using HuggingFace SAM2 model")
        self.processor = AutoProcessor.from_pretrained("facebook/sam2-hiera-large")
        self.model     = SAM2Model.from_pretrained("facebook/sam2-hiera-large")
        self.method    = "hf"

    def detect(self, image_bgr: np.ndarray,
               depth_threshold: float = 30.0,
               depth_map: np.ndarray | None = None) -> list[dict]:
        """Detect buildings. Returns list of dicts with bbox and mask info."""
        if self.method == "ultralytics":
            return self._detect_ultralytics(image_bgr, depth_threshold, depth_map)
        elif self.method == "hf":
            return self._detect_hf(image_bgr, depth_threshold, depth_map)
        else:
            return self._detect_fallback(image_bgr, depth_threshold, depth_map)

    def _detect_ultralytics(self, img, depth_thresh, depth):
        results = self.model(img, texts="building", verbose=False)
        detections = []
        for r in results:
            if r.masks is None:
                continue
            for i, (box, mask) in enumerate(zip(r.boxes.xyxy, r.masks.data)):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                mask_np   = mask.cpu().numpy().astype(np.uint8)
                mask_area = int(mask_np.sum())
                if mask_area < 400:
                    continue
                w, h  = x2 - x1, y2 - y1
                depth_val = compute_crop_depth(depth, x1, y1, x2, y2)
                if depth_val is not None and depth_val > depth_thresh:
                    continue
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "mask_area_pixels": mask_area,
                    "aspect_ratio": round(w / h, 3) if h > 0 else 1.0,
                    "mean_depth_m": round(depth_val, 2) if depth_val else None,
                    "depth_valid": depth_val is not None,
                })
        return detections

    def _detect_hf(self, img, depth_thresh, depth):
        # HuggingFace SAM2 path (text-prompted)
        import torch
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text_inputs=[["building"]], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
        detections = []
        for mask in masks[0]:
            mask_np   = mask.cpu().numpy().astype(np.uint8)
            mask_area = int(mask_np.sum())
            if mask_area < 400:
                continue
            ys, xs   = np.where(mask_np > 0)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            w, h = x2 - x1, y2 - y1
            depth_val = compute_crop_depth(depth, x1, y1, x2, y2)
            if depth_val is not None and depth_val > depth_thresh:
                continue
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "mask_area_pixels": mask_area,
                "aspect_ratio": round(w / h, 3) if h > 0 else 1.0,
                "mean_depth_m": round(depth_val, 2) if depth_val else None,
                "depth_valid": depth_val is not None,
            })
        return detections

    def _detect_fallback(self, img, depth_thresh, depth):
        """
        Fallback: Canny edge + contour detection for building-like regions.
        Less accurate than SAM3 but works without model weights.
        """
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150)
        kernel  = np.ones((10, 10), np.uint8)
        closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        H, W    = img.shape[:2]
        min_area = 0.01 * H * W   # at least 1% of image
        max_area = 0.80 * H * W

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x + w, y + h
            depth_val = compute_crop_depth(depth, x1, y1, x2, y2)
            if depth_val is not None and depth_val > depth_thresh:
                continue
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "mask_area_pixels": int(area),
                "aspect_ratio": round(w / h, 3) if h > 0 else 1.0,
                "mean_depth_m": round(depth_val, 2) if depth_val else None,
                "depth_valid": depth_val is not None,
            })
        return detections


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",          default="data/curriculum/")
    parser.add_argument("--depth",           default="data/curriculum_depths/")
    parser.add_argument("--output",          default="data/annotations/sam3_detections.csv")
    parser.add_argument("--depth_threshold", type=float, default=30.0)
    parser.add_argument("--method",          default="auto",
                        choices=["auto", "ultralytics", "hf", "fallback"])
    parser.add_argument("--resume",          action="store_true",
                        help="Skip already processed images")
    args = parser.parse_args()

    image_root = Path(args.images)
    depth_root = Path(args.depth)
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    done_imgs = set()
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        done_imgs = set(existing["image_path"].tolist())
        print(f"[Resume] {len(done_imgs)} images already processed")

    detector = BuildingDetector(method=args.method)
    rows = []

    for split in SPLITS:
        for diff in DIFFS:
            img_dir = image_root / split / diff
            if not img_dir.exists():
                continue
            imgs = sorted(list(img_dir.glob("**/*.jpg")) + list(img_dir.glob("**/*.png")))
            for img_path in tqdm(imgs, desc=f"{split}/{diff}", leave=False):
                img_str = str(img_path)
                if img_str in done_imgs:
                    continue

                img = cv2.imread(img_str)
                if img is None:
                    continue

                depth_map = load_depth(depth_root, split, diff, img_path.stem)
                yaw       = extract_yaw(img_path.name)
                country, city = parse_country_city(img_path, image_root)

                detections = detector.detect(img, args.depth_threshold, depth_map)

                for det_i, det in enumerate(detections):
                    # Unique building ID: hash of image path + bbox
                    uid = hashlib.md5(
                        f"{img_str}_{det['x1']}_{det['y1']}_{det['x2']}_{det['y2']}".encode()
                    ).hexdigest()[:12]
                    rows.append({
                        "image_path":       img_str,
                        "split":            split,
                        "difficulty":       diff,
                        "country":          country,
                        "city":             city,
                        "building_id":      uid,
                        "x1":               det["x1"],
                        "y1":               det["y1"],
                        "x2":               det["x2"],
                        "y2":               det["y2"],
                        "mask_area_pixels": det["mask_area_pixels"],
                        "aspect_ratio":     det["aspect_ratio"],
                        "mean_depth_m":     det["mean_depth_m"],
                        "depth_valid":      det["depth_valid"],
                        "yaw_angle":        yaw,
                    })

    # Merge with any existing results
    df_new = pd.DataFrame(rows)
    if args.resume and out_path.exists():
        df_new = pd.concat([pd.read_csv(out_path), df_new], ignore_index=True)

    df_new.to_csv(out_path, index=False)
    print(f"\nDetections saved: {out_path}  ({len(df_new)} buildings)")

    # Summary
    print(f"\nStats:")
    for split in SPLITS:
        n = (df_new["split"] == split).sum() if "split" in df_new.columns else 0
        print(f"  {split}: {n} buildings")
    if "mean_depth_m" in df_new.columns:
        within_30 = (df_new["mean_depth_m"] < 30).sum()
        print(f"  Within 30m: {within_30} ({100*within_30/len(df_new):.1f}%)")


if __name__ == "__main__":
    main()
