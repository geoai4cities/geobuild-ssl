#!/usr/bin/env python3
"""
Usage:
    python src/inference/geobuild_inference.py --image path/to/image.jpg
    python src/inference/geobuild_inference.py --folder path/to/images/ --output results/
"""

import ssl  # noqa: must precede src/ path additions
import os, sys, json, argparse, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ssl"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "detection"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
# After cloning, download the trained weights from the Google Drive link in
# final_models/README.md and place each file at the path shown below
# (relative to the repo root). They are not tracked in git.
DETECTION_CKPT = str(PROJECT_ROOT / "final_models" / "detection_model" / "best.pth")
CLASSIFICATION_CKPT = str(PROJECT_ROOT / "final_models" / "classification_model" / "v5_best.pth")
SSL_BACKBONE = str(PROJECT_ROOT / "final_models" / "ssl_model" / "dinov3_rgbd_ssl.pth")

CLASSES = ["residential", "commercial", "industrial", "institutional"]
CLASS_COLORS = {
    "residential": (76, 175, 80),     # green
    "commercial":  (33, 150, 243),    # blue
    "industrial":  (255, 152, 0),     # orange
    "institutional": (156, 39, 176),  # purple
}
IMG_SIZE = 512
MIN_BUILDING_AREA_PCT = 0.01  # minimum 1% of image
SHOW_CONFIDENCE = True  # draw "Class XX%"; set False for clean class-only labels


def norm_depth(d: np.ndarray) -> np.ndarray:
    """Per-patch percentile normalization (2nd–98th pct → [0, 1]).

    Must match the depth-map FiLM training normalization so the depth encoder
    sees the same input distribution as at training time.
    """
    d = d.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# DEPTH ESTIMATION
# ════════════════════════════════════════════════════════════════════

class DepthEstimator:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.processor = None
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.processor = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Large-hf", use_fast=False)
            self.model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Large-hf").to(device).eval()
            print("  [OK] Depth Anything V2")
        except Exception as e:
            print(f"  [WARN] Depth unavailable: {e}")

    @torch.no_grad()
    def estimate(self, image_pil):
        w, h = image_pil.size
        if self.model is not None:
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            raw = self.model(**inputs).predicted_depth
            raw = F.interpolate(raw.unsqueeze(1), size=(h, w), mode="bicubic",
                                align_corners=False).squeeze().cpu().numpy()
            raw = np.clip(raw, 0, None)
            p_hi, p_lo = np.percentile(raw, 99.5), np.percentile(raw, 0.5)
            inv = 1.0 - np.clip((raw - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
            return (3.0 + inv * 32.0).astype(np.float32)
        y = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)
        return (15.0 + (y / h) * 15.0).astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# BUILDING DETECTION
# ════════════════════════════════════════════════════════════════════

class BuildingDetector:
    def __init__(self, device, ckpt_path, backbone_path):
        self.device = device
        from train_building_segmentation import BuildingSegModel
        self.model = BuildingSegModel(backbone_path).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()
        self.iou = ckpt.get("best_iou", 0)
        print(f"  [OK] Building detector (IoU={self.iou:.3f})")

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _make_depth_3ch(depth_map):
        d = cv2.resize(depth_map, (IMG_SIZE, IMG_SIZE))
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        sx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        e = np.sqrt(sx**2 + sy**2); e /= (e.max() + 1e-6)
        l = cv2.Laplacian(d, cv2.CV_32F, ksize=3)
        r = l.max() - l.min()
        if r > 0: l = (l - l.min()) / r
        return torch.from_numpy(np.stack([d, e, l], axis=0).astype(np.float32))

    @torch.no_grad()
    def detect(self, image_pil, depth_map):
        h, w = np.array(image_pil).shape[:2]
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)
        depth_t = self._make_depth_3ch(depth_map).unsqueeze(0).to(self.device)

        l512, _, _, bnd_pred = self.model(img_t, depth_t)
        bld_mask = (torch.sigmoid(l512) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        # Lower boundary threshold → more aggressive separation
        bnd_mask = (torch.sigmoid(bnd_pred) > 0.15).squeeze().cpu().numpy().astype(np.uint8)

        bld_full = cv2.resize(bld_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        bnd_full = cv2.resize(bnd_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Dilate boundaries to ensure clean separation between adjacent buildings
        kernel = np.ones((3, 3), np.uint8)
        bnd_full = cv2.dilate(bnd_full, kernel, iterations=2)

        # Separate instances: building - boundary
        separated = bld_full.copy()
        separated[bnd_full > 0] = 0

        # Erode briefly then dilate back to break thin connections
        ek = np.ones((5, 5), np.uint8)
        separated = cv2.erode(separated, ek, iterations=1)
        separated = cv2.dilate(separated, ek, iterations=1)

        n_labels, labels = cv2.connectedComponents(separated)
        min_area = MIN_BUILDING_AREA_PCT * h * w

        buildings = []
        for i in range(1, n_labels):
            mask = (labels == i).astype(np.uint8)
            if mask.sum() < min_area:
                continue
            ys, xs = np.where(mask > 0)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            # Mean depth for this building
            d_crop = depth_map[y1:y2, x1:x2]
            mean_d = float(d_crop.mean()) if d_crop.size > 0 else 99.0

            buildings.append({
                "id": len(buildings) + 1,
                "bbox": [x1, y1, x2, y2],
                "mask": mask,
                "area_pct": round(float(mask.sum()) / (h * w) * 100, 1),
                "depth_m": round(mean_d, 1),
            })

        return buildings


# ════════════════════════════════════════════════════════════════════
# OCR + TEXT EMBEDDING
# ════════════════════════════════════════════════════════════════════

class OCREncoder:
    def __init__(self, device):
        self.device = device
        self.ocr = None
        self.clip_model = None
        try:
            from paddleocr import PaddleOCR
            try:
                # PaddleOCR >= 3.x (show_log / use_gpu / use_angle_cls removed)
                self.ocr = PaddleOCR(use_textline_orientation=True, lang="japan")
            except (TypeError, ValueError):
                # PaddleOCR 2.x fallback
                self.ocr = PaddleOCR(use_angle_cls=True, lang="japan", show_log=False)
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            print("  [OK] PaddleOCR + CLIP")
        except Exception as e:
            print(f"  [WARN] OCR unavailable: {e}")

    def _ocr_one(self, img):
        """Run PaddleOCR on a single image, return (texts, mean_conf).

        Handles both PaddleOCR 3.x (``.predict`` -> list of result dicts with
        ``rec_texts``/``rec_scores``) and 2.x (``.ocr(cls=True)`` -> nested
        ``[[ [box, (text, conf)], ... ]]``).
        """
        texts, confs = [], []
        try:
            result = self.ocr.predict(img)            # PaddleOCR 3.x
        except AttributeError:
            result = self.ocr.ocr(img, cls=True)      # PaddleOCR 2.x

        if not result:
            return texts, 0.0
        r0 = result[0]

        if hasattr(r0, "get") and r0.get("rec_texts") is not None:
            # 3.x: dict-like OCRResult
            pairs = zip(r0.get("rec_texts", []), r0.get("rec_scores", []) or [])
        else:
            # 2.x: r0 is a list of [box, (text, conf)] lines
            pairs = ((line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1]),
                      line[1][1] if isinstance(line[1], (list, tuple)) else 0.5)
                     for line in (r0 or []))

        for t, c in pairs:
            t = str(t).strip()
            c = float(c)
            if c > 0.3 and len(t) > 1:
                texts.append(t)
                confs.append(c)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return texts, mean_conf

    @torch.no_grad()
    def extract(self, crop_bgr):
        if self.ocr is None or self.clip_model is None:
            return np.zeros(512, dtype=np.float32), []
        try:
            # Try original orientation
            texts_orig, conf_orig = self._ocr_one(crop_bgr)

            # Try horizontal flip (mirrored text from rear-facing views)
            import cv2
            crop_flip = cv2.flip(crop_bgr, 1)
            texts_flip, conf_flip = self._ocr_one(crop_flip)

            # Keep the better result
            texts = texts_flip if conf_flip > conf_orig else texts_orig

            if not texts:
                return np.zeros(512, dtype=np.float32), []
            import clip
            tokens = clip.tokenize([" ".join(texts[:5])], truncate=True).to(self.device)
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten().astype(np.float32), texts
        except Exception:
            return np.zeros(512, dtype=np.float32), []


# ════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ════════════════════════════════════════════════════════════════════

class BuildingClassifier:
    def __init__(self, device, cls_ckpt, backbone_ckpt):
        self.device = device
        from classification.train_v5_curriculum import V5BuildingClassifier
        self.model = V5BuildingClassifier()
        self.model.load_backbone(backbone_ckpt, freeze_blocks_until=0)
        state = torch.load(cls_ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(device).eval()
        print("  [OK] V5 Classifier")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def classify(self, crop_pil, depth_patch, text_emb):
        rgb = self.transform(crop_pil).unsqueeze(0).to(self.device)
        # depth_patch: HxW relative-depth crop → [1,1,224,224] per-patch
        # percentile-normalized to [0,1], matching the depth-map FiLM training input.
        dp  = cv2.resize(depth_patch.astype(np.float32), (224, 224))
        dp  = norm_depth(dp)
        dep = torch.from_numpy(dp).float().unsqueeze(0).unsqueeze(0).to(self.device)
        txt = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        out = self.model(rgb, dep, txt)
        probs = F.softmax(out["logits"], dim=1)[0].cpu().numpy()
        pid = int(probs.argmax())
        return {
            "class": CLASSES[pid],
            "confidence": round(float(probs[pid]), 4),
            "probabilities": {c: round(float(probs[i]), 4) for i, c in enumerate(CLASSES)},
        }


# ════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════════

# Distinct color shades per class so adjacent same-class buildings differ
CLASS_COLOR_VARIANTS = {
    "residential": [
        (76, 175, 80),    # green
        (139, 195, 74),   # light green
        (46, 125, 50),    # dark green
        (104, 159, 56),   # mid green
        (165, 214, 167),  # pale green
    ],
    "commercial": [
        (243, 150, 33),   # blue (BGR: orange in RGB)
        (255, 167, 38),   # lighter
        (251, 192, 45),   # amber
        (255, 138, 101),  # coral
        (255, 87, 34),    # deep orange
    ],
    "industrial": [
        (0, 152, 255),    # orange (BGR)
        (33, 150, 243),   # blue
        (3, 169, 244),    # light blue
        (2, 119, 189),    # dark blue
        (79, 195, 247),   # pale blue
    ],
    "institutional": [
        (176, 39, 156),   # purple (BGR)
        (171, 71, 188),   # light purple
        (123, 31, 162),   # deep purple
        (186, 104, 200),  # mid
        (149, 117, 205),  # lavender
    ],
}


def draw_usage_map(img_bgr, buildings):
    """Clean visualization: distinct colors per building + small readable labels."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Track per-class instance for color variation
    class_counts = {}

    for b in buildings:
        cls = b.get("class", "unknown")
        conf = b.get("confidence", 0)
        mask = b["mask"]
        x1, y1, x2, y2 = b["bbox"]
        bw, bh = x2 - x1, y2 - y1

        # Pick a color variant
        variants = CLASS_COLOR_VARIANTS.get(cls, [CLASS_COLORS.get(cls, (128, 128, 128))])
        idx = class_counts.get(cls, 0)
        color = variants[idx % len(variants)]
        class_counts[cls] = idx + 1

        # Colored mask overlay
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, 0.40, vis, 1.0, 0, vis)

        # Border
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # ── Small clean label like reference ──
        label = f"{cls.capitalize()} {conf*100:.0f}%" if SHOW_CONFIDENCE else f"{cls.capitalize()}"
        lbl_sc = 0.55
        lbl_th = 1
        (tw, txh), _ = cv2.getTextSize(label, font, lbl_sc, lbl_th)
        pad_x = 6
        pad_y = 4

        # Place label at top-left inside the building bbox
        lbl_x = x1 + 4
        lbl_y = y1 + 4

        # Clamp to image
        if lbl_x + tw + 2 * pad_x > w:
            lbl_x = w - tw - 2 * pad_x - 2
        if lbl_y + txh + 2 * pad_y > h:
            lbl_y = h - txh - 2 * pad_y - 2
        lbl_x = max(2, lbl_x)
        lbl_y = max(2, lbl_y)

        # Solid colored background pill
        bg_x1 = lbl_x
        bg_y1 = lbl_y
        bg_x2 = lbl_x + tw + 2 * pad_x
        bg_y2 = lbl_y + txh + 2 * pad_y
        cv2.rectangle(vis, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # White text on the colored background
        cv2.putText(vis, label, (lbl_x + pad_x, lbl_y + txh + pad_y - 1),
                    font, lbl_sc, (255, 255, 255), lbl_th + 1, cv2.LINE_AA)

    return vis


# ════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════

class GeoBuildPipeline:
    def __init__(self):
        print("=" * 60)
        print("GeoBuild-SSL Inference Pipeline")
        print(f"  Device: {DEVICE}")
        print("=" * 60)
        print("Loading models...")
        self.depth = DepthEstimator(DEVICE)
        self.detector = BuildingDetector(DEVICE, DETECTION_CKPT, SSL_BACKBONE)
        self.ocr = OCREncoder(DEVICE)
        self.classifier = BuildingClassifier(DEVICE, CLASSIFICATION_CKPT, SSL_BACKBONE)
        print("All models loaded.\n")

    def process_image(self, image_path, output_dir=None):
        image_path = Path(image_path)
        print(f"  Processing: {image_path.name}")
        t0 = time.time()

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"    [SKIP] Cannot read: {image_path}")
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        h, w = img_bgr.shape[:2]

        # 1. Depth
        depth_map = self.depth.estimate(img_pil)

        # 2. Detection (building mask + boundary → instances)
        buildings = self.detector.detect(img_pil, depth_map)
        print(f"    Detected {len(buildings)} buildings")

        # 3+4. For each building: OCR + Classification
        for b in buildings:
            x1, y1, x2, y2 = b["bbox"]
            crop_bgr = img_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

            # OCR
            text_emb, ocr_texts = self.ocr.extract(crop_bgr)
            b["ocr_texts"] = ocr_texts

            # Classify — feed the depth-map *patch* for this building (paper FiLM),
            # cropping the full-image depth map to the building bbox.
            depth_patch = depth_map[y1:y2, x1:x2]
            if depth_patch.size == 0:
                depth_patch = np.full((224, 224), b["depth_m"], dtype=np.float32)
            cls_result = self.classifier.classify(crop_pil, depth_patch, text_emb)
            b.update(cls_result)

        elapsed = time.time() - t0
        cls_strs = [f"{b['class']}({b['confidence']:.0%})" for b in buildings if "class" in b]
        print(f"    Classes: {cls_strs}")
        print(f"    Time: {elapsed:.1f}s")

        # 5. Save outputs
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Usage map
            usage_map = draw_usage_map(img_bgr, buildings)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_usage_map.jpg"),
                        usage_map, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # JSON
            result = {
                "image": str(image_path),
                "image_size": [w, h],
                "n_buildings": len(buildings),
                "time_s": round(elapsed, 2),
                "buildings": [{k: v for k, v in b.items() if k != "mask"}
                              for b in buildings],
            }
            with open(output_dir / f"{image_path.stem}_results.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return buildings

    def process_folder(self, folder, output_dir):
        folder = Path(folder)
        images = sorted([f for f in folder.rglob("*")
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        print(f"\nFound {len(images)} images in {folder}\n")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(images):
            # Create a unique subdirectory mirroring the source tree
            # e.g. train/easy/india/delhi/ → output_dir/train_easy_india_delhi/
            rel = img_path.relative_to(folder).parent
            sub_dir = output_dir / str(rel).replace(os.sep, "_")
            sub_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{i+1}/{len(images)}]", end="")
            self.process_image(img_path, sub_dir)

        print(f"\nDone. Results saved to {output_dir}")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GeoBuild-SSL Building Usage Mapping")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--folder", type=str, help="Folder of images")
    parser.add_argument("--output", type=str, default="results/inference",
                        help="Output directory")
    parser.add_argument("--no-confidence", action="store_true",
                        help="Draw class-only labels (omit the XX%% confidence text)")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help()
        return

    if args.no_confidence:
        global SHOW_CONFIDENCE
        SHOW_CONFIDENCE = False

    pipeline = GeoBuildPipeline()

    if args.image:
        pipeline.process_image(args.image, args.output)
    elif args.folder:
        pipeline.process_folder(args.folder, args.output)


if __name__ == "__main__":
    main()
