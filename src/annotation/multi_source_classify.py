"""

Usage:
    python src/annotation/multi_source_classify.py \
        --detections data/annotations/sam3_detections.csv \
        --images     data/curriculum/ \
        --output     data/annotations/multi_source_labels.csv \
        --resume
"""

import argparse, re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ── CLIP ─────────────────────────────────────────────────────────────────────
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    CLIP_AVAILABLE = False
    print("[WARN] CLIP not available — install with: pip install git+https://github.com/openai/CLIP.git")

# ── PaddleOCR ─────────────────────────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("[WARN] PaddleOCR not available — install with: pip install paddleocr")

CLASSES = ["residential", "commercial", "industrial", "institutional"]


# ── CLIP classifier ───────────────────────────────────────────────────────────

CLIP_TEMPLATES = {
    "residential": [
        "a photo of a residential building",
        "a photo of a house or apartment",
        "a photo of a flat or housing complex",
        "a photo of a dormitory or hostel",
        "a photo of a villa or bungalow",
        "a photo of a residential tower block",
    ],
    "commercial": [
        "a photo of a commercial building",
        "a photo of a shop or store front",
        "a photo of a hotel or restaurant",
        "a photo of a bank or office building",
        "a photo of a shopping mall or market",
        "a photo of a showroom or retail space",
    ],
    "industrial": [
        "a photo of an industrial building",
        "a photo of a factory or manufacturing plant",
        "a photo of a warehouse or storage depot",
        "a photo of a workshop or industrial facility",
        "a photo of a power plant or processing facility",
    ],
    "institutional": [
        "a photo of an institutional building",
        "a photo of a school or college campus",
        "a photo of a hospital or medical centre",
        "a photo of a government building",
        "a photo of a temple, church, or mosque",
        "a photo of a police station or courthouse",
    ],
}


class CLIPClassifier:
    def __init__(self, model_names: list[str] = ("ViT-B/32", "ViT-L/14")):
        if not CLIP_AVAILABLE:
            self.available = False
            return
        self.available = True
        self.models    = []
        self.preproc   = []
        for name in model_names:
            m, p = clip.load(name, device=DEVICE)
            m.eval()
            self.models.append(m)
            self.preproc.append(p)
            print(f"  Loaded CLIP {name}")

        # Pre-encode all text templates
        self.text_features = []
        for model in self.models:
            feats = {}
            for cls, templates in CLIP_TEMPLATES.items():
                tokens = clip.tokenize(templates).to(DEVICE)
                with torch.no_grad():
                    f = model.encode_text(tokens)
                    f = f / f.norm(dim=-1, keepdim=True)
                feats[cls] = f.mean(dim=0)   # average over templates
            self.text_features.append(feats)

    def classify_crop(self, crop_bgr: np.ndarray) -> dict:
        """Returns per-class scores averaged across all loaded models."""
        if not self.available or crop_bgr is None or crop_bgr.size == 0:
            return {cls: 0.25 for cls in CLASSES}

        from PIL import Image
        rgb   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)

        all_scores = []
        for model, preproc, text_feats in zip(self.models, self.preproc, self.text_features):
            img_t = preproc(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_f = model.encode_image(img_t)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)

            sims = {}
            for cls in CLASSES:
                sims[cls] = float((img_f @ text_feats[cls].unsqueeze(-1)).squeeze())
            # Softmax over similarities
            vals = np.array([sims[c] for c in CLASSES])
            vals = np.exp(vals - vals.max())
            vals = vals / vals.sum()
            all_scores.append({c: float(v) for c, v in zip(CLASSES, vals)})

        # Average across models
        avg = {c: np.mean([s[c] for s in all_scores]) for c in CLASSES}
        return avg

    def models_agree(self, crop_bgr: np.ndarray) -> tuple[str, float, bool]:
        """Returns (pred_class, confidence, models_agree)."""
        if not self.available:
            return "residential", 0.25, False
        if len(self.models) < 2:
            scores = self.classify_crop(crop_bgr)
            pred   = max(scores, key=scores.get)
            return pred, scores[pred], True

        from PIL import Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        preds_per_model = []
        scores_per_model = []
        for model, preproc, text_feats in zip(self.models, self.preproc, self.text_features):
            img_t = preproc(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_f = model.encode_image(img_t)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            sims  = {c: float((img_f @ text_feats[c].unsqueeze(-1)).squeeze()) for c in CLASSES}
            vals  = np.array([sims[c] for c in CLASSES])
            vals  = np.exp(vals - vals.max()); vals /= vals.sum()
            scores = {c: float(v) for c, v in zip(CLASSES, vals)}
            preds_per_model.append(max(scores, key=scores.get))
            scores_per_model.append(scores)

        avg_scores = {c: np.mean([s[c] for s in scores_per_model]) for c in CLASSES}
        pred       = max(avg_scores, key=avg_scores.get)
        agree      = len(set(preds_per_model)) == 1
        conf       = avg_scores[pred]
        return pred, conf, agree


# ── OCR classifier ────────────────────────────────────────────────────────────

OCR_KEYWORDS = {
    "commercial": [
        "hotel", "restaurant", "cafe", "coffee", "shop", "store", "bank", "mall",
        "pharmacy", "salon", "market", "bakery", "showroom", "atm", "supermarket",
        "gym", "spa", "boutique", "plaza", "motors", "bar", "pub", "clinic",
        # Japanese
        "ホテル", "レストラン", "銀行", "ショップ", "コンビニ", "スーパー", "カフェ",
        # Hindi
        "होटल", "दुकान", "बैंक", "रेस्टोरेंट", "शॉप", "बाज़ार", "मार्ट",
    ],
    "institutional": [
        "hospital", "school", "college", "university", "church", "temple",
        "mosque", "police", "government", "court", "library", "academy",
        "institute", "ministry", "municipal", "public office",
        # Japanese
        "病院", "学校", "大学", "警察署", "寺", "神社", "図書館", "市役所", "区役所",
        # Hindi
        "अस्पताल", "स्कूल", "विद्यालय", "मंदिर", "पुलिस", "सरकार", "न्यायालय",
    ],
    "industrial": [
        "factory", "warehouse", "plant", "industry", "manufacturing",
        "ltd", "pvt", "works", "depot", "storage", "steel", "chemicals",
        "engineering", "industries", "corporation",
        # Japanese
        "工場", "倉庫", "製造", "工業", "産業",
        # Hindi
        "कारखाना", "गोदाम", "उद्योग", "प्राइवेट लिमिटेड",
    ],
}

OCR_MIN_CONFIDENCE = 0.70


class OCRClassifier:
    def __init__(self):
        self.available = PADDLE_AVAILABLE
        if PADDLE_AVAILABLE:
            def _make(lang):
                try:
                    # PaddleOCR >= 3.x (show_log / use_angle_cls removed)
                    return PaddleOCR(use_textline_orientation=True, lang=lang)
                except (TypeError, ValueError):
                    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            # Load once for all languages
            self.ocr_en = _make("en")
            self.ocr_ja = _make("japan")
            self.ocr_hi = _make("hi")
            print("  PaddleOCR loaded (en, ja, hi)")

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """CLAHE + upscale if too small."""
        h, w = crop.shape[:2]
        if h < 64:
            scale = 64 / h
            crop  = cv2.resize(crop, (int(w * scale), 64), interpolation=cv2.INTER_LANCZOS4)
        lab   = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        crop  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return crop

    def _run_ocr(self, crop: np.ndarray) -> list[tuple[str, float]]:
        """Run all 3 language OCR, return list of (text, confidence)."""
        if not self.available:
            return []
        crop = self._preprocess(crop)
        results = []
        for ocr_engine in [self.ocr_en, self.ocr_ja, self.ocr_hi]:
            try:
                try:
                    res = ocr_engine.predict(crop)        # PaddleOCR 3.x
                except AttributeError:
                    res = ocr_engine.ocr(crop, cls=True)  # PaddleOCR 2.x
                if not res or res[0] is None:
                    continue
                r0 = res[0]
                if hasattr(r0, "get") and r0.get("rec_texts") is not None:
                    pairs = zip(r0.get("rec_texts", []), r0.get("rec_scores", []) or [])
                else:
                    pairs = ((line[1][0], line[1][1]) for line in r0 if line is not None)
                for t, c in pairs:
                    text = str(t).strip().lower()
                    conf = float(c)
                    if conf >= OCR_MIN_CONFIDENCE and len(text) >= 2:
                        results.append((text, conf))
            except Exception:
                pass
        return results

    def classify_crop(self, crop_bgr: np.ndarray) -> dict:
        """
        Returns dict with:
          ocr_text, ocr_confidence, ocr_class, ocr_class_confidence, has_text
        """
        texts = self._run_ocr(crop_bgr)
        if not texts:
            return {"ocr_text": "", "ocr_confidence": 0.0,
                    "ocr_class": None, "ocr_class_confidence": 0.0, "has_text": False}

        full_text = " ".join([t for t, _ in texts])
        max_conf  = max([c for _, c in texts])

        # Match keywords
        matched = {}
        for cls, keywords in OCR_KEYWORDS.items():
            hits = [kw for kw in keywords if kw.lower() in full_text]
            if hits:
                matched[cls] = len(hits)

        if not matched:
            return {"ocr_text": full_text, "ocr_confidence": max_conf,
                    "ocr_class": None, "ocr_class_confidence": 0.0, "has_text": True}

        best_cls  = max(matched, key=matched.get)
        ocr_class_conf = min(max_conf, 0.95)  # cap at 0.95

        return {
            "ocr_text":             full_text[:200],
            "ocr_confidence":       round(max_conf, 3),
            "ocr_class":            best_cls,
            "ocr_class_confidence": round(ocr_class_conf, 3),
            "has_text":             True,
        }


# ── Geometric classifier ──────────────────────────────────────────────────────

def geometric_scores(aspect_ratio: float, mask_area_pixels: int,
                     mean_depth_m: float | None) -> dict[str, float]:
    """
    Rule-based soft scores from geometric features.
    Outputs: {class: score 0-1}
    These are NEVER used alone — only to break CLIP ties.
    """
    scores = {c: 0.25 for c in CLASSES}   # uniform prior

    # Industrial: wide and low, large footprint
    if aspect_ratio > 2.0 and mask_area_pixels > 50000:
        scores["industrial"] = 0.75
        scores["residential"] = 0.10
    elif aspect_ratio > 1.5 and mask_area_pixels > 30000:
        scores["industrial"] = 0.55

    # Residential: taller than wide, moderate area
    if 0.3 < aspect_ratio < 1.1 and 1000 < mask_area_pixels < 50000:
        scores["residential"] = max(scores["residential"], 0.55)

    # Commercial: variable ratio, tends to be wide
    if 0.8 < aspect_ratio < 2.0 and mask_area_pixels > 5000:
        scores["commercial"] = max(scores["commercial"], 0.40)

    # Institutional: large area, moderate aspect ratio
    if mask_area_pixels > 80000:
        scores["institutional"] = max(scores["institutional"], 0.50)

    # Normalise
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    return {k: round(v, 4) for k, v in scores.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def crop_from_image(img_path: str, x1: int, y1: int, x2: int, y2: int,
                    pad: int = 5) -> np.ndarray | None:
    img = cv2.imread(img_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(W, x2 + pad), min(H, y2 + pad)
    crop = img[y1p:y2p, x1p:x2p]
    return crop if crop.size > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", default="data/annotations/sam3_detections.csv")
    parser.add_argument("--output",     default="data/annotations/multi_source_labels.csv")
    parser.add_argument("--clip_models", nargs="+", default=["ViT-B/32", "ViT-L/14"])
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    det_path = Path(args.detections)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not det_path.exists():
        print(f"[ERROR] {det_path} not found. Run sam3_detect.py first.")
        return

    df = pd.read_csv(det_path)
    print(f"Loaded {len(df)} building detections")

    # Resume: skip already processed building_ids
    done_ids = set()
    if args.resume and out_path.exists():
        done_df  = pd.read_csv(out_path)
        done_ids = set(done_df["building_id"].tolist())
        print(f"[Resume] {len(done_ids)} already classified")

    clip_clf = CLIPClassifier(model_names=args.clip_models)
    ocr_clf  = OCRClassifier()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        if row["building_id"] in done_ids:
            continue

        crop = crop_from_image(row["image_path"],
                                int(row["x1"]), int(row["y1"]),
                                int(row["x2"]), int(row["y2"]))

        # Source A: CLIP ensemble
        clip_pred, clip_conf, clip_agree = clip_clf.models_agree(crop)
        clip_scores = clip_clf.classify_crop(crop) if crop is not None else {c: 0.25 for c in CLASSES}

        # Source B: OCR
        ocr_result = ocr_clf.classify_crop(crop)

        # Source C: Geometric
        geom_scores = geometric_scores(
            float(row.get("aspect_ratio", 1.0)),
            int(row.get("mask_area_pixels", 5000)),
            row.get("mean_depth_m"),
        )

        rec = {
            "building_id":   row["building_id"],
            "image_path":    row["image_path"],
            "split":         row.get("split", ""),
            "difficulty":    row.get("difficulty", ""),
            "country":       row.get("country", ""),
            "city":          row.get("city", ""),
            "x1":            row["x1"], "y1": row["y1"],
            "x2":            row["x2"], "y2": row["y2"],
            "mask_area_pixels": row.get("mask_area_pixels"),
            "aspect_ratio":  row.get("aspect_ratio"),
            "mean_depth_m":  row.get("mean_depth_m"),
            "yaw_angle":     row.get("yaw_angle"),
            # Source A
            "clip_pred":     clip_pred,
            "clip_conf":     round(clip_conf, 4),
            "clip_agree":    clip_agree,
            **{f"clip_prob_{c}": round(clip_scores.get(c, 0), 4) for c in CLASSES},
            # Source B
            **ocr_result,
            # Source C
            **{f"geom_score_{c}": geom_scores.get(c, 0.25) for c in CLASSES},
        }
        rows.append(rec)

    df_new = pd.DataFrame(rows)
    if args.resume and out_path.exists():
        df_new = pd.concat([pd.read_csv(out_path), df_new], ignore_index=True)

    df_new.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df_new)} rows)")

    # Stats
    if "clip_pred" in df_new.columns:
        print("\nCLIP prediction distribution:")
        print(df_new["clip_pred"].value_counts().to_string())
    if "has_text" in df_new.columns:
        has_text = df_new["has_text"].sum()
        print(f"\nBuildings with OCR text: {has_text}/{len(df_new)} "
              f"({100*has_text/len(df_new):.1f}%)")
    if "ocr_class" in df_new.columns:
        print("\nOCR class distribution (when text detected):")
        print(df_new[df_new["ocr_class"].notna()]["ocr_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
