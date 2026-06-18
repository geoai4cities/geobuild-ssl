"""
Usage:
    python src/ocr/extract_text.py
    python src/ocr/extract_text.py --labels data/annotations/final_labels.csv
    python src/ocr/extract_text.py --labels data/annotations/final_labels.csv \\
        --output data/ocr_outputs/ocr_results.csv --resume
"""

import argparse
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

CLASSES = ["residential", "commercial", "industrial", "institutional"]

# ── OCR keyword dictionary (English + Hindi + Japanese) ──────────────────────

OCR_KEYWORDS: dict[str, list[str]] = {
    "commercial": [
        # English
        "hotel", "resort", "lodge", "inn", "motel",
        "restaurant", "cafe", "coffee", "bakery", "kitchen", "food",
        "bar", "pub", "grill", "bistro", "diner",
        "shop", "store", "mart", "mall", "market", "plaza",
        "supermarket", "superstore", "outlet",
        "bank", "finance", "atm", "credit",
        "pharmacy", "medical", "clinic", "hospital",  
        "salon", "spa", "beauty", "fitness", "gym",
        "petrol", "fuel", "gas station", "service",
        "mobile", "telecom", "electronics",
        "fashion", "clothing", "boutique",
        # Japanese
        "ホテル", "レストラン", "銀行", "カフェ", "スーパー", "ショップ",
        "マート", "モール", "コンビニ", "薬局", "百貨店", "デパート",
        # Hindi (Devanagari)
        "होटल", "रेस्टोरेंट", "दुकान", "बैंक", "बाजार", "मार्ट",
        "शॉप", "मेडिकल", "फार्मेसी",
    ],
    "institutional": [
        # English
        "hospital", "clinic", "health centre", "medical centre", "dispensary",
        "school", "college", "university", "institute", "academy", "education",
        "church", "temple", "mosque", "masjid", "gurudwara", "mandir", "shrine",
        "police", "fire station", "court", "jail", "prison",
        "government", "govt", "municipal", "corporation", "ministry",
        "post office", "embassy", "consulate",
        "library", "museum", "gallery", "auditorium", "stadium",
        "charity", "ngo", "trust", "foundation",
        # Japanese
        "病院", "学校", "大学", "警察", "消防", "役所", "図書館",
        "寺", "神社", "教会", "市役所", "区役所", "裁判所",
        # Hindi
        "अस्पताल", "स्कूल", "कॉलेज", "विश्वविद्यालय", "पुलिस",
        "सरकार", "मंदिर", "मस्जिद", "चर्च", "न्यायालय", "पुस्तकालय",
    ],
    "industrial": [
        # English
        "factory", "warehouse", "manufacturing", "plant", "mill",
        "industry", "industries", "industrial", "ltd", "pvt ltd", "pvt. ltd",
        "limited", "co. ltd", "corp", "corporation",
        "logistics", "cargo", "freight", "transport", "depot",
        "chemical", "steel", "cement", "textile", "printing",
        "workshop", "garage", "auto", "service centre",
        # Japanese
        "工場", "倉庫", "製造", "産業", "会社", "株式会社",
        "物流", "輸送", "化学", "鉄鋼", "製鉄",
        # Hindi
        "कारखाना", "उद्योग", "कंपनी", "फैक्ट्री", "गोदाम",
    ],
}

# ── CLAHE preprocessing ───────────────────────────────────────────────────────

def preprocess_crop(crop_bgr: np.ndarray, min_h: int = 64,
                    upscale: float = 1.5) -> np.ndarray:
    """Enhance contrast + upscale for better OCR on small signage text.

    A 1.5x upscale substantially improves PaddleOCR recall on CJK and
    Devanagari scripts where strokes are fine-grained at street-view
    resolution.
    """
    h, w = crop_bgr.shape[:2]

    # Always upscale by the given factor (1.5x default); also enforce min_h
    effective_scale = max(upscale, min_h / h if h < min_h else upscale)
    if effective_scale > 1.0:
        crop_bgr = cv2.resize(
            crop_bgr,
            (int(w * effective_scale), int(h * effective_scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # CLAHE on L channel (LAB space)
    lab  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Keyword matching ──────────────────────────────────────────────────────────

def match_text_to_class(text: str) -> tuple[str | None, float]:
    """
    Match OCR text against keyword dictionary.
    Returns (class_name, confidence) or (None, 0.0).
    """
    if not text or not text.strip():
        return None, 0.0

    text_lower = text.lower().strip()
    tokens = re.split(r"[\s,|/\-\.]+", text_lower)

    class_scores: dict[str, float] = {}

    for cls, keywords in OCR_KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            kw_l = kw.lower()
            if kw_l in text_lower:
                # Weight by keyword length (longer = more specific)
                weight = min(1.0, len(kw_l) / 8.0 + 0.3)
                score  = max(score, weight)
        if score > 0:
            class_scores[cls] = score

    if not class_scores:
        return None, 0.0

    best_cls  = max(class_scores, key=class_scores.get)
    best_conf = class_scores[best_cls]
    return best_cls, round(min(best_conf, 0.95), 4)


# ── OCR engine ────────────────────────────────────────────────────────────────

class OCREngine:
    """PaddleOCR with English + Hindi + Japanese, EasyOCR fallback."""

    def __init__(self):
        self.paddle_ocr = None
        self.easy_ocr   = None
        self._init_paddle()

    def _init_paddle(self):
        try:
            from paddleocr import PaddleOCR

            def _make(lang):
                try:
                    # PaddleOCR >= 3.x (show_log / use_angle_cls removed)
                    return PaddleOCR(use_textline_orientation=True, lang=lang)
                except (TypeError, ValueError):
                    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

            # Multi-language: run English + Hindi + Japanese separately
            # (PaddleOCR supports one language per instance)
            self.paddle_ocr = {"en": _make("en"), "hi": _make("hi"), "ja": _make("japan")}
            print("[OCR] PaddleOCR initialized (en + hi + ja)")
        except Exception as e:
            print(f"[OCR] PaddleOCR unavailable ({e}), trying EasyOCR...")
            self._init_easyocr()

    def _init_easyocr(self):
        try:
            import easyocr
            # Japanese can't mix with Hindi in EasyOCR; use two readers
            self.easy_ocr = [
                easyocr.Reader(["en", "hi"], gpu=True, verbose=False),
                easyocr.Reader(["en", "ja"], gpu=True, verbose=False),
            ]
            print("[OCR] EasyOCR initialized (en+hi reader, en+ja reader)")
        except Exception as e:
            print(f"[OCR] EasyOCR also unavailable ({e}). Text extraction disabled.")

    def _run_paddle(self, img: np.ndarray) -> tuple[list[str], list[float]]:
        """Run PaddleOCR on a single image, return (texts, confs)."""
        texts, confs = [], []
        if not self.paddle_ocr:
            return texts, confs
        for lang, ocr in self.paddle_ocr.items():
            try:
                try:
                    results = ocr.predict(img)        # PaddleOCR 3.x
                except AttributeError:
                    results = ocr.ocr(img, cls=True)  # PaddleOCR 2.x
                if not results or results[0] is None:
                    continue
                r0 = results[0]
                if hasattr(r0, "get") and r0.get("rec_texts") is not None:
                    pairs = zip(r0.get("rec_texts", []), r0.get("rec_scores", []) or [])
                else:
                    pairs = ((line[1][0], line[1][1]) for line in r0 if line is not None)
                for text, conf in pairs:
                    if text and float(conf) > 0.3:
                        texts.append(str(text).strip())
                        confs.append(float(conf))
            except Exception:
                pass
        return texts, confs

    def _run_easyocr(self, img: np.ndarray) -> tuple[list[str], list[float]]:
        """Run EasyOCR on a single image, return (texts, confs)."""
        texts, confs = [], []
        if not self.easy_ocr:
            return texts, confs
        readers = self.easy_ocr if isinstance(self.easy_ocr, list) else [self.easy_ocr]
        seen: set[str] = set()
        for reader in readers:
            try:
                results = reader.readtext(img, detail=1)
                for _, text, conf in results:
                    text = text.strip()
                    if text and conf > 0.3 and text not in seen:
                        texts.append(text)
                        confs.append(float(conf))
                        seen.add(text)
            except Exception:
                pass
        return texts, confs

    def _ocr_one(self, img: np.ndarray) -> tuple[str, float]:
        """Run OCR (Paddle then EasyOCR fallback) on a single image."""
        texts, confs = self._run_paddle(img)
        if not texts:
            texts, confs = self._run_easyocr(img)
        if texts:
            return " ".join(texts), float(np.mean(confs))
        return "", 0.0

    def extract(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Run OCR on a building crop.
        Returns (concatenated_text, avg_confidence).

        Panoramic gnomonic projections at certain yaw angles (e.g. 157.5°,
        180°) produce horizontally mirrored text.  We run OCR on both the
        original and a horizontal flip, keeping whichever scores higher.
        """
        crop_bgr = preprocess_crop(crop_bgr)

        # ── Try original orientation ─────────────────────────────────────────
        text_orig, conf_orig = self._ocr_one(crop_bgr)

        # ── Try horizontal flip (mirrored text from rear-facing views) ───────
        crop_flip = cv2.flip(crop_bgr, 1)  # 1 = horizontal flip
        text_flip, conf_flip = self._ocr_one(crop_flip)

        # ── Keep the better result ───────────────────────────────────────────
        if conf_flip > conf_orig:
            return text_flip, conf_flip
        return text_orig, conf_orig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",  default="data/annotations/final_labels.csv",
                        help="CSV with image_path, x1,y1,x2,y2 columns")
    parser.add_argument("--output",  default="data/ocr_outputs/ocr_results.csv")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip rows already in output CSV")
    parser.add_argument("--min_conf", type=float, default=0.30,
                        help="Minimum OCR confidence to record text")
    args = parser.parse_args()

    print("Phase 4a — OCR Text Extraction")
    print("=" * 60)

    # ── Load labels ───────────────────────────────────────────────────────────
    label_path = Path(args.labels)
    if not label_path.exists():
        # Fallback: load all curriculum labels
        dfs = []
        for p in Path("data/curriculum").glob("**/labels.csv"):
            df = pd.read_csv(p)
            for old, new in [("source_image", "image_path"), ("primary_usage", "final_label")]:
                if old in df.columns and new not in df.columns:
                    df = df.rename(columns={old: new})
            dfs.append(df)
        if not dfs:
            print(f"[ERROR] No labels found at {args.labels} or data/curriculum")
            return
        df = pd.concat(dfs, ignore_index=True)
        print(f"[INFO] Using curriculum labels ({len(df)} rows)")
    else:
        df = pd.read_csv(label_path)
        for old, new in [("source_image", "image_path"), ("primary_usage", "final_label")]:
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    print(f"Loaded {len(df)} building annotations\n")

    # ── Resume ────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[str] = set()
    if args.resume and out_path.exists():
        done_df = pd.read_csv(out_path)
        for _, r in done_df.iterrows():
            done_keys.add(f"{r['image_path']}_{r.get('x1', 0):.0f}_{r.get('y1', 0):.0f}")
        print(f"Resuming: {len(done_keys)} crops already processed")

    # ── Initialize OCR ───────────────────────────────────────────────────────
    engine = OCREngine()

    # ── Process each building ─────────────────────────────────────────────────
    results = []
    skipped = 0
    n_text  = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="OCR extraction"):
        img_path = str(row["image_path"])
        x1 = int(float(row.get("x1", 0)))
        y1 = int(float(row.get("y1", 0)))
        x2 = int(float(row.get("x2", 0)))
        y2 = int(float(row.get("y2", 0)))

        key = f"{img_path}_{x1}_{y1}"
        if key in done_keys:
            continue

        # Load and crop image
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        H, W = img.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        if x2c <= x1c or y2c <= y1c:
            skipped += 1
            continue

        crop = img[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            skipped += 1
            continue

        # Run OCR
        ocr_text, ocr_conf = engine.extract(crop)

        has_text = ocr_text != "" and ocr_conf >= args.min_conf
        text_cls, text_cls_conf = (None, 0.0)
        if has_text:
            text_cls, text_cls_conf = match_text_to_class(ocr_text)
            n_text += 1

        results.append({
            "image_path":      img_path,
            "building_id":     row.get("building_id", idx),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "depth_m":         float(row.get("mean_depth_m", 0) or 0),
            "final_label":     str(row.get("final_label", "")),
            "ocr_text":        ocr_text,
            "ocr_raw_conf":    round(ocr_conf, 4),
            "has_text":        has_text,
            "text_class_prior": text_cls or "",
            "text_class_conf":  text_cls_conf,
        })

        # Periodic save every 500 rows
        if len(results) % 500 == 0:
            _append_csv(results, out_path, write_header=not out_path.exists())
            results = []

    # Final save
    if results:
        _append_csv(results, out_path, write_header=not out_path.exists())

    # ── Report ────────────────────────────────────────────────────────────────
    final_df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    total = len(final_df)

    print(f"\n{'='*60}")
    print("OCR EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total processed: {total}")
    print(f"  With text:       {(final_df['has_text'].sum() if 'has_text' in final_df else n_text)}")
    print(f"  Skipped (bad img): {skipped}")

    if "text_class_prior" in final_df.columns:
        print("\n  Text class distribution:")
        for cls, n in final_df[final_df["text_class_prior"] != ""]["text_class_prior"].value_counts().items():
            print(f"    {cls:15s}: {n}")

    print(f"\nSaved: {out_path}")
    print("Next: python src/ocr/precompute_text_embeds.py")


def _append_csv(rows: list[dict], path: Path, write_header: bool):
    df = pd.DataFrame(rows)
    df.to_csv(str(path), mode="a", header=write_header, index=False)


if __name__ == "__main__":
    main()
