"""


Combines CLIP ensemble + OCR + geometric scores into a single high-quality label
using a priority fusion rule:

Usage:
    python src/annotation/fuse_labels.py \
        --input  data/annotations/multi_source_labels.csv \
        --output data/annotations/fused_labels.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["residential", "commercial", "industrial", "institutional"]

# Fusion thresholds (match annotation.yaml)
OCR_OVERRIDE_THRESH    = 0.70
CLIP_ENSEMBLE_THRESH   = 0.55
GEOM_FALLBACK_THRESH   = 0.65
DISCARD_BELOW          = 0.35


def fuse_row(row: pd.Series) -> dict:
    """Apply priority fusion to a single building row."""

    # ── Read sources ──────────────────────────────────────────────────────────
    clip_pred  = row.get("clip_pred", None)
    clip_conf  = float(row.get("clip_conf", 0.0))
    clip_agree = bool(row.get("clip_agree", False))

    ocr_class  = row.get("ocr_class", None)
    ocr_conf   = float(row.get("ocr_class_confidence", 0.0))
    has_text   = bool(row.get("has_text", False))

    geom_scores = {c: float(row.get(f"geom_score_{c}", 0.25)) for c in CLASSES}
    geom_pred   = max(geom_scores, key=geom_scores.get)
    geom_conf   = geom_scores[geom_pred]

    clip_probs  = {c: float(row.get(f"clip_prob_{c}", 0.25)) for c in CLASSES}

    # ── Priority 1: OCR override ──────────────────────────────────────────────
    if ocr_class is not None and ocr_conf >= OCR_OVERRIDE_THRESH:
        # Boost commercial/institutional detected by OCR
        final_label  = ocr_class
        final_conf   = round(min(0.92, ocr_conf + 0.05), 4)  # slight boost for text certainty
        label_source = "ocr"
        label_tier   = "A"

    # ── Priority 2: CLIP ensemble agreement ───────────────────────────────────
    elif clip_agree and clip_conf >= CLIP_ENSEMBLE_THRESH:
        # Mild geometric boost if geometry agrees with CLIP
        geom_boost = 0.05 if (geom_pred == clip_pred and geom_conf > 0.50) else 0.0
        # Mild OCR boost if OCR has any text signal matching
        ocr_boost  = 0.03 if (has_text and ocr_class == clip_pred) else 0.0
        final_conf   = round(min(0.90, clip_conf + geom_boost + ocr_boost), 4)
        final_label  = clip_pred
        label_source = "clip_ensemble"
        label_tier   = "B"

    # ── Priority 3: Geometric tiebreaker ─────────────────────────────────────
    elif geom_conf >= GEOM_FALLBACK_THRESH:
        final_label  = geom_pred
        final_conf   = round(min(0.65, geom_conf * 0.80), 4)   # penalise — geometry alone is weak
        label_source = "geometric"
        label_tier   = "C"

    # ── Fallback: best CLIP with penalty ─────────────────────────────────────
    else:
        final_label  = max(clip_probs, key=clip_probs.get)
        final_conf   = round(clip_probs[final_label] * 0.55, 4)  # heavy penalty for uncertain
        label_source = "uncertain"
        label_tier   = "D"

    return {
        "final_label":   final_label,
        "final_conf":    final_conf,
        "label_source":  label_source,
        "label_tier":    label_tier,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/annotations/multi_source_labels.csv")
    parser.add_argument("--output", default="data/annotations/fused_labels.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Apply fusion
    fused = df.apply(fuse_row, axis=1, result_type="expand")
    df    = pd.concat([df, fused], axis=1)

    # Save full fused labels
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FUSION REPORT")
    print("=" * 60)

    # Tier distribution
    tier_counts = df["label_tier"].value_counts().sort_index()
    total = len(df)
    print("\nLabel Tier Distribution:")
    tier_descs = {"A": "Gold (OCR override)", "B": "Silver (CLIP ensemble)",
                  "C": "Bronze (Geometric)", "D": "Noisy (uncertain)"}
    for tier, n in tier_counts.items():
        print(f"  Tier {tier} [{tier_descs.get(tier, '?')}]: {n:6d}  ({100*n/total:.1f}%)")

    # Source distribution
    print("\nLabel Source Distribution:")
    for src, n in df["label_source"].value_counts().items():
        print(f"  {src:20s}: {n:6d}  ({100*n/total:.1f}%)")

    # Class distribution (all tiers)
    print("\nClass Distribution (All Tiers):")
    for cls in CLASSES:
        n   = (df["final_label"] == cls).sum()
        pct = 100 * n / total
        bar = "█" * int(pct / 2)
        print(f"  {cls:15s}: {n:6d}  ({pct:5.1f}%)  {bar}")

    # After filtering Tier D
    df_keep = df[df["label_tier"].isin(["A", "B", "C"])]
    df_discard = df[df["label_tier"] == "D"]
    print(f"\nAfter discarding Tier D:")
    print(f"  Keep:    {len(df_keep)} ({100*len(df_keep)/total:.1f}%)")
    print(f"  Discard: {len(df_discard)} ({100*len(df_discard)/total:.1f}%)")
    print("\nClass Distribution (Tier A+B+C only):")
    for cls in CLASSES:
        n   = (df_keep["final_label"] == cls).sum()
        pct = 100 * n / len(df_keep) if len(df_keep) > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {cls:15s}: {n:6d}  ({pct:5.1f}%)  {bar}")

    # Confidence stats
    print(f"\nConfidence stats (all):")
    arr = df["final_conf"].values
    print(f"  min={arr.min():.3f}  median={np.median(arr):.3f}  "
          f"mean={arr.mean():.3f}  max={arr.max():.3f}")
    print(f"  Before quality filter: {len(df)}")
    print(f"  After  quality filter (>=0.40): "
          f"{(df['final_conf'] >= 0.40).sum()}")
    print(f"  After  quality filter (>=0.50): "
          f"{(df['final_conf'] >= 0.50).sum()}")

    # Save Tier D IDs for inspection
    discard_path = out_path.parent / "discarded_uncertain.csv"
    df_discard[["building_id", "image_path", "clip_pred", "clip_conf",
                "ocr_class", "final_conf"]].to_csv(discard_path, index=False)
    print(f"\nDiscarded uncertain labels → {discard_path}")
    print("Run DINOv3 prototype check next: python src/annotation/prototype_consistency.py")


if __name__ == "__main__":
    main()
