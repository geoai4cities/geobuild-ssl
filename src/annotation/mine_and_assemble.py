"""

Usage:
    python src/annotation/mine_and_assemble.py \
        --checked   data/annotations/prototype_checked_labels.csv \
        --original  data/curriculum/ \
        --output    data/annotations/final_labels.csv \
        --target    1500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES  = ["residential", "commercial", "industrial", "institutional"]


# ── Mining helpers ────────────────────────────────────────────────────────────

def score_industrial(row: pd.Series) -> float:
    """Score a building row for likelihood of being industrial (0-1)."""
    score = 0.0
    ar    = float(row.get("aspect_ratio", 1.0))
    area  = float(row.get("mask_area_pixels", 0))
    depth = row.get("mean_depth_m", None)

    # Wide and low → industrial
    if ar > 2.0:   score += 0.35
    elif ar > 1.5: score += 0.20
    elif ar > 1.2: score += 0.10

    # Large footprint → industrial
    if area > 100000:  score += 0.35
    elif area > 60000: score += 0.25
    elif area > 40000: score += 0.15

    # Low CLIP residential probability
    clip_res = float(row.get("clip_prob_residential", 0.5))
    if clip_res < 0.25: score += 0.15
    elif clip_res < 0.35: score += 0.08

    # Reasonable depth
    if depth is not None and 10 < float(depth) < 30:
        score += 0.05

    # No text → slightly more industrial (factories rarely have English signs)
    if not bool(row.get("has_text", False)):
        score += 0.05

    # CLIP industrial score if available
    clip_ind = float(row.get("clip_prob_industrial", 0.0))
    score += 0.30 * clip_ind

    return min(1.0, score)


def score_institutional(row: pd.Series) -> float:
    """Score a building row for likelihood of being institutional (0-1)."""
    score = 0.0
    area  = float(row.get("mask_area_pixels", 0))

    # OCR keyword match (strongest signal)
    ocr_cls = row.get("ocr_class", None)
    if ocr_cls == "institutional":
        score += 0.60
    elif ocr_cls == "commercial":
        score -= 0.10  # mild negative signal

    # CLIP institutional probability
    clip_inst = float(row.get("clip_prob_institutional", 0.0))
    score += 0.35 * clip_inst

    # Large mask area (schools, hospitals tend to be large)
    if area > 80000:   score += 0.20
    elif area > 40000: score += 0.10

    # Not strongly residential
    clip_res = float(row.get("clip_prob_residential", 0.5))
    if clip_res < 0.30: score += 0.10

    # Has text (institutions often have signs)
    if bool(row.get("has_text", False)):
        score += 0.05

    return min(1.0, score)


def mine_rare_class(df_full: pd.DataFrame, cls: str,
                    current_count: int, target: int) -> pd.DataFrame:
    """Mine additional instances of a rare class from the full pool."""
    n_needed = max(0, target - current_count)
    if n_needed == 0:
        return pd.DataFrame()

    print(f"\n  Mining {n_needed} additional '{cls}' instances...")

    # Exclude already quality-labeled instances
    # Score each row
    if cls == "industrial":
        df_full = df_full.copy()
        df_full["_mine_score"] = df_full.apply(score_industrial, axis=1)
    elif cls == "institutional":
        df_full = df_full.copy()
        df_full["_mine_score"] = df_full.apply(score_institutional, axis=1)
    else:
        return pd.DataFrame()

    # Sort by mining score, take top candidates
    candidates = df_full[df_full["_mine_score"] > 0.45].sort_values(
        "_mine_score", ascending=False
    ).head(n_needed * 3)   # over-sample, then filter

    if len(candidates) == 0:
        print(f"    [WARN] No candidates found for '{cls}'")
        return pd.DataFrame()

    # Build new rows
    rows = []
    for _, row in candidates.head(n_needed).iterrows():
        new_row = row.copy()
        new_row["final_label"]  = cls
        new_row["final_conf"]   = round(min(0.75, float(row["_mine_score"])), 4)
        new_row["label_source"] = f"mined_{cls}"
        new_row["label_tier"]   = "C"
        new_row["knn_flagged"]  = False
        rows.append(new_row)

    mined = pd.DataFrame(rows).drop(columns=["_mine_score"], errors="ignore")
    print(f"    Mined {len(mined)} candidates (score > 0.45)")
    return mined


# ── Assembly ──────────────────────────────────────────────────────────────────

def apply_confidence_filter(df: pd.DataFrame, majority_thresh: float,
                             rare_thresh: float, rare_classes: list) -> pd.DataFrame:
    """Keep Tier A/B/C samples that pass confidence thresholds."""
    # Remove Tier D
    df = df[df["label_tier"].isin(["A", "B", "C"])].copy()

    # Apply class-specific confidence thresholds
    keep_mask = pd.Series(False, index=df.index)
    for cls in CLASSES:
        thresh = rare_thresh if cls in rare_classes else majority_thresh
        cls_mask = (df["final_label"] == cls) & (df["final_conf"] >= thresh)
        keep_mask |= cls_mask

    df = df[keep_mask].copy()
    return df


def balance_dataset(df: pd.DataFrame, target_per_class: int,
                    seed: int = 42) -> pd.DataFrame:
    """Subsample majority classes and keep all rare class samples."""
    rng    = np.random.default_rng(seed)
    parts  = []
    for cls in CLASSES:
        cls_df = df[df["final_label"] == cls]
        if len(cls_df) > target_per_class:
            # Prioritise higher-confidence samples
            cls_df = cls_df.sort_values("final_conf", ascending=False)
            cls_df = cls_df.iloc[:target_per_class]
        parts.append(cls_df)

    return pd.concat(parts, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_original_pool(root: Path) -> pd.DataFrame:
    """Load all labels from original curriculum dataset (105K pool)."""
    dfs = []
    for split in ["train"]:      # only train for mining
        for diff in ["easy", "medium", "hard"]:
            csv = root / split / diff / "labels.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                df["split"]      = split
                df["difficulty"] = diff
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checked",        default="data/annotations/prototype_checked_labels.csv")
    parser.add_argument("--original",       default="data/curriculum/")
    parser.add_argument("--output",         default="data/annotations/final_labels.csv")
    parser.add_argument("--target",         type=int, default=1500,
                        help="Target instances per class")
    parser.add_argument("--majority_thresh", type=float, default=0.50)
    parser.add_argument("--rare_thresh",     type=float, default=0.40)
    args = parser.parse_args()

    print("Phase 2 — Stage 2.5+2.6: Rare Class Mining + Final Assembly")
    print("=" * 60)

    # Load quality-checked labels
    checked_path = Path(args.checked)
    if not checked_path.exists():
        print(f"[ERROR] {checked_path} not found. Run prototype_consistency.py first.")
        return

    df_checked = pd.read_csv(checked_path)
    print(f"Loaded quality-checked labels: {len(df_checked)}")

    # ── Stage 2.5: Confidence filtering ──────────────────────────────────────
    print("\nStage 2.5a: Applying confidence filter...")
    rare_classes = ["industrial", "institutional"]
    df_filtered  = apply_confidence_filter(df_checked, args.majority_thresh,
                                            args.rare_thresh, rare_classes)
    print(f"After filter: {len(df_filtered)} (was {len(df_checked)})")
    for cls in CLASSES:
        n = (df_filtered["final_label"] == cls).sum()
        print(f"  {cls:15s}: {n}")

    # ── Stage 2.5b: Mine rare classes from original pool ─────────────────────
    print("\nStage 2.5b: Mining rare classes from original 105K pool...")
    orig_path = Path(args.original)
    if orig_path.exists():
        df_orig = load_original_pool(orig_path)
        print(f"  Loaded original pool: {len(df_orig)} rows")

        # Rename columns to match our schema where possible
        col_map = {"final_usage": "final_label", "final_confidence": "final_conf",
                   "source_image": "image_path"}
        for old, new in col_map.items():
            if old in df_orig.columns and new not in df_orig.columns:
                df_orig = df_orig.rename(columns={old: new})

        mined_all = []
        for cls in rare_classes:
            current_count = (df_filtered["final_label"] == cls).sum()
            if current_count >= args.target:
                print(f"  {cls}: already has {current_count} >= {args.target} → skip mining")
                continue
            mined = mine_rare_class(df_orig, cls, current_count, args.target)
            if not mined.empty:
                mined_all.append(mined)

        if mined_all:
            df_mined   = pd.concat(mined_all, ignore_index=True)
            df_filtered = pd.concat([df_filtered, df_mined], ignore_index=True)
            print(f"\n  After mining: {len(df_filtered)} total")
    else:
        print(f"  [WARN] Original pool not found at {orig_path}")

    # ── Stage 2.6: Balance to target_per_class ────────────────────────────────
    print(f"\nStage 2.6: Balancing to {args.target} instances per class...")
    df_final = balance_dataset(df_filtered, args.target)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("FINAL LABEL ASSEMBLY REPORT")
    print(f"{'='*60}")
    print(f"\nTotal instances: {len(df_final)}")
    for cls in CLASSES:
        n     = (df_final["final_label"] == cls).sum()
        arr   = df_final[df_final["final_label"] == cls]["final_conf"].values
        med   = np.median(arr) if len(arr) > 0 else 0
        print(f"  {cls:15s}: {n:5d}   median_conf={med:.3f}")

    print("\nTier distribution:")
    for tier, n in df_final["label_tier"].value_counts().sort_index().items():
        print(f"  Tier {tier}: {n}")

    print("\nLabel source distribution:")
    for src, n in df_final["label_source"].value_counts().items():
        print(f"  {src:30s}: {n}")

    # Confidence improvement
    print(f"\nConfidence stats:")
    arr = df_final["final_conf"].values
    print(f"  min={arr.min():.3f}  median={np.median(arr):.3f}  "
          f"mean={arr.mean():.3f}  max={arr.max():.3f}")

    print(f"\nSaved: {out_path}")

    # Split into train/val/test maintaining panorama disjointness
    val_test_note = ("NOTE: val/test splits preserved from original curriculum splits.\n"
                     "Final labels are for TRAIN only. Val/test use original labels.csv")
    print(f"\n{val_test_note}")
    print("\nDone. Final labels written to data/annotations/final_labels.csv")


if __name__ == "__main__":
    main()
