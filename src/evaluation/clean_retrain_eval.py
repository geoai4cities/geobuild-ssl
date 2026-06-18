"""
Usage:
    python src/evaluation/clean_retrain_eval.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classification.train_v5_curriculum import (
    V5BuildingClassifier, BuildingCropDataset,
    CLASSES, CLASS2IDX, DEVICE, train_stage,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# The released CSVs are already a disjoint, panorama-level split, so the
# de-leak step below is a no-op; the derived *_clean.csv are written as
# intermediates and are not part of the released data.
ANNOT_DIR    = PROJECT_ROOT / "data/annotations"
TRAIN_CSV    = ANNOT_DIR / "clean_train_labels.csv"
VAL_CSV      = ANNOT_DIR / "clean_val_labels.csv"
TEST_CSV     = ANNOT_DIR / "clean_test_labels.csv"

CLEAN_TRAIN  = ANNOT_DIR / "train_labels_clean.csv"
CLEAN_VAL    = ANNOT_DIR / "val_labels_clean.csv"

OCR_CSV      = PROJECT_ROOT / "data/ocr_outputs/ocr_results.csv"
TEXT_CACHE   = PROJECT_ROOT / "data/ocr_outputs/clip_text_embeds.pkl"
DINOV3_CKPT  = PROJECT_ROOT / "final_models/ssl_model/dinov3_rgbd_ssl.pth"

OUT_DIR      = PROJECT_ROOT / "outputs/clean_retrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_DIR / "results.json"
LOG_PATH     = OUT_DIR / "progress.log"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── Step 1: clean CSVs ────────────────────────────────────────────────────────
def make_clean_csvs():
    log("Step 1: build clean train/val CSVs (exclude test panoramas)")
    test = pd.read_csv(TEST_CSV)
    test_panos = set(test["source_image"].unique())
    log(f"  Clean test: {len(test)} instances across {len(test_panos)} panoramas")

    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)

    ct = train[~train["source_image"].isin(test_panos)].reset_index(drop=True)
    cv = val[~val["source_image"].isin(test_panos)].reset_index(drop=True)
    ct.to_csv(CLEAN_TRAIN, index=False)
    cv.to_csv(CLEAN_VAL, index=False)

    log(f"  Clean train: {len(ct)} rows (was {len(train)})  by country: {dict(ct['country'].value_counts())}")
    log(f"  Clean val:   {len(cv)} rows (was {len(val)})    by country: {dict(cv['country'].value_counts())}")


# ── Step 2-4: training ────────────────────────────────────────────────────────
def country_csv(base_csv, country):
    df = pd.read_csv(base_csv)
    f  = df[df["country"] == country].reset_index(drop=True)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.to_csv(tmp.name, index=False); tmp.close()
    return tmp.name, len(f)


def train_curriculum(tag: str, train_csv: str, val_csv: str, out_subdir: Path) -> str:
    """Run 3-stage curriculum training and return the best stage-3 checkpoint."""
    log(f"\n=== Training '{tag}' ===")
    log(f"  train_csv: {train_csv}")
    log(f"  val_csv:   {val_csv}")
    out_subdir.mkdir(parents=True, exist_ok=True)

    model = V5BuildingClassifier()
    model.load_backbone(str(DINOV3_CKPT), freeze_blocks_until=22)
    model.float().to(DEVICE)

    teacher  = None
    last_path = None
    for s in [1, 2, 3]:
        log(f"  Stage {s} starting...")
        t0 = time.time()
        best_path, best_wf1 = train_stage(
            stage      = s,
            model      = model,
            teacher    = teacher,
            train_csv  = train_csv,
            val_csv    = val_csv,
            ocr_csv    = str(OCR_CSV) if OCR_CSV.exists() else None,
            text_cache = str(TEXT_CACHE) if TEXT_CACHE.exists() else None,
            out_dir    = out_subdir,
            batch_size = 32,
            no_depth   = False,
        )
        dt = (time.time() - t0) / 60
        log(f"  Stage {s} done in {dt:.1f}min, best wF1={best_wf1:.4f}, ckpt={best_path}")

        # rename so different tags don't overwrite each other
        renamed = out_subdir / f"{tag}_stage{s}_best.pth"
        shutil.copy(best_path, renamed)

        # next-stage teacher = current best
        from copy import deepcopy
        teacher = deepcopy(model); teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)

        state = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state, strict=False)
        last_path = str(renamed)

    final_path = out_subdir / f"{tag}_final.pth"
    torch.save(model.state_dict(), final_path)
    log(f"  Final '{tag}' saved: {final_path}")
    return last_path  # use stage-3 best


# ── Step 5: evaluation ────────────────────────────────────────────────────────
def make_loader(csv_path: str):
    ds = BuildingCropDataset(
        csv_path=csv_path,
        ocr_csv=str(OCR_CSV) if OCR_CSV.exists() else None,
        clip_text_cache=str(TEXT_CACHE) if TEXT_CACHE.exists() else None,
        augment=False,
    )
    return DataLoader(ds, batch_size=48, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate(ckpt_path: str, loader, label_for_log: str):
    model = V5BuildingClassifier()
    model.load_backbone(str(DINOV3_CKPT), freeze_blocks_until=0)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()

    ys, yps = [], []
    for rgb, depth, text_emb, label, _diff, _ctry in loader:
        rgb = rgb.to(DEVICE); depth = depth.to(DEVICE); text_emb = text_emb.to(DEVICE)
        out = model(rgb, depth, text_emb)
        ys.extend(label.tolist())
        yps.extend(out["logits"].argmax(dim=-1).cpu().tolist())

    wf1 = float(f1_score(ys, yps, average="weighted", zero_division=0))
    pc  = f1_score(ys, yps, average=None, zero_division=0, labels=list(range(len(CLASSES))))
    res = {"wF1": wf1, "n": len(ys),
           "per_class": {CLASSES[i]: float(pc[i]) for i in range(len(CLASSES))}}
    log(f"  {label_for_log}: wF1={wf1*100:.1f}%  per-class={res['per_class']}")
    return res


def main():
    log("=" * 60)
    log("CLEAN RETRAIN + IN-DOMAIN EVALUATION (Option B)")
    log("=" * 60)

    # 1. Build clean CSVs
    make_clean_csvs()

    # 2-4. Train three models
    mixed_ckpt = train_curriculum(
        tag="mixed",
        train_csv=str(CLEAN_TRAIN),
        val_csv=str(CLEAN_VAL),
        out_subdir=OUT_DIR / "mixed",
    )

    india_train, _ = country_csv(CLEAN_TRAIN, "india")
    india_val,   _ = country_csv(CLEAN_VAL, "india")
    india_ckpt = train_curriculum(
        tag="india_only",
        train_csv=india_train,
        val_csv=india_val,
        out_subdir=OUT_DIR / "india_only",
    )
    os.unlink(india_train); os.unlink(india_val)

    japan_train, _ = country_csv(CLEAN_TRAIN, "japan")
    japan_val,   _ = country_csv(CLEAN_VAL, "japan")
    japan_ckpt = train_curriculum(
        tag="japan_only",
        train_csv=japan_train,
        val_csv=japan_val,
        out_subdir=OUT_DIR / "japan_only",
    )
    os.unlink(japan_train); os.unlink(japan_val)

    # 5. Evaluate all three on the clean test set, per country
    log("\n" + "=" * 60)
    log("EVALUATION on the clean test set")
    log("=" * 60)

    india_test_csv, n_i = country_csv(TEST_CSV, "india")
    japan_test_csv, n_j = country_csv(TEST_CSV, "japan")
    log(f"  India test: {n_i}   Japan test: {n_j}")

    ind_loader = make_loader(india_test_csv)
    jap_loader = make_loader(japan_test_csv)

    results = {}
    for tag, ckpt in [("mixed", mixed_ckpt),
                      ("india_only", india_ckpt),
                      ("japan_only", japan_ckpt)]:
        log(f"\n[{tag}] checkpoint: {ckpt}")
        results[tag] = {
            "india": evaluate(ckpt, ind_loader, f"{tag} on India"),
            "japan": evaluate(ckpt, jap_loader, f"{tag} on Japan"),
            "checkpoint": ckpt,
        }

    os.unlink(india_test_csv); os.unlink(japan_test_csv)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nFinal results written to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
