"""
Usage:
    python src/annotation/prototype_consistency.py \
        --labels  data/annotations/fused_labels.csv \
        --images  data/curriculum/ \
        --weights final_models/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --output  data/annotations/prototype_checked_labels.csv \
        --k 10
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

CLASSES = ["residential", "commercial", "industrial", "institutional"]
IMG_SIZE = 224
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ── DINOv3 feature extractor ─────────────────────────────────────────────────

class DINOv3FeatureExtractor:
    """Loads DINOv3 ViT-L/16 frozen and extracts CLS token features."""

    def __init__(self, weights_path: str | None = None):
        print(f"Loading DINOv3 on {DEVICE}...")
        # Try loading from local weights first, then from torch hub
        try:
            if weights_path and Path(weights_path).exists():
                # Load via timm with local weights
                import timm
                self.model = timm.create_model("vit_large_patch16_224.dino",
                                               pretrained=False, num_classes=0)
                state = torch.load(weights_path, map_location="cpu")
                # Handle various checkpoint formats
                if "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]
                # Strip 'backbone.' prefix if present
                state = {k.replace("backbone.", ""): v for k, v in state.items()}
                self.model.load_state_dict(state, strict=False)
                print(f"  Loaded from local: {weights_path}")
            else:
                # Fall back to torch hub DINOv2 (compatible features)
                self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                                            pretrained=True)
                print("  Loaded DINOv2 ViT-L from torch hub (compatible features)")
        except Exception as e:
            print(f"  [WARN] DINOv3 load failed: {e}")
            print("  Falling back to torchvision ViT-L/16")
            import torchvision.models as tvm
            self.model = tvm.vit_l_16(weights="IMAGENET1K_V1")
            self.model.heads = torch.nn.Identity()

        self.model = self.model.to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Standard ImageNet normalisation
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> torch.Tensor | None:
        """Returns L2-normalised feature vector [D]."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            t    = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
            t    = ((t.unsqueeze(0).to(DEVICE) - self.mean) / self.std)
            feat = self.model(t)
            if feat.ndim > 2:   # handle (B, tokens, D) from ViT
                feat = feat[:, 0]  # CLS token
            feat = F.normalize(feat, dim=-1)
            return feat.squeeze(0).cpu()
        except Exception:
            return None

    @torch.no_grad()
    def extract_batch(self, crops: list[np.ndarray], batch_size: int = 32) -> list:
        """Extract features for a list of crops in batches."""
        feats = []
        for i in range(0, len(crops), batch_size):
            batch_bgr = crops[i:i + batch_size]
            tensors   = []
            valid_idx = []
            for j, crop in enumerate(batch_bgr):
                if crop is None or crop.size == 0:
                    continue
                try:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                    t   = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
                    tensors.append(t)
                    valid_idx.append(i + j)
                except Exception:
                    pass
            if not tensors:
                feats.extend([None] * len(batch_bgr))
                continue
            batch_t = torch.stack(tensors).to(DEVICE)
            batch_t = (batch_t - self.mean) / self.std
            out     = self.model(batch_t)
            if out.ndim > 2:
                out = out[:, 0]
            out = F.normalize(out, dim=-1).cpu()
            feat_map = {vi: out[k] for k, vi in enumerate(valid_idx)}
            feats.extend([feat_map.get(i + j) for j in range(len(batch_bgr))])

        return feats


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_crop(image_path: str, x1: int, y1: int, x2: int, y2: int,
              pad: int = 4) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        return None
    H, W  = img.shape[:2]
    x1p   = max(0, x1 - pad)
    y1p   = max(0, y1 - pad)
    x2p   = min(W, x2 + pad)
    y2p   = min(H, y2 + pad)
    crop  = img[y1p:y2p, x1p:x2p]
    return crop if crop.size > 0 else None


def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - torch.dot(a, b).clamp(-1, 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",  default="data/annotations/fused_labels.csv")
    parser.add_argument("--output",  default="data/annotations/prototype_checked_labels.csv")
    parser.add_argument("--weights", default="final_models/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--k",       type=int, default=10, help="KNN neighbours")
    parser.add_argument("--relabel_ratio", type=float, default=0.85,
                        help="Relabel if dist(nearest_other)/dist(own_class) < this")
    parser.add_argument("--knn_min_agree", type=float, default=0.40,
                        help="Flag if fewer than this fraction of KNN agree")
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    print(f"Loaded {len(df)} labels")

    extractor = DINOv3FeatureExtractor(args.weights)

    # ── Step 1: Extract features for ALL buildings ────────────────────────────
    print("\nStep 1: Extracting DINOv3 features...")
    features = []
    crops    = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading crops"):
        crop = load_crop(str(row["image_path"]),
                         int(row["x1"]), int(row["y1"]),
                         int(row["x2"]), int(row["y2"]))
        crops.append(crop)

    print("  Running batch feature extraction...")
    features = extractor.extract_batch(crops, batch_size=32)
    valid_mask = [f is not None for f in features]
    print(f"  Valid features: {sum(valid_mask)}/{len(features)}")

    df["_feat_idx"] = range(len(df))
    df["_feat_valid"] = valid_mask

    # ── Step 2: Compute prototypes from Tier A (Gold) labels only ────────────
    print("\nStep 2: Computing class prototypes from Tier A labels...")
    prototypes = {}
    for cls in CLASSES:
        tier_a_mask = (df["label_tier"] == "A") & (df["final_label"] == cls) & df["_feat_valid"]
        tier_a_idx  = df[tier_a_mask].index.tolist()
        cls_feats   = [features[i] for i in tier_a_idx if features[i] is not None]
        if cls_feats:
            proto = torch.stack(cls_feats).mean(0)
            proto = F.normalize(proto, dim=0)
            prototypes[cls] = proto
            print(f"  {cls:15s}: {len(cls_feats)} gold samples → prototype computed")
        else:
            # Fallback: use Tier B
            tier_b_mask = (df["label_tier"].isin(["A", "B"])) & (df["final_label"] == cls) & df["_feat_valid"]
            tier_b_idx  = df[tier_b_mask].index.tolist()[:200]
            cls_feats   = [features[i] for i in tier_b_idx if features[i] is not None]
            if cls_feats:
                proto = torch.stack(cls_feats).mean(0)
                proto = F.normalize(proto, dim=0)
                prototypes[cls] = proto
                print(f"  {cls:15s}: [WARN] No Tier A — used {len(cls_feats)} Tier B samples")
            else:
                print(f"  {cls:15s}: [WARN] No prototype could be computed!")

    # ── Step 3: Check prototype consistency for Tier B/C ─────────────────────
    print("\nStep 3: Prototype consistency check for Tier B/C labels...")
    n_relabeled    = 0
    n_flagged_knn  = 0

    new_labels  = df["final_label"].tolist()
    new_confs   = df["final_conf"].tolist()
    new_sources = df["label_source"].tolist()
    new_tiers   = df["label_tier"].tolist()
    knn_flags   = [False] * len(df)

    # Build feature matrix for KNN
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    feat_matrix   = torch.stack([features[i] for i in valid_indices])  # [N_valid, D]

    for row_i, row in tqdm(df.iterrows(), total=len(df), desc="Checking consistency"):
        if not valid_mask[row_i]:
            continue
        if new_tiers[row_i] not in ("B", "C"):
            continue    # only check non-gold labels

        feat    = features[row_i]
        cur_cls = new_labels[row_i]

        # Distance to each prototype
        if prototypes:
            dists = {}
            for cls in CLASSES:
                if cls in prototypes:
                    dists[cls] = cosine_dist(feat, prototypes[cls])

            own_dist   = dists.get(cur_cls, 1.0)
            other_dists = {c: d for c, d in dists.items() if c != cur_cls}
            if other_dists:
                nearest_other_cls  = min(other_dists, key=other_dists.get)
                nearest_other_dist = other_dists[nearest_other_cls]

                # Relabel if much closer to another prototype
                if nearest_other_dist < own_dist * args.relabel_ratio:
                    new_labels[row_i]  = nearest_other_cls
                    new_confs[row_i]   = round(new_confs[row_i] * 0.80, 4)  # reduce confidence
                    new_sources[row_i] = "prototype_corrected"
                    new_tiers[row_i]   = "C"  # down-tier after relabeling
                    n_relabeled += 1

        # KNN consistency check
        if len(valid_indices) > args.k:
            sims = torch.mv(feat_matrix, feat)
            top_k_idx = sims.topk(args.k + 1).indices[1:]  # exclude self
            knn_global = [valid_indices[i] for i in top_k_idx.tolist()]
            knn_labels = [new_labels[gi] for gi in knn_global]
            agree_frac = knn_labels.count(new_labels[row_i]) / len(knn_labels)
            if agree_frac < args.knn_min_agree:
                knn_flags[row_i] = True
                n_flagged_knn   += 1

    df["final_label"]  = new_labels
    df["final_conf"]   = new_confs
    df["label_source"] = new_sources
    df["label_tier"]   = new_tiers
    df["knn_flagged"]  = knn_flags

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop helper columns
    df = df.drop(columns=["_feat_idx", "_feat_valid"], errors="ignore")
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("PROTOTYPE CONSISTENCY REPORT")
    print(f"{'='*60}")
    print(f"  Labels relabeled by prototype: {n_relabeled}")
    print(f"  Labels flagged by KNN:         {n_flagged_knn}")
    print(f"\nFinal class distribution (after correction):")
    for cls in CLASSES:
        n   = (df["final_label"] == cls).sum()
        pct = 100 * n / len(df)
        print(f"  {cls:15s}: {n:6d}  ({pct:.1f}%)")
    print(f"\nTier distribution:")
    for tier, n in df["label_tier"].value_counts().sort_index().items():
        print(f"  Tier {tier}: {n:6d}")
    print(f"\nSaved: {out_path}")
    print("\nNext: python src/annotation/mine_and_assemble.py")


if __name__ == "__main__":
    main()
