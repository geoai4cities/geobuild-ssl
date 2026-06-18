"""

Usage:
    python src/classification/train_v5_curriculum.py
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

CLASSES    = ["residential", "commercial", "industrial", "institutional"]
CLASS2IDX  = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = 4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEPTH_THRESH_M = 30.0

# ── File logger (bypass conda buffering) ──────────────────────────────────────
import logging

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("v5_train")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

_logger = None

def log(msg: str):
    if _logger:
        _logger.info(msg)
    else:
        print(msg, flush=True)

# ── Image normalization ────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def normalize(img_chw: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
    s = torch.tensor(STD,  dtype=torch.float32)[:, None, None]
    return (img_chw - m) / s


def norm_depth(d: np.ndarray) -> np.ndarray:
    """Per-patch percentile normalization (2nd–98th pct → [0, 1]).

    This is the depth normalization the depth-map FiLM model was trained with —
    robust to the arbitrary scale of monocular relative depth, so the encoder
    sees the same input distribution at train and inference time.
    """
    d = d.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class BuildingCropDataset(Dataset):
    """
    Loads building crops from annotation CSV.
    Returns: rgb_tensor, depth_patch [1,224,224], text_emb, label_idx, difficulty
    """

    def __init__(self,
                 csv_path: str,
                 ocr_csv: str | None,
                 clip_text_cache: str | None,
                 image_size: int = 224,
                 augment: bool = True,
                 difficulty_filter: list[str] | None = None,
                 depth_dir: str | None = None):

        self.df = pd.read_csv(csv_path)
        if difficulty_filter:
            self.df = self.df[self.df["difficulty"].isin(difficulty_filter)]
        self.df = self.df.reset_index(drop=True)

        self.image_size = image_size
        self.augment    = augment
        # Directory of precomputed depth maps ({source_image_stem}_depth.npy),
        # as written by src/gis/save_depth_maps.py. When absent, the depth
        # patch falls back to a constant map from the CSV's mean_depth_m.
        self.depth_dir  = Path(depth_dir) if depth_dir else None

        # ── Merge OCR text (join on composite key to avoid cartesian product) ──
        if ocr_csv and Path(ocr_csv).exists():
            ocr_df = pd.read_csv(ocr_csv)
            # Rename image_path → source_image for alignment
            if "image_path" in ocr_df.columns and "source_image" not in ocr_df.columns:
                ocr_df = ocr_df.rename(columns={"image_path": "source_image"})
            # Build composite key (source_image + bbox) — truly unique per crop
            for df_ in [self.df, ocr_df]:
                df_["_crop_key"] = (df_["source_image"].astype(str) + "_" +
                                    df_["x1"].astype(int).astype(str) + "_" +
                                    df_["y1"].astype(int).astype(str) + "_" +
                                    df_["x2"].astype(int).astype(str) + "_" +
                                    df_["y2"].astype(int).astype(str))
            ocr_cols = ["_crop_key", "ocr_text", "has_text",
                        "text_class_prior", "text_class_conf"]
            ocr_cols = [c for c in ocr_cols if c in ocr_df.columns]
            # Deduplicate OCR (take first per crop key)
            ocr_dedup = ocr_df[ocr_cols].drop_duplicates("_crop_key")
            self.df = self.df.merge(ocr_dedup, on="_crop_key", how="left")
            self.df.drop(columns=["_crop_key"], inplace=True, errors="ignore")
            ocr_df.drop(columns=["_crop_key"], inplace=True, errors="ignore")
        else:
            self.df["ocr_text"]        = ""
            self.df["has_text"]        = False
            self.df["text_class_prior"]= ""
            self.df["text_class_conf"] = 0.0

        # ── Load precomputed CLIP text embeddings ──────────────────────────────
        self.text_embeds: dict[str, np.ndarray] = {}
        if clip_text_cache and Path(clip_text_cache).exists():
            cache = torch.load(clip_text_cache, map_location="cpu", weights_only=False)
            # Convert tensor values to numpy for fast lookup
            if isinstance(cache, dict):
                self.text_embeds = {
                    k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                    for k, v in cache.items()
                }
            else:
                self.text_embeds = {}
            print(f"  Loaded {len(self.text_embeds)} CLIP text embeddings")
        else:
            print("  No CLIP text cache — text modality = zeros")

    def __len__(self) -> int:
        return len(self.df)

    def _load_crop(self, row) -> np.ndarray | None:
        img_path = row["source_image"]
        if not Path(img_path).exists():
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1 = max(0, int(row["x1"]))
        y1 = max(0, int(row["y1"]))
        x2 = min(w, int(row["x2"]))
        y2 = min(h, int(row["y2"]))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        return cv2.resize(crop, (self.image_size, self.image_size))

    def _load_depth_patch(self, row) -> np.ndarray:
        """
        Return the 224×224 depth-map *patch* for this building, normalized to
        [0, 1]. Loads {stem}_depth.npy from depth_dir and crops to the bbox;
        falls back to a constant map from mean_depth_m when no map is found.
        """
        size = self.image_size
        patch = None
        if self.depth_dir is not None:
            stem  = Path(str(row["source_image"])).stem
            dpath = self.depth_dir / f"{stem}_depth.npy"
            if dpath.exists():
                dmap = np.load(dpath).astype(np.float32)
                h, w = dmap.shape[:2]
                x1 = max(0, int(row["x1"])); y1 = max(0, int(row["y1"]))
                x2 = min(w, int(row["x2"])); y2 = min(h, int(row["y2"]))
                if x2 > x1 and y2 > y1:
                    patch = dmap[y1:y2, x1:x2]
        if patch is None or patch.size == 0:
            md = float(row.get("mean_depth_m", 15.0))
            patch = np.full((size, size), md, dtype=np.float32)
        patch = cv2.resize(patch, (size, size))
        return norm_depth(patch)

    def _augment(self, crop: np.ndarray) -> np.ndarray:
        # Horizontal flip
        if np.random.rand() > 0.5:
            crop = cv2.flip(crop, 1)
        # Random brightness / contrast
        alpha = np.random.uniform(0.8, 1.2)
        beta  = np.random.randint(-15, 15)
        crop  = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        # Random crop (5% padding)
        pad = max(1, int(self.image_size * 0.05))
        x1  = np.random.randint(0, pad)
        y1  = np.random.randint(0, pad)
        x2  = self.image_size - np.random.randint(0, pad)
        y2  = self.image_size - np.random.randint(0, pad)
        crop = crop[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.image_size, self.image_size))
        return crop

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        label  = CLASS2IDX.get(str(row["final_usage"]), 0)

        # ── Depth-map patch (224×224, normalized) ──────────────────────────────
        depth_patch = torch.from_numpy(self._load_depth_patch(row)).unsqueeze(0)  # [1,224,224]

        # ── RGB crop ──────────────────────────────────────────────────────────
        crop = self._load_crop(row)
        if crop is None:
            # Return a blank image if file missing
            crop = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        if self.augment:
            crop = self._augment(crop)

        rgb_t = torch.from_numpy(crop[:, :, ::-1].copy()).float() / 255.0
        rgb_t = rgb_t.permute(2, 0, 1)   # (3, H, W)
        rgb_t = normalize(rgb_t)

        # ── CLIP text embedding ────────────────────────────────────────────────
        bid = str(row.get("building_id", ""))
        if bid in self.text_embeds:
            text_emb = torch.from_numpy(self.text_embeds[bid].astype(np.float32))
        else:
            text_emb = torch.zeros(512, dtype=torch.float32)

        difficulty = str(row.get("difficulty", "easy"))
        country    = str(row.get("country", "unknown"))
        return rgb_t, depth_patch.float(), text_emb, label, difficulty, country


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class DepthMapFiLMEncoder(nn.Module):
    """
    Paper Fig. 4 / Eq. 2: a four-block stride-2 depth CNN processes the
    224×224 depth-map *patch* into a 128-d conditioning vector, from which an
    MLP head produces FiLM parameters applied to the visual feature v:

        z          = DepthCNN(depth_map)          # [B, 128]
        (γ, β)     = MLP(z)                        # each [B, feat_dim]
        ṽ          = γ ⊙ v + β                     # feature-wise modulation

    Conditioning on the spatial structure of the depth patch (not a single
    scalar distance) lets the model suppress channels characteristic of
    distant buildings and amplify those of close ones.
    """

    def __init__(self, feat_dim: int = 1024):
        super().__init__()
        self.feat_dim = feat_dim
        # Four-block stride-2 CNN: 224 → 112 → 56 → 28 → 14 → global pool → 128
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  32, 3, 2, 1),  nn.BatchNorm2d(32),  nn.ReLU(inplace=True),  # 112
            nn.Conv2d(32, 64, 3, 2, 1),  nn.BatchNorm2d(64),  nn.ReLU(inplace=True),  # 56
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 28
            nn.Conv2d(128, 128, 3, 2, 1),nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 14
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                                    # 128
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim * 2),                   # γ concat β
        )
        # Init to identity (γ=1, β=0) so an untrained encoder is a no-op
        nn.init.zeros_(self.head[-1].weight)
        self.head[-1].bias.data[:feat_dim] = 1.0  # γ init to 1
        self.head[-1].bias.data[feat_dim:] = 0.0  # β init to 0

    def forward(self, depth_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        depth_map: [B, 1, 224, 224] float, normalized to [0, 1]
        returns:   γ [B, feat_dim], β [B, feat_dim]
        """
        if depth_map.dim() == 3:        # [B, H, W] → [B, 1, H, W]
            depth_map = depth_map.unsqueeze(1)
        out   = self.head(self.cnn(depth_map))    # [B, 2*feat_dim]
        gamma = out[:, :self.feat_dim]
        beta  = out[:, self.feat_dim:]
        return gamma, beta


class V5BuildingClassifier(nn.Module):
    """
    DINOv3 + FiLM(depth) + CLIP(text) → 4-class building usage.
    """

    def __init__(self, dinov3_dim: int = 1024, text_dim: int = 512,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()

        # ── DINOv3 backbone (loaded separately, stored as self.backbone) ───────
        self.backbone = None   # set via load_backbone()

        # ── FiLM depth encoder (depth-map CNN, paper Fig. 4 / Eq. 2) ───────────
        self.depth_film = DepthMapFiLMEncoder(feat_dim=dinov3_dim)

        # ── Text projection (CLIP 512 → 512, with LayerNorm) ──────────────────
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
        )

        # ── Fusion + classifier ───────────────────────────────────────────────
        fused_dim = dinov3_dim + text_dim   # 1024 + 512 = 1536
        self.fusion_norm = nn.LayerNorm(fused_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        # ── Projection head for SupCon loss ────────────────────────────────────
        self.proj_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def load_backbone(self, ckpt_path: str, freeze_blocks_until: int = 20):
        """Load DINOv3 ViT-L/16, freeze first N blocks."""
        try:
            import timm
            self.backbone = timm.create_model(
                "vit_large_patch16_224", pretrained=False,
                num_classes=0, global_pool="token"
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]
            missing, unexpected = self.backbone.load_state_dict(ckpt, strict=False)
            print(f"  DINOv3 loaded: missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"  [WARN] DINOv3 load failed ({e}), using random init")
            import timm
            self.backbone = timm.create_model(
                "vit_large_patch16_224", pretrained=False,
                num_classes=0, global_pool="token"
            )

        self.set_frozen_blocks(freeze_until=freeze_blocks_until)

    def set_frozen_blocks(self, freeze_until: int = 20):
        """Freeze ViT blocks 0..freeze_until-1, unfreeze the rest."""
        if self.backbone is None:
            return
        total_blocks = len(self.backbone.blocks)
        for i, block in enumerate(self.backbone.blocks):
            for p in block.parameters():
                p.requires_grad = (i >= freeze_until)
        # Always train patch_embed, norm, head
        for p in self.backbone.norm.parameters():
            p.requires_grad_(True)
        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        print(f"  DINOv3 blocks: {total_blocks} total, freeze_until={freeze_until} "
              f"| frozen={frozen} trainable={trainable} params")

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor,
                text_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        rgb:      [B, 3, 224, 224]
        depth:    [B, 1, 224, 224]  depth-map patch, normalized to [0, 1]
        text_emb: [B, 512]

        Returns dict with:
          'logits'   : [B, 4]
          'embedding': [B, 1536]  (fused, for SupCon)
          'proj'     : [B, 128]   (projected, for SupCon loss)
        """
        # 1) Visual features from DINOv3
        visual = self.backbone(rgb)     # [B, 1024]

        # 2) FiLM depth modulation
        gamma, beta = self.depth_film(depth)
        visual = gamma * visual + beta  # element-wise FiLM

        # 3) Text features (CLIP, frozen outside)
        text = self.text_proj(text_emb)  # [B, 512]

        # 4) Fuse
        fused = torch.cat([visual, text], dim=1)  # [B, 1536]
        fused = self.fusion_norm(fused)

        # 5) Classify + project
        logits = self.classifier(fused)            # [B, 4]
        proj   = F.normalize(self.proj_head(fused), dim=1)  # [B, 128] L2-normed

        return {"logits": logits, "embedding": fused, "proj": proj}


# ══════════════════════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).
    Pulls together embeddings of the same class, pushes apart different classes.
    Crucial for preventing collapse on rare classes.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def  forward(self, proj: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        proj:   [B, D] L2-normalized projections
        labels: [B]
        """
        B = proj.size(0)
        if B < 2:
            return torch.tensor(0.0, device=proj.device)

        sim = torch.mm(proj, proj.T) / self.temp   # [B, B]
        # Mask: True where same class (but not self)
        label_eq  = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        self_mask = ~torch.eye(B, dtype=torch.bool, device=proj.device)
        pos_mask  = label_eq & self_mask

        # If no positives in batch, return 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=proj.device)

        # Log-sum-exp over all non-self pairs
        exp_sim = torch.exp(sim) * self_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        log_prob  = sim - log_denom

        # Mean over positives
        loss = -(log_prob * pos_mask.float()).sum() / pos_mask.float().sum()
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def make_weighted_sampler(dataset: BuildingCropDataset) -> WeightedRandomSampler:
    """Class-balanced sampling: rare classes get upsampled."""
    labels     = dataset.df["final_usage"].map(CLASS2IDX).values
    class_cnt  = np.bincount(labels, minlength=NUM_CLASSES)
    class_w    = 1.0 / (class_cnt + 1e-6)
    sample_w   = class_w[labels]
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_w).float(),
        num_samples = len(dataset),
        replacement = True,
    )


def compute_class_weights(dataset: BuildingCropDataset) -> torch.Tensor:
    labels    = dataset.df["final_usage"].map(CLASS2IDX).values
    class_cnt = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    weights   = (class_cnt.sum() / (NUM_CLASSES * class_cnt + 1e-6))
    weights   = np.clip(weights, 0.5, 4.0)
    print(f"  Class weights: {dict(zip(CLASSES, weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


@torch.no_grad()
def evaluate(model: V5BuildingClassifier, loader: DataLoader,
             no_depth: bool = False) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        rgb, depth, text_emb, labels = batch[0], batch[1], batch[2], batch[3]
        rgb, depth, text_emb = rgb.to(DEVICE), depth.to(DEVICE), text_emb.to(DEVICE)
        if no_depth:
            depth = torch.zeros_like(depth)
        out    = model(rgb, depth, text_emb)
        preds  = out["logits"].argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import f1_score, classification_report
    wf1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(labels, preds, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    per_class = {c: round(report[c]["f1-score"], 4) for c in CLASSES}
    return {"wF1": round(wf1, 4), "per_class": per_class}


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor, T: float = 4.0) -> torch.Tensor:
    """KL divergence between softened teacher and student distributions."""
    stu = F.log_softmax(student_logits / T, dim=1)
    tch = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(stu, tch, reduction="batchmean") * (T ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

STAGE_CFG = {
    1: {
        "name":           "Stage1_Easy",
        "difficulties":   ["easy"],
        "epochs":         15,
        "lr":             2e-4,
        "freeze_until":   22,   # unfreeze last 2 ViT blocks (0-indexed, total=24)
        "hard_mix_ratio": 0.0,
        "distill":        False,
        "desc":           "Easy views only — stable feature learning",
    },
    2: {
        "name":           "Stage2_EasyMedium",
        "difficulties":   ["easy", "medium"],
        "epochs":         15,
        "lr":             1e-4,
        "freeze_until":   20,   # unfreeze last 4 ViT blocks
        "hard_mix_ratio": 0.0,
        "distill":        True,
        "desc":           "Add medium views + distill from Stage1",
    },
    3: {
        "name":           "Stage3_AllViews",
        "difficulties":   ["easy", "medium", "hard"],
        "epochs":         20,
        "lr":             5e-5,
        "freeze_until":   16,   # unfreeze last 8 ViT blocks
        "hard_mix_ratio": [0.25, 0.50, 1.0],  # progressively increase hard ratio
        "distill":        True,
        "desc":           "All views + gradual hard mixing + distillation",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_stage(
    stage: int,
    model: V5BuildingClassifier,
    teacher: V5BuildingClassifier | None,
    train_csv: str,
    val_csv: str,
    ocr_csv: str | None,
    text_cache: str | None,
    out_dir: Path,
    batch_size: int = 32,
    no_depth: bool = False,
    depth_dir: str | None = None,
):
    cfg = STAGE_CFG[stage]
    print(f"\n{'='*70}")
    print(f"  {cfg['name']}  |  {cfg['desc']}")
    print(f"  difficulties={cfg['difficulties']}  epochs={cfg['epochs']}  lr={cfg['lr']}")
    print(f"{'='*70}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = BuildingCropDataset(
        csv_path          = train_csv,
        ocr_csv           = ocr_csv,
        clip_text_cache   = text_cache,
        difficulty_filter = cfg["difficulties"],
        augment           = True,
        depth_dir         = depth_dir,
    )
    val_ds = BuildingCropDataset(
        csv_path          = val_csv,
        ocr_csv           = ocr_csv,
        clip_text_cache   = text_cache,
        difficulty_filter = None,
        augment           = False,
        depth_dir         = depth_dir,
    )
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"  Class dist (train): {train_ds.df['final_usage'].value_counts().to_dict()}")

    # ── Losses ────────────────────────────────────────────────────────────────
    class_w  = compute_class_weights(train_ds)
    focal    = FocalLoss(gamma=2.0, alpha=class_w).to(DEVICE)
    supcon   = SupConLoss(temperature=0.07)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    model.set_frozen_blocks(freeze_until=cfg["freeze_until"])
    model.to(DEVICE)

    # Separate LR for backbone (lower) vs head (higher)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = (list(model.depth_film.parameters()) +
                       list(model.text_proj.parameters()) +
                       list(model.fusion_norm.parameters()) +
                       list(model.classifier.parameters()) +
                       list(model.proj_head.parameters()))
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg["lr"] * 0.1},
        {"params": head_params,     "lr": cfg["lr"]},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    # ── Hard mix schedule (Stage 3 only) ──────────────────────────────────────
    hard_mix_schedule = cfg.get("hard_mix_ratio", 0.0)
    if isinstance(hard_mix_schedule, list):
        # e.g. [0.25, 0.50, 1.0] → assign per-epoch thresholds
        thresholds = hard_mix_schedule
        hard_ds_all = BuildingCropDataset(
            csv_path          = train_csv,
            ocr_csv           = ocr_csv,
            clip_text_cache   = text_cache,
            difficulty_filter = ["hard"],
            augment           = True,
            depth_dir         = depth_dir,
        )
    else:
        thresholds  = None
        hard_ds_all = None

    best_wf1 = 0.0
    best_path = out_dir / f"v5_{cfg['name']}_best.pth"
    history   = []
    log_path  = out_dir / f"v5_{cfg['name']}_log.txt"
    scaler    = GradScaler("cuda", enabled=DEVICE.type == "cuda")

    def write_log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Hard mix ratio update (Stage 3) ───────────────────────────────────
        if thresholds is not None and hard_ds_all is not None:
            t_idx = min(int((epoch - 1) / cfg["epochs"] * len(thresholds)),
                        len(thresholds) - 1)
            hard_ratio = thresholds[t_idx]
        else:
            hard_ratio = 0.0

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        total_loss = ce_loss_sum = focal_sum = supcon_sum = distill_sum = 0.0
        n_batches  = 0
        t_epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            rgb, depth, text_emb, labels, diff = batch[0], batch[1], batch[2], batch[3], batch[4]
            rgb      = rgb.to(DEVICE, non_blocking=True)
            depth    = depth.to(DEVICE, non_blocking=True)
            text_emb = text_emb.to(DEVICE, non_blocking=True)
            labels   = labels.to(DEVICE, non_blocking=True)
            if no_depth:
                depth = torch.zeros_like(depth)

            with autocast("cuda", enabled=DEVICE.type == "cuda"):
                out    = model(rgb, depth, text_emb)
                logits = out["logits"]
                proj   = out["proj"]

                ce = F.cross_entropy(logits, labels, weight=class_w)
                fl = focal(logits, labels)
                sc = supcon(proj, labels)

                dl = torch.tensor(0.0, device=DEVICE)
                if teacher is not None and cfg["distill"]:
                    with torch.no_grad():
                        t_out = teacher(rgb, depth, text_emb)
                    dl = distillation_loss(logits, t_out["logits"])

                loss = ce + 0.5 * fl + 0.3 * sc + 0.2 * dl

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss  += loss.item()
            ce_loss_sum += ce.item()
            focal_sum   += fl.item()
            supcon_sum  += sc.item()
            distill_sum += dl.item()
            n_batches   += 1

            # Progress every 50 batches
            if batch_idx % 50 == 0:
                elapsed = time.time() - t_epoch_start
                write_log(f"  Ep{epoch} [{batch_idx}/{len(train_loader)}] "
                          f"loss={loss.item():.4f} elapsed={elapsed:.0f}s")

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, no_depth=no_depth)
        wf1     = metrics["wF1"]

        avg = total_loss / max(n_batches, 1)
        epoch_time = time.time() - t_epoch_start
        msg = (f"  Epoch {epoch:3d}/{cfg['epochs']} | "
               f"loss={avg:.4f} ce={ce_loss_sum/n_batches:.3f} "
               f"fl={focal_sum/n_batches:.3f} sc={supcon_sum/n_batches:.3f} | "
               f"wF1={wf1:.4f}  {metrics['per_class']} | {epoch_time:.0f}s")
        write_log(msg)

        history.append({"epoch": epoch, "wF1": wf1, "loss": avg,
                        "per_class": metrics["per_class"],
                        "epoch_time_s": round(epoch_time, 1)})

        if wf1 > best_wf1:
            best_wf1 = wf1
            torch.save(model.state_dict(), best_path)
            write_log(f"  *** New best: wF1={wf1:.4f} → {best_path}")

    print(f"\n  Stage {stage} done. Best wF1={best_wf1:.4f}")

    # Save history
    with open(out_dir / f"v5_{cfg['name']}_history.json", "w") as f:
        json.dump({"best_wF1": best_wf1, "history": history}, f, indent=2)

    return best_path, best_wf1


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",       type=int,  default=0,
                        help="0=all stages, 1/2/3=specific stage")
    parser.add_argument("--resume_from", type=str,  default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--train_csv",   default="data/annotations/clean_train_labels.csv")
    parser.add_argument("--val_csv",     default="data/annotations/clean_val_labels.csv")
    parser.add_argument("--ocr_csv",     default="data/ocr_outputs/ocr_results.csv")
    parser.add_argument("--text_cache",  default="data/ocr_outputs/clip_text_embeds.pkl")
    parser.add_argument("--dinov3_ckpt", default="final_models/ssl_model/dinov3_rgbd_ssl.pth")
    parser.add_argument("--depth_dir",   default="data/depth_maps",
                        help="Dir of precomputed {stem}_depth.npy maps for FiLM "
                             "(see src/gis/save_depth_maps.py). Falls back to a "
                             "constant map from mean_depth_m when missing.")
    parser.add_argument("--batch_size",  type=int,  default=32)
    parser.add_argument("--out_dir",     default="outputs/checkpoints")
    # ── Ablation flags ────────────────────────────────────────────────────────
    parser.add_argument("--no_depth", action="store_true",
                        help="Zero out depth input (RGB-only ablation)")
    parser.add_argument("--no_ocr",   action="store_true",
                        help="Zero out OCR text embeddings (no-OCR ablation)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  Building Usage Classifier v5")
    print(f"  DEVICE: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    # ── Build model ───────────────────────────────────────────────────────────
    model = V5BuildingClassifier()
    model.load_backbone(args.dinov3_ckpt, freeze_blocks_until=22)

    if args.resume_from and Path(args.resume_from).exists():
        state = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        print(f"  Resumed from: {args.resume_from}")

    model.to(DEVICE)

    stages = [args.stage] if args.stage > 0 else [1, 2, 3]
    teacher = None
    prev_ckpt = args.resume_from

    for s in stages:
        best_path, best_wf1 = train_stage(
            stage      = s,
            model      = model,
            teacher    = teacher,
            train_csv  = args.train_csv,
            val_csv    = args.val_csv,
            ocr_csv    = None if args.no_ocr else args.ocr_csv,
            text_cache = None if args.no_ocr else args.text_cache,
            out_dir    = out_dir,
            batch_size = args.batch_size,
            no_depth   = args.no_depth,
            depth_dir  = args.depth_dir,
        )
        # Teacher for next stage = current best
        teacher = deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Load best checkpoint before next stage
        state = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state, strict=False)
        prev_ckpt = best_path

        print(f"\n  Stage {s} complete. Best wF1={best_wf1:.4f}")
        print(f"  Best checkpoint: {best_path}")

    # ── Save final ────────────────────────────────────────────────────────────
    final_path = out_dir / "v5_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n  Final model saved: {final_path}")


if __name__ == "__main__":
    main()
