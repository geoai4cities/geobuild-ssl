# src/ssl/detection_head_dinov3_ssl.py
"""
Detection Head Fine-tuning on SSL-Pretrained DINOv3 Backbone
=============================================================

Identical to detection_head_dinov3.py but uses the RGBD SSL-fine-tuned
backbone instead of raw ImageNet DINOv3 weights.

Backbone checkpoint : out/ssl_dinov3_rgbd/ssl_dinov3_rgbd_backbone.pth
  (ViT-L/16, fine-tuned on 63K unlabelled RGBD street-level images via
   masked-image-modelling + depth-contrastive SSL)

Architecture:
  - DINOv3 ViT-L/16 (SSL fine-tuned, completely frozen)
    → Intermediate patch tokens from blocks 8, 16, 23
    → Reshaped to 3 spatial feature maps  [B, 1024, 32, 32]
  - Lightweight FPN neck  → 256 channels at each of 3 levels
  - FCOS head (per level):
      cls  branch → objectness score  [1, H, W]
      reg  branch → lrtb distances    [4, H, W]
      ctr  branch → centerness score  [1, H, W]

Data:
  - Images  : data/curriculum/{split}/{diff}/{country}/{city}/*.jpg
  - Depths  : data/curriculum_depths/{split}/{diff}/{country}/{city}/*.npy
  - Labels  : data/annotations/instance_labels.csv
              (per-view building boxes; generate with the annotation pipeline,
               src/annotation/annotate.py — not shipped with the repo)

Usage:
    python src/ssl/detection_head_dinov3_ssl.py
"""

import os, sys, csv, time, math, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (only DINOV3_CKPT and OUT_DIR differ from detection_head_dinov3.py)
# ──────────────────────────────────────────────────────────────────────────────
ANNO_CSV        = "data/annotations/instance_labels.csv"
CURRICULUM_ROOT = "data/curriculum"
DEPTH_ROOT      = "data/curriculum_depths"
DINOV3_CKPT     = "final_models/ssl_model/dinov3_rgbd_ssl.pth"  # SSL backbone
OUT_DIR         = "out/detection_head_ssl"
CKPT_DIR        = os.path.join(OUT_DIR, "checkpoints")
VIS_DIR         = os.path.join(OUT_DIR, "visualizations")

# Model
IMG_SIZE        = 512
FPN_CH          = 256
FPN_LEVELS      = 3
STRIDES         = [16, 32, 64]
SCALE_RANGES    = [(0, 96), (64, 192), (128, 1e6)]
CENTER_SAMPLE_R = 1.5

# Training
EPOCHS          = 80
BATCH_SIZE      = 6
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
WARMUP_EPOCHS   = 5
TRAIN_SPLIT     = 0.85
NUM_WORKERS     = 8
MIXED_PREC      = True
LOG_INTERVAL    = 20
EVAL_INTERVAL   = 5

# Data filters
MIN_BOX_AREA    = 400
MIN_BOX_SIDE    = 16
DEPTH_THRESH    = 30.0

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DIR,  exist_ok=True)

if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
class BuildingDetectionDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, samples: list, augment: bool = True):
        self.samples  = samples
        self.augment  = augment
        self.to_tensor = transforms.ToTensor()
        self.normalise = transforms.Normalize(self.MEAN, self.STD)

    def _load_depth(self, depth_path):
        try:
            depth = np.load(depth_path).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
        except Exception:
            depth = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        from PIL import Image as PILImg
        d_img  = PILImg.fromarray(np.clip(depth * 255, 0, 255).astype(np.uint8))
        d_img  = d_img.resize((IMG_SIZE, IMG_SIZE), PILImg.BILINEAR)
        d_norm = np.array(d_img, dtype=np.float32) / 255.0

        import cv2
        sobx  = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
        soby  = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(sobx**2 + soby**2)
        if edges.max() > 0:
            edges /= edges.max()
        lap = cv2.Laplacian(d_norm, cv2.CV_32F, ksize=3)
        rng = lap.max() - lap.min()
        if rng > 0:
            lap = (lap - lap.min()) / rng

        return torch.from_numpy(np.stack([d_norm, edges, lap], axis=0))

    def _resize_boxes(self, boxes, orig_w, orig_h):
        sx = IMG_SIZE / orig_w
        sy = IMG_SIZE / orig_h
        b  = boxes.clone()
        b[:, 0] *= sx;  b[:, 2] *= sx
        b[:, 1] *= sy;  b[:, 3] *= sy
        return b.clamp(0, IMG_SIZE)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path, boxes_np, (orig_w, orig_h) = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        orig_w_real, orig_h_real = img.size
        img   = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        boxes = torch.tensor(boxes_np, dtype=torch.float32)
        boxes = self._resize_boxes(boxes, orig_w_real, orig_h_real)

        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x1 = boxes[:, 0].clone()
                boxes[:, 0] = IMG_SIZE - boxes[:, 2]
                boxes[:, 2] = IMG_SIZE - x1
            if random.random() > 0.3:
                img = transforms.ColorJitter(0.3, 0.3, 0.2, 0.05)(img)

        img_t   = self.normalise(self.to_tensor(img))
        depth_t = self._load_depth(depth_path)

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        valid = (w * h >= MIN_BOX_AREA) & (w >= MIN_BOX_SIDE) & (h >= MIN_BOX_SIDE)
        return img_t, depth_t, boxes[valid]


def collate_fn(batch):
    imgs   = torch.stack([b[0] for b in batch])
    depths = torch.stack([b[1] for b in batch])
    boxes  = [b[2] for b in batch]
    return imgs, depths, boxes


def build_datasets():
    img2boxes = defaultdict(list)

    with open(ANNO_CSV) as f:
        for row in csv.DictReader(f):
            src = row["source_image"]
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1","y1","x2","y2"))
            depth_m = float(row.get("mean_depth", 0) or 0)
            if depth_m > DEPTH_THRESH:
                continue
            w, h = x2 - x1, y2 - y1
            if w * h < MIN_BOX_AREA or w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
                continue
            img2boxes[src].append([x1, y1, x2, y2])

    samples = []
    missing = 0
    for src, boxes in img2boxes.items():
        if not os.path.exists(src):
            missing += 1
            continue
        rel        = os.path.relpath(src, CURRICULUM_ROOT)
        depth_path = os.path.join(DEPTH_ROOT, os.path.splitext(rel)[0] + ".npy")
        if not os.path.exists(depth_path):
            depth_path_png = os.path.join(DEPTH_ROOT,
                                          os.path.splitext(rel)[0] + ".png")
            depth_path = depth_path_png if os.path.exists(depth_path_png) else depth_path
        try:
            with Image.open(src) as im:
                orig_size = im.size
        except Exception:
            continue
        samples.append((src, depth_path,
                        np.array(boxes, dtype=np.float32), orig_size))

    print(f"[DATA] Annotated images: {len(img2boxes)} "
          f"| Loaded: {len(samples)} | Missing: {missing}")

    random.shuffle(samples)
    n_train = int(len(samples) * TRAIN_SPLIT)
    train_s, val_s = samples[:n_train], samples[n_train:]
    print(f"[DATA] Train: {len(train_s)} | Val: {len(val_s)}")

    return (BuildingDetectionDataset(train_s, augment=True),
            BuildingDetectionDataset(val_s,   augment=False))


# ──────────────────────────────────────────────────────────────────────────────
# BACKBONE: SSL-fine-tuned DINOv3  (FROZEN)
# ──────────────────────────────────────────────────────────────────────────────
class DINOv3Backbone(nn.Module):
    """
    SSL-fine-tuned DINOv3 ViT-L/16 (completely frozen).
    Loads from out/ssl_dinov3_rgbd/ssl_dinov3_rgbd_backbone.pth — a clean
    ViT-L/16 OrderedDict with 294 keys (patch_embed, blocks.0-23, norm).
    Extracts patch tokens from blocks [8, 16, 23].
    """
    EXTRACT_BLOCKS = [8, 16, 23]

    def __init__(self, checkpoint_path: str):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("pip install timm")

        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="",
            img_size=IMG_SIZE,
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # SSL backbone is saved as plain OrderedDict
        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "teacher", "backbone"):
                if key in ckpt:
                    state_dict = ckpt[key]
                    break

        cleaned = {k.replace("module.", "").replace("backbone.", ""): v
                   for k, v in state_dict.items()}

        # Interpolate pos_embed if it was saved at a different resolution.
        # SSL backbone saved at 224px → [1, 197, 1024] (14×14 patches + CLS).
        # We need                      → [1, 1025, 1024] (32×32 patches + CLS)
        # for IMG_SIZE=512 with patch_size=16.
        if "pos_embed" in cleaned:
            pe = cleaned["pos_embed"]           # [1, N_src+1, C]
            n_src = pe.shape[1] - 1             # number of patch tokens in ckpt
            n_dst = (IMG_SIZE // 16) ** 2       # target patch tokens
            if n_src != n_dst:
                cls_pe  = pe[:, :1, :]          # [1, 1, C]
                patch_pe = pe[:, 1:, :]         # [1, n_src, C]
                src_hw  = int(n_src ** 0.5)     # 14 for 224px
                dst_hw  = IMG_SIZE // 16        # 32 for 512px
                # Reshape to spatial grid, bicubic interpolate, flatten back
                patch_pe = (patch_pe
                            .reshape(1, src_hw, src_hw, -1)
                            .permute(0, 3, 1, 2))              # [1, C, src_hw, src_hw]
                patch_pe = F.interpolate(patch_pe,
                                         size=(dst_hw, dst_hw),
                                         mode="bicubic",
                                         align_corners=False)  # [1, C, dst_hw, dst_hw]
                patch_pe = (patch_pe
                            .permute(0, 2, 3, 1)
                            .reshape(1, dst_hw * dst_hw, -1))  # [1, n_dst, C]
                cleaned["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)
                print(f"[BACKBONE-SSL] pos_embed interpolated  "
                      f"{n_src} → {n_dst} patches  ({src_hw}px → {dst_hw*16}px grid)")

        msg = self.model.load_state_dict(cleaned, strict=False)
        print(f"[BACKBONE-SSL] Loaded  "
              f"missing={len(msg.missing_keys)}  "
              f"unexpected={len(msg.unexpected_keys)}")

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.patch_size = 16
        self.feat_dim   = self.model.embed_dim   # 1024

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)

        features = []
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i in self.EXTRACT_BLOCKS:
                patch = x[:, 1:]
                feat  = patch.permute(0, 2, 1)
                feat  = feat.reshape(B, self.feat_dim, H_p, W_p)
                features.append(feat)

        return features   # [B, 1024, 32, 32] × 3


# ──────────────────────────────────────────────────────────────────────────────
# FPN NECK
# ──────────────────────────────────────────────────────────────────────────────
class FPN(nn.Module):
    """
    3-level FPN.  All DINOv3 features are 32×32 → uses strided convs to
    build P4 (16×16) and P5 (8×8).
    """

    def __init__(self, in_ch: int = 1024, out_ch: int = FPN_CH):
        super().__init__()
        self.lat3 = nn.Conv2d(in_ch, out_ch, 1)
        self.lat4 = nn.Conv2d(in_ch, out_ch, 1)
        self.lat5 = nn.Conv2d(in_ch, out_ch, 1)

        self.down_3to4 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.down_4to5 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

        self.out3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                   nn.BatchNorm2d(out_ch), nn.ReLU())
        self.out4 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                   nn.BatchNorm2d(out_ch), nn.ReLU())
        self.out5 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                   nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, feats):
        c3, c4, c5 = feats   # all [B, 1024, 32, 32]

        l3 = self.lat3(c3)
        l4 = self.lat4(c4)
        l5 = self.lat5(c5)

        p3_pre = l3
        p4_pre = self.down_3to4(p3_pre) + self.down_3to4(l4)
        l5_8x8 = self.down_4to5(self.down_3to4(l5))
        p5_pre = self.down_4to5(p4_pre) + l5_8x8

        p5 = p5_pre
        p4 = p4_pre + F.interpolate(p5, size=p4_pre.shape[-2:], mode="nearest")
        p3 = p3_pre + F.interpolate(p4, size=p3_pre.shape[-2:], mode="nearest")

        return self.out3(p3), self.out4(p4), self.out5(p5)


# ──────────────────────────────────────────────────────────────────────────────
# FCOS HEAD
# ──────────────────────────────────────────────────────────────────────────────
class FCOSHead(nn.Module):
    def __init__(self, in_ch: int = FPN_CH, num_convs: int = 3):
        super().__init__()

        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GroupNorm(32, c_out),
                nn.ReLU(),
            )

        self.cls_tower = nn.Sequential(
            *[conv_block(in_ch, in_ch) for _ in range(num_convs)])
        self.reg_tower = nn.Sequential(
            *[conv_block(in_ch, in_ch) for _ in range(num_convs)])

        self.cls_pred = nn.Conv2d(in_ch, 1, 1)
        self.reg_pred = nn.Conv2d(in_ch, 4, 1)
        self.ctr_pred = nn.Conv2d(in_ch, 1, 1)
        self.scales   = nn.Parameter(torch.ones(FPN_LEVELS))

        nn.init.constant_(self.cls_pred.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, fpn_feats):
        cls_preds, reg_preds, ctr_preds = [], [], []
        for i, feat in enumerate(fpn_feats):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)
            cls_preds.append(self.cls_pred(cls_feat))
            reg_preds.append(torch.exp(self.scales[i]) *
                              torch.exp(self.reg_pred(reg_feat)))
            ctr_preds.append(self.ctr_pred(cls_feat))
        return cls_preds, reg_preds, ctr_preds


# ──────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ──────────────────────────────────────────────────────────────────────────────
class BuildingDetector(nn.Module):
    """SSL-DINOv3 (frozen) + FPN + FCOS."""

    def __init__(self, dinov3_ckpt: str):
        super().__init__()
        self.backbone = DINOv3Backbone(dinov3_ckpt)
        self.fpn      = FPN(in_ch=self.backbone.feat_dim, out_ch=FPN_CH)
        self.head     = FCOSHead(in_ch=FPN_CH)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[MODEL] Total: {total/1e6:.1f}M  "
              f"Trainable (FPN+Head): {trainable/1e6:.1f}M")

    def forward(self, img, _depth=None):
        feats = self.backbone(img)
        fpn   = self.fpn(feats)
        return self.head(fpn)


# ──────────────────────────────────────────────────────────────────────────────
# LABEL ASSIGNMENT
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def fcos_assign(gt_boxes: torch.Tensor, device: torch.device):
    cls_out, reg_out, ctr_out = [], [], []

    for lvl, (stride, (smin, smax)) in enumerate(zip(STRIDES, SCALE_RANGES)):
        H = IMG_SIZE // stride
        W = IMG_SIZE // stride

        ys = (torch.arange(H, device=device).float() + 0.5) * stride
        xs = (torch.arange(W, device=device).float() + 0.5) * stride
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)

        if gt_boxes.numel() == 0:
            cls_out.append(torch.zeros(H * W, dtype=torch.bool,  device=device))
            reg_out.append(torch.zeros(H * W, 4,                 device=device))
            ctr_out.append(torch.zeros(H * W,                    device=device))
            continue

        N  = gt_boxes.shape[0]
        gx = gx.unsqueeze(1).expand(-1, N)
        gy = gy.unsqueeze(1).expand(-1, N)
        x1, y1, x2, y2 = (gt_boxes[:, i] for i in range(4))

        l = gx - x1
        r = x2 - gx
        t = gy - y1
        b = y2 - gy

        inside_box = (l > 0) & (r > 0) & (t > 0) & (b > 0)
        max_dist   = torch.stack([l, r, t, b], dim=-1).max(dim=-1).values
        in_range   = (max_dist >= smin) & (max_dist <= smax)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        in_center = ((gx - cx).abs() < CENTER_SAMPLE_R * stride) & \
                    ((gy - cy).abs() < CENTER_SAMPLE_R * stride)

        valid = inside_box & in_range & in_center

        areas            = (x2 - x1) * (y2 - y1)
        areas_expanded   = areas.unsqueeze(0).expand_as(valid)
        areas_expanded   = areas_expanded.masked_fill(~valid, 1e8)
        min_area, best_gt = areas_expanded.min(dim=1)
        is_pos           = min_area < 1e7

        best_gt_c  = best_gt.clamp(0, N - 1)
        matched    = gt_boxes[best_gt_c]
        loc_x = (torch.arange(H * W, device=device) % W + 0.5) * stride
        loc_y = (torch.arange(H * W, device=device) // W + 0.5) * stride

        lrtb = torch.stack([
            loc_x - matched[:, 0],
            matched[:, 2] - loc_x,
            loc_y - matched[:, 1],
            matched[:, 3] - loc_y,
        ], dim=1)

        lrtb_c = lrtb.clamp(min=1e-6)
        l_, r_, t_, b_ = lrtb_c[:, 0], lrtb_c[:, 1], lrtb_c[:, 2], lrtb_c[:, 3]
        ctr = torch.sqrt(
            (torch.min(l_, r_) / (torch.max(l_, r_) + 1e-6)) *
            (torch.min(t_, b_) / (torch.max(t_, b_) + 1e-6))
        )

        cls_out.append(is_pos)
        reg_out.append(lrtb)
        ctr_out.append(ctr)

    return cls_out, reg_out, ctr_out


# ──────────────────────────────────────────────────────────────────────────────
# LOSSES
# ──────────────────────────────────────────────────────────────────────────────
def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    p  = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.where(target == 1, p, 1 - p)
    return (alpha * (1 - pt) ** gamma * ce).sum()


def giou_loss(pred_lrtb, gt_lrtb, loc_x, loc_y):
    eps = 1e-6
    pl, pr = pred_lrtb[:, 0], pred_lrtb[:, 1]
    pt, pb = pred_lrtb[:, 2], pred_lrtb[:, 3]
    gl, gr = gt_lrtb[:, 0],   gt_lrtb[:, 1]
    gt_, gb = gt_lrtb[:, 2],  gt_lrtb[:, 3]

    px1, px2 = loc_x - pl, loc_x + pr
    py1, py2 = loc_y - pt, loc_y + pb
    gx1, gx2 = loc_x - gl, loc_x + gr
    gy1, gy2 = loc_y - gt_, loc_y + gb

    inter = ((torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(0) *
             (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(0))
    pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    gt_area   = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = pred_area + gt_area - inter + eps
    iou   = inter / union

    enc = ((torch.max(px2, gx2) - torch.min(px1, gx1)).clamp(0) *
           (torch.max(py2, gy2) - torch.min(py1, gy1)).clamp(0) + eps)
    giou = iou - (enc - union) / enc
    return (1 - giou).mean()


def compute_loss(cls_preds, reg_preds, ctr_preds, gt_boxes_batch, device):
    total_cls = torch.tensor(0., device=device)
    total_reg = torch.tensor(0., device=device)
    total_ctr = torch.tensor(0., device=device)
    n_pos_total = 0

    for b_idx, gt_boxes in enumerate(gt_boxes_batch):
        gt_boxes = gt_boxes.to(device)
        cls_tgts, reg_tgts, ctr_tgts = fcos_assign(gt_boxes, device)

        for lvl in range(FPN_LEVELS):
            stride = STRIDES[lvl]
            H = IMG_SIZE // stride
            W = IMG_SIZE // stride

            cls_p = cls_preds[lvl][b_idx].permute(1, 2, 0).reshape(-1, 1)
            reg_p = reg_preds[lvl][b_idx].permute(1, 2, 0).reshape(-1, 4)
            ctr_p = ctr_preds[lvl][b_idx].permute(1, 2, 0).reshape(-1, 1)
            is_pos = cls_tgts[lvl].to(device)
            n_pos  = is_pos.sum().item()

            total_cls = total_cls + focal_loss(cls_p.squeeze(1), is_pos.float())

            if n_pos > 0:
                n_pos_total += n_pos
                loc_x = (torch.arange(H * W, device=device) % W + 0.5) * stride
                loc_y = (torch.arange(H * W, device=device) // W + 0.5) * stride
                total_reg = total_reg + giou_loss(
                    reg_p[is_pos], reg_tgts[lvl].to(device)[is_pos],
                    loc_x[is_pos], loc_y[is_pos],
                )
                total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                    ctr_p.squeeze(1)[is_pos], ctr_tgts[lvl].to(device)[is_pos]
                )

    norm = max(n_pos_total, 1)
    return total_cls / norm + total_reg + total_ctr, {
        "cls": total_cls.item() / norm,
        "reg": total_reg.item(),
        "ctr": total_ctr.item(),
        "n_pos": n_pos_total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE / NMS  (exported for detect_buildings.py)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def decode_predictions(cls_preds, reg_preds, ctr_preds,
                        score_thresh: float = 0.25,
                        nms_thresh:   float = 0.3):
    all_boxes, all_scores = [], []

    for lvl in range(FPN_LEVELS):
        stride = STRIDES[lvl]
        H = IMG_SIZE // stride
        W = IMG_SIZE // stride

        cls_s = torch.sigmoid(cls_preds[lvl][0]).squeeze(0)
        reg_s = reg_preds[lvl][0]
        ctr_s = torch.sigmoid(ctr_preds[lvl][0]).squeeze(0)
        scores = (cls_s * ctr_s).reshape(-1)

        ys = (torch.arange(H, device=cls_s.device).float() + 0.5) * stride
        xs = (torch.arange(W, device=cls_s.device).float() + 0.5) * stride
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.reshape(-1); gy = gy.reshape(-1)

        l = reg_s[0].reshape(-1); r = reg_s[1].reshape(-1)
        t = reg_s[2].reshape(-1); b = reg_s[3].reshape(-1)

        x1 = (gx - l).clamp(0, IMG_SIZE)
        y1 = (gy - t).clamp(0, IMG_SIZE)
        x2 = (gx + r).clamp(0, IMG_SIZE)
        y2 = (gy + b).clamp(0, IMG_SIZE)

        keep = scores > score_thresh
        all_boxes.append(torch.stack([x1[keep], y1[keep], x2[keep], y2[keep]], 1))
        all_scores.append(scores[keep])

    if not any(b.numel() > 0 for b in all_boxes):
        return torch.zeros((0, 4)), torch.zeros(0)

    boxes  = torch.cat(all_boxes,  dim=0)
    scores = torch.cat(all_scores, dim=0)

    try:
        from torchvision.ops import nms
        idx = nms(boxes, scores, nms_thresh)
        return boxes[idx], scores[idx]
    except Exception:
        topk = scores.topk(min(50, len(scores))).indices
        return boxes[topk], scores[topk]


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, iou_thresh=0.5):
    model.eval()
    tp_total = fp_total = fn_total = 0

    for imgs, depths, gt_boxes_batch in loader:
        imgs = imgs.to(device)
        cls_preds, reg_preds, ctr_preds = model(imgs)

        for b in range(imgs.shape[0]):
            cls_b = [c[b:b+1] for c in cls_preds]
            reg_b = [r[b:b+1] for r in reg_preds]
            ctr_b = [c[b:b+1] for c in ctr_preds]
            pred_boxes, _ = decode_predictions(cls_b, reg_b, ctr_b)
            gt_b = gt_boxes_batch[b].to(device)

            if gt_b.numel() == 0:
                fp_total += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                fn_total += len(gt_b)
                continue

            from torchvision.ops import box_iou
            iou_mat    = box_iou(pred_boxes, gt_b)
            matched_gt = set()
            for p in range(len(pred_boxes)):
                best_iou, best_g = iou_mat[p].max(0)
                if best_iou >= iou_thresh and best_g.item() not in matched_gt:
                    tp_total += 1
                    matched_gt.add(best_g.item())
                else:
                    fp_total += 1
            fn_total += len(gt_b) - len(matched_gt)

    prec   = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1     = 2 * prec * recall / max(prec + recall, 1e-6)
    return {"precision": prec, "recall": recall, "f1": f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total}


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def format_time(s):
    return f"{int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d}"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    train_ds, val_ds = build_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False,
                              num_workers=4, collate_fn=collate_fn)

    model = BuildingDetector(DINOV3_CKPT).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable_params, lr=LR,
                                  weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        pct = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * pct))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=MIXED_PREC)

    start_epoch = 0
    best_recall = 0.0
    latest_ckpt = os.path.join(CKPT_DIR, "latest.pth")
    if os.path.exists(latest_ckpt):
        ck = torch.load(latest_ckpt, map_location=device)
        model.fpn.load_state_dict(ck["fpn"])
        model.head.load_state_dict(ck["head"])
        optimiser.load_state_dict(ck["optimiser"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"]
        best_recall = ck.get("best_recall", 0)
        print(f"[RESUME] epoch {start_epoch}  best_recall {best_recall:.4f}")

    history = []
    print("=" * 70)
    print("DETECTION HEAD TRAINING  –  SSL DINOv3 (frozen) + FPN + FCOS")
    print("=" * 70)
    print(f"  SSL backbone : {DINOV3_CKPT}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  LR           : {LR}")
    print(f"  IMG_SIZE     : {IMG_SIZE}")
    print(f"  FPN levels   : P3(32×32) / P4(16×16) / P5(8×8)")
    print(f"  Train imgs   : {len(train_ds)}   Val imgs : {len(val_ds)}")
    print("=" * 70)

    global_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        model.backbone.eval()
        epoch_start = time.time()
        running     = defaultdict(float)

        for step, (imgs, depths, gt_boxes_batch) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=MIXED_PREC):
                cls_preds, reg_preds, ctr_preds = model(imgs)
                loss, loss_dict = compute_loss(
                    cls_preds, reg_preds, ctr_preds,
                    gt_boxes_batch, device
                )

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimiser)
            scaler.update()

            for k, v in loss_dict.items():
                running[k] += v

            if step % LOG_INTERVAL == 0:
                elapsed    = time.time() - global_start
                total_steps = EPOCHS * len(train_loader)
                done_steps  = epoch * len(train_loader) + step + 1
                eta         = elapsed / done_steps * (total_steps - done_steps)
                print(f"  [E{epoch+1:03d}/{EPOCHS} S{step:04d}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}  "
                      f"cls={loss_dict['cls']:.3f}  "
                      f"reg={loss_dict['reg']:.3f}  "
                      f"ctr={loss_dict['ctr']:.3f}  "
                      f"n_pos={loss_dict['n_pos']}  "
                      f"ETA={format_time(eta)}")

        scheduler.step()

        n   = len(train_loader)
        avg = {k: v / n for k, v in running.items()}
        print(f"[Epoch {epoch+1}] "
              f"cls={avg['cls']:.4f}  reg={avg['reg']:.4f}  "
              f"ctr={avg['ctr']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={format_time(time.time() - epoch_start)}")

        metrics = {}
        if (epoch + 1) % EVAL_INTERVAL == 0:
            print("  [EVAL] ...")
            metrics = evaluate(model, val_loader, device)
            print(f"  [EVAL] P={metrics['precision']:.4f}  "
                  f"R={metrics['recall']:.4f}  "
                  f"F1={metrics['f1']:.4f}  "
                  f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")

        ck = {
            "epoch":       epoch + 1,
            "fpn":         model.fpn.state_dict(),
            "head":        model.head.state_dict(),
            "optimiser":   optimiser.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "best_recall": best_recall,
            "loss":        avg,
            "metrics":     metrics,
        }
        torch.save(ck, latest_ckpt)
        if (epoch + 1) % 10 == 0:
            torch.save(ck, os.path.join(CKPT_DIR, f"epoch_{epoch+1:03d}.pth"))

        recall = metrics.get("recall", 0)
        if recall > best_recall:
            best_recall = recall
            ck["best_recall"] = best_recall
            torch.save(ck, os.path.join(CKPT_DIR, "best.pth"))
            print(f"  -> New best recall: {best_recall:.4f}")

        history.append({
            "epoch": epoch + 1, "loss": avg,
            "lr": scheduler.get_last_lr()[0], **metrics
        })

    total_time = time.time() - global_start
    final_path = os.path.join(OUT_DIR, "detection_head_ssl_final.pth")
    torch.save({
        "fpn":         model.fpn.state_dict(),
        "head":        model.head.state_dict(),
        "backbone":    DINOV3_CKPT,
        "img_size":    IMG_SIZE,
        "fpn_ch":      FPN_CH,
        "strides":     STRIDES,
    }, final_path)

    with open(os.path.join(OUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"Training complete in {format_time(total_time)}")
    print(f"Best recall @IoU=0.5 : {best_recall:.4f}")
    print(f"Model saved to       : {final_path}")
    print("=" * 70)
