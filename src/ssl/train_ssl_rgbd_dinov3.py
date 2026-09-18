# src/ssl/train_ssl_rgbd_dinov3.py
"""
Usage:
    python src/ssl/train_ssl_rgbd_dinov3.py
"""

import os
import sys
import cv2
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

# ==================================================
# CONFIG - OPTIMIZED FOR RTX 4070 Ti SUPER 16GB
# ==================================================
RGB_ROOT = "data/ssl_pretraining/input"
DEPTH_ROOT = "data/ssl_pretraining/depth"
DINOV3_CHECKPOINT = "final_models/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
OUT_DIR = "out/ssl_dinov3_rgbd"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

EPOCHS = 50

# VRAM-optimized for 16GB with unfrozen DINOv3
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 16  # effective batch = 128
NUM_WORKERS = 14
MIXED_PRECISION = True

# Differential learning rates
DINOV3_LR = 1e-5     # Pretrained DINOv3: very small updates
DEPTH_LR = 5e-5      # Depth encoder: moderate
PROJECTION_LR = 1e-3 # Projection head: learning from scratch

MODALITY_DROPOUT = 0.2
LOG_INTERVAL = 50
GRADIENT_CHECKPOINTING = True  # Saves ~2-3GB VRAM

os.makedirs(CKPT_DIR, exist_ok=True)

if torch.cuda.is_available():
    print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ==================================================
# DEPTH FEATURE EXTRACTION (from rgbd_curriculum_v2)
# ==================================================
def extract_depth_edges(depth_map):
    """Depth discontinuities via Sobel."""
    depth_float = depth_map.astype(np.float32)
    sobelx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    if edges.max() > 0:
        edges = edges / edges.max()
    return edges.astype(np.float32)


def extract_depth_structure(depth_map):
    """Structural features via Laplacian."""
    depth_float = depth_map.astype(np.float32)
    laplacian = cv2.Laplacian(depth_float, cv2.CV_32F, ksize=3)
    if laplacian.max() - laplacian.min() > 0:
        laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
    return laplacian.astype(np.float32)


# ==================================================
# DATASET
# ==================================================
class DINOv3RGBDDataset(Dataset):
    """RGB-D dataset returning separate RGB and 3-channel geometric depth."""

    def __init__(self, rgb_root, depth_root, rgb_transform, img_size=224):
        self.rgb_transform = rgb_transform
        self.img_size = img_size
        self.samples = []

        for root, _, files in os.walk(rgb_root):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    rgb_path = os.path.join(root, f)
                    rel = os.path.relpath(rgb_path, rgb_root)
                    depth_path = os.path.join(
                        depth_root, os.path.splitext(rel)[0] + ".png"
                    )
                    if os.path.exists(depth_path):
                        self.samples.append((rgb_path, depth_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No RGB-D pairs found in {rgb_root} / {depth_root}")

        print(f"[DATA] Found {len(self.samples)} RGB-D pairs")

    def __len__(self):
        return len(self.samples)

    def _load_depth_geometric(self, depth_path):
        """Load depth and compute 3-channel geometric features."""
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth is None:
            depth = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # Resize
        depth = cv2.resize(depth, (self.img_size, self.img_size))

        # Normalize depth to [0, 1]
        depth_norm = depth.astype(np.float32) / 255.0

        # Compute geometric features
        edges = extract_depth_edges(depth)
        structure = extract_depth_structure(depth)

        # Resize features to match
        edges = cv2.resize(edges, (self.img_size, self.img_size))
        structure = cv2.resize(structure, (self.img_size, self.img_size))

        # Stack: [3, H, W]
        depth_multi = np.stack([depth_norm, edges, structure], axis=0)
        return torch.from_numpy(depth_multi).float()

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        # Load RGB
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Use same random seed for geometric consistency between views
        seed = np.random.randint(2147483647)

        # RGB view 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        rgb_v1 = self.rgb_transform(rgb)

        # RGB view 2 (different augmentation)
        rgb_v2 = self.rgb_transform(rgb)

        # Depth (geometric features, no random augmentation for contrastive)
        depth_v1 = self._load_depth_geometric(depth_path)
        depth_v2 = depth_v1.clone()  # Same depth for both views

        return rgb_v1, depth_v1, rgb_v2, depth_v2


# ==================================================
# MODELS
# ==================================================
class DINOv3EncoderSSL(nn.Module):
    """DINOv3 ViT-L/16 for SSL (unfrozen, with gradient checkpointing)."""

    def __init__(self, checkpoint_path, use_grad_checkpoint=True):
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint

        try:
            import timm
        except ImportError:
            raise ImportError("pip install timm")

        self.model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            num_classes=0,
            global_pool='',
            img_size=224
        )

        # Load DINOv3 weights
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get('model', ckpt.get('state_dict',
                                  ckpt.get('teacher', ckpt)))
        else:
            state_dict = ckpt

        cleaned = {k.replace('module.', '').replace('backbone.', ''): v
                   for k, v in state_dict.items()}
        msg = self.model.load_state_dict(cleaned, strict=False)
        print(f"[DINOv3] Loaded weights (missing={len(msg.missing_keys)}, "
              f"unexpected={len(msg.unexpected_keys)})")

        # All parameters unfrozen for SSL
        for param in self.model.parameters():
            param.requires_grad = True

        self.feat_dim = self.model.embed_dim  # 1024
        print(f"[DINOv3] Feature dim: {self.feat_dim}, grad_checkpoint: {use_grad_checkpoint}")

    def forward(self, x):
        if self.use_grad_checkpoint and self.training:
            # Patch embedding (no checkpointing needed, small)
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
            x = self.model.patch_drop(x)
            x = self.model.norm_pre(x)

            # Gradient checkpointing on transformer blocks
            for block in self.model.blocks:
                x = grad_checkpoint(block, x, use_reentrant=False)

            x = self.model.norm(x)
            return x[:, 0]  # CLS token
        else:
            x = self.model.forward_features(x)
            return x[:, 0]  # CLS token


class DepthGeometricEncoder(nn.Module):
    """Small CNN for depth geometric features (from rgbd_curriculum_v2)."""

    def __init__(self, output_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(self, depth_multi):
        """depth_multi: [B, 3, 224, 224] (depth, edges, structure)."""
        x = self.encoder(depth_multi)
        x = x.view(x.size(0), -1)
        return self.projection(x)


class DINOv3RGBDSSL(nn.Module):
    """
    DINOv3 + Depth SSL model with late fusion and contrastive learning.

    Architecture:
      DINOv3 ViT-L/16 (1024-dim) + DepthGeometricEncoder (128-dim)
      -> Concatenate (1152-dim) -> SimCLR Projection Head -> 128-dim
    """

    def __init__(self, dinov3_checkpoint, modality_dropout=0.2,
                 use_grad_checkpoint=True):
        super().__init__()
        self.modality_dropout = modality_dropout

        # RGB encoder: DINOv3 ViT-L/16 (unfrozen)
        self.rgb_encoder = DINOv3EncoderSSL(
            dinov3_checkpoint, use_grad_checkpoint=use_grad_checkpoint
        )
        rgb_dim = self.rgb_encoder.feat_dim  # 1024

        # Depth encoder: Small geometric CNN
        self.depth_encoder = DepthGeometricEncoder(output_dim=128, dropout=0.3)
        depth_dim = self.depth_encoder.output_dim  # 128

        # Fusion + Projection
        fused_dim = rgb_dim + depth_dim  # 1152
        self.projection_head = SimCLRProjectionHead(fused_dim, 512, 128)

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] Total params: {total_params/1e6:.1f}M, "
              f"Trainable: {trainable/1e6:.1f}M")
        print(f"[MODEL] Fusion dim: {fused_dim}, Output dim: 128")

    def forward(self, rgb, depth_multi):
        """
        Args:
            rgb: [B, 3, 224, 224]
            depth_multi: [B, 3, 224, 224] (depth, edges, structure)
        Returns:
            [B, 128] projected features
        """
        # Modality dropout during training
        if self.training and self.modality_dropout > 0:
            if torch.rand(1).item() < self.modality_dropout:
                if torch.rand(1).item() < 0.5:
                    depth_multi = torch.zeros_like(depth_multi)
                else:
                    rgb = torch.zeros_like(rgb)

        rgb_feat = self.rgb_encoder(rgb)           # [B, 1024]
        depth_feat = self.depth_encoder(depth_multi)  # [B, 128]
        fused = torch.cat([rgb_feat, depth_feat], dim=1)  # [B, 1152]
        return self.projection_head(fused)  # [B, 128]

    def get_dinov3_state_dict(self):
        """Extract only the DINOv3 encoder weights for curriculum loading."""
        return self.rgb_encoder.model.state_dict()

    def get_depth_encoder_state_dict(self):
        """Extract depth encoder weights."""
        return self.depth_encoder.state_dict()


# ==================================================
# UTILS
# ==================================================
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==================================================
# TRAINING
# ==================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

    # RGB transform (SimCLR-style augmentation)
    rgb_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset
    dataset = DINOv3RGBDDataset(RGB_ROOT, DEPTH_ROOT, rgb_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * EPOCHS

    # Model
    model = DINOv3RGBDSSL(
        dinov3_checkpoint=DINOV3_CHECKPOINT,
        modality_dropout=MODALITY_DROPOUT,
        use_grad_checkpoint=GRADIENT_CHECKPOINTING,
    ).to(device)

    # Optimizer with differential learning rates
    param_groups = [
        {'params': model.rgb_encoder.parameters(), 'lr': DINOV3_LR,
         'name': 'dinov3_encoder'},
        {'params': model.depth_encoder.parameters(), 'lr': DEPTH_LR,
         'name': 'depth_encoder'},
        {'params': model.projection_head.parameters(), 'lr': PROJECTION_LR,
         'name': 'projection_head'},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    print("[OPT] Learning rates:")
    for g in param_groups:
        print(f"  {g['name']}: {g['lr']}")

    # Loss
    criterion = NTXentLoss()

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )

    # Resume support
    start_epoch = 0
    latest_ckpt = os.path.join(CKPT_DIR, "latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        print(f"[RESUME] From epoch {start_epoch}")

    print("=" * 80)
    print("SSL PRETRAINING: DINOv3 ViT-L/16 + Depth Geometric Encoder")
    print("=" * 80)
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = "
          f"{BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f"Mixed precision: {MIXED_PRECISION} | Grad checkpoint: {GRADIENT_CHECKPOINTING}")
    print("=" * 80)

    global_start = time.time()
    best_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        running_loss = 0.0
        model.train()
        optimizer.zero_grad()

        for step, (rgb_v1, depth_v1, rgb_v2, depth_v2) in enumerate(dataloader):
            rgb_v1 = rgb_v1.to(device, non_blocking=True)
            depth_v1 = depth_v1.to(device, non_blocking=True)
            rgb_v2 = rgb_v2.to(device, non_blocking=True)
            depth_v2 = depth_v2.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                z1 = model(rgb_v1, depth_v1)
                z2 = model(rgb_v2, depth_v2)
                loss = criterion(z1, z2)
                loss = loss / GRADIENT_ACCUMULATION

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * GRADIENT_ACCUMULATION

            if step % LOG_INTERVAL == 0:
                global_step = epoch * steps_per_epoch + step + 1
                elapsed = time.time() - global_start
                eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                print(
                    f"[Epoch {epoch+1}/{EPOCHS}] "
                    f"Step {step}/{steps_per_epoch} "
                    f"Loss {loss.item() * GRADIENT_ACCUMULATION:.4f} "
                    f"LR {optimizer.param_groups[0]['lr']:.2e} "
                    f"ETA {format_time(eta)}"
                )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / steps_per_epoch

        print(
            f"[Epoch {epoch+1}] Avg Loss {avg_loss:.4f} | "
            f"Time {format_time(epoch_time)} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save checkpoint
        ckpt_data = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": avg_loss,
        }
        torch.save(ckpt_data, latest_ckpt)

        # Save periodic + best
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data, os.path.join(CKPT_DIR, f"epoch_{epoch+1}.pth"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, os.path.join(CKPT_DIR, "best.pth"))
            print(f"  -> New best loss: {best_loss:.4f}")

    total_time = time.time() - global_start

    # Save DINOv3 backbone for curriculum loading
    dinov3_backbone_path = os.path.join(OUT_DIR, "ssl_dinov3_rgbd_backbone.pth")
    torch.save(model.get_dinov3_state_dict(), dinov3_backbone_path)

    # Save depth encoder separately
    depth_encoder_path = os.path.join(OUT_DIR, "ssl_depth_encoder.pth")
    torch.save(model.get_depth_encoder_state_dict(), depth_encoder_path)

    # Save full model
    full_model_path = os.path.join(OUT_DIR, "full_model.pth")
    torch.save(model.state_dict(), full_model_path)

    print("=" * 80)
    print(f"Training complete in {format_time(total_time)}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"DINOv3 backbone: {dinov3_backbone_path}")
    print(f"Depth encoder:   {depth_encoder_path}")
    print(f"Full model:      {full_model_path}")
    print("=" * 80)
