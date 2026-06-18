#!/usr/bin/env python3
"""

Usage:
    python src/detection/train_building_segmentation.py
"""

import os, sys, csv, time, random, glob, math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ssl"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DINOV3_CKPT    = str(PROJECT_ROOT / "final_models" / "ssl_model" / "dinov3_rgbd_ssl.pth")
MASK_DIR       = str(PROJECT_ROOT / "data" / "annotations" / "sam3_building_masks")
CURRICULUM     = str(PROJECT_ROOT / "data" / "curriculum")
DEPTH_ROOT     = str(PROJECT_ROOT / "data" / "curriculum_depths")
OUT_DIR        = str(PROJECT_ROOT / "out" / "building_segmentation")
CKPT_DIR       = os.path.join(OUT_DIR, "checkpoints")

IMG_SIZE       = 512
FPN_CH         = 256
EPOCHS         = 200
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
LR_BACKBONE    = 2e-6
LR_DECODER     = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4
LOG_INTERVAL   = 100
EVAL_INTERVAL  = 10
FREEZE_UNTIL   = 12

os.makedirs(CKPT_DIR, exist_ok=True)
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════

class DINOv3BackboneFT(nn.Module):
    EXTRACT_BLOCKS = [8, 16, 23]
    def __init__(self, ckpt_path, freeze_until=12):
        super().__init__()
        import timm
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False,
                                       num_classes=0, global_pool="", img_size=IMG_SIZE)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cleaned = {k.replace("module.","").replace("backbone.",""): v for k,v in ckpt.items()}
        if "pos_embed" in cleaned:
            pe = cleaned["pos_embed"]; ns = pe.shape[1]-1; nd = (IMG_SIZE//16)**2
            if ns != nd:
                cp = pe[:,:1,:]; pp = pe[:,1:,:].reshape(1,int(ns**0.5),int(ns**0.5),-1).permute(0,3,1,2)
                pp = F.interpolate(pp, size=(IMG_SIZE//16,IMG_SIZE//16), mode="bicubic", align_corners=False)
                cleaned["pos_embed"] = torch.cat([cp, pp.permute(0,2,3,1).reshape(1,nd,-1)], dim=1)
        self.model.load_state_dict(cleaned, strict=False)
        for name, p in self.model.named_parameters():
            freeze = True
            for b in range(freeze_until, 24):
                if f"blocks.{b}." in name: freeze = False; break
            if "norm." in name and "blocks." not in name: freeze = False
            p.requires_grad_(not freeze)
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[BACKBONE] Blocks {freeze_until}-23 trainable ({n/1e6:.1f}M)")
        self.feat_dim = self.model.embed_dim

    def forward(self, x):
        B = x.shape[0]; HP = WP = IMG_SIZE // 16
        x = self.model.patch_embed(x); x = self.model._pos_embed(x)
        x = self.model.patch_drop(x); x = self.model.norm_pre(x)
        feats = []
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i in self.EXTRACT_BLOCKS:
                feats.append(x[:,1:].permute(0,2,1).reshape(B, self.feat_dim, HP, WP))
        return feats


class DepthFiLM(nn.Module):
    def __init__(self, out_ch=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.gamma = nn.Linear(128, out_ch)
        self.beta = nn.Linear(128, out_ch)
        nn.init.ones_(self.gamma.weight[:out_ch//2]); nn.init.zeros_(self.gamma.weight[out_ch//2:])
        nn.init.zeros_(self.gamma.bias); nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, d):
        f = self.enc(d).flatten(1)
        return self.gamma(f).unsqueeze(-1).unsqueeze(-1), self.beta(f).unsqueeze(-1).unsqueeze(-1)


class AttentionRefine(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(ch,ch//4), nn.ReLU(), nn.Linear(ch//4,ch), nn.Sigmoid())
    def forward(self, x): return x * self.attn(x).unsqueeze(-1).unsqueeze(-1)


class MultiScaleDecoder(nn.Module):
    """Dual-head decoder: predicts building mask + boundary mask.

    Building head: which pixels are buildings?
    Boundary head: which pixels are boundaries between adjacent buildings?

    At inference: building_mask - boundary_mask → separated instances
    """
    def __init__(self, ch=256):
        super().__init__()
        self.r3 = nn.Sequential(nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),AttentionRefine(ch))
        self.r4 = nn.Sequential(nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),AttentionRefine(ch))
        self.r5 = nn.Sequential(nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),AttentionRefine(ch))
        self.fuse = nn.Sequential(nn.Conv2d(ch*3,ch,1),nn.BatchNorm2d(ch),nn.ReLU(),
                                  nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU())
        # Shared upsampling path
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ch,128,4,stride=2,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128,64,4,stride=2,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64,32,4,stride=2,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.up4 = nn.Sequential(nn.ConvTranspose2d(32,16,4,stride=2,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        # Building prediction heads (multi-scale)
        self.pred_512 = nn.Conv2d(16,1,1)
        self.pred_256 = nn.Conv2d(32,1,1)
        self.pred_128 = nn.Conv2d(64,1,1)
        # Boundary prediction head (512 only — boundaries are thin, need full res)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(16,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16,1,1))

    def forward(self, p3, p4, p5):
        p3=self.r3(p3); p4=self.r4(p4); p5=self.r5(p5)
        p4u = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5u = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([p3,p4u,p5u], dim=1))
        x1=self.up1(f); x2=self.up2(x1); x3=self.up3(x2); x4=self.up4(x3)
        return self.pred_512(x4), self.pred_256(x3), self.pred_128(x2), self.boundary_head(x4)


class BuildingSegModel(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.backbone = DINOv3BackboneFT(ckpt, freeze_until=FREEZE_UNTIL)
        self.depth_film = DepthFiLM(out_ch=FPN_CH)
        self.lat3=nn.Conv2d(1024,FPN_CH,1); self.lat4=nn.Conv2d(1024,FPN_CH,1); self.lat5=nn.Conv2d(1024,FPN_CH,1)
        self.smooth3 = nn.Sequential(nn.Conv2d(FPN_CH,FPN_CH,3,padding=1),nn.BatchNorm2d(FPN_CH),nn.ReLU())
        self.down4 = nn.Conv2d(FPN_CH,FPN_CH,3,stride=2,padding=1)
        self.down5 = nn.Conv2d(FPN_CH,FPN_CH,3,stride=2,padding=1)
        self.decoder = MultiScaleDecoder(FPN_CH)
        bb=sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        dec=sum(p.numel() for p in self.parameters() if p.requires_grad)-bb
        print(f"[MODEL] Backbone: {bb/1e6:.1f}M | Decoder: {dec/1e6:.1f}M")

    def forward(self, rgb, depth_3ch):
        f8,f16,f23 = self.backbone(rgb)
        l5=self.lat5(f23);l4=self.lat4(f16);l3=self.lat3(f8)
        p5=l5; p4=l4+F.interpolate(p5,size=l4.shape[-2:],mode="nearest")
        p3=l3+F.interpolate(p4,size=l3.shape[-2:],mode="nearest")
        p3=self.smooth3(p3); p4=self.down4(p3); p5=self.down5(p4)
        g,b = self.depth_film(depth_3ch)
        p3=g*p3+b; p4=g*p4+b; p5=g*p5+b
        return self.decoder(p3,p4,p5)  # returns (bld_512, bld_256, bld_128, boundary_512)


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

class BuildingSegDataset(Dataset):
    MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
    def __init__(self, samples, augment=True):
        self.samples=samples; self.augment=augment
        self.norm=transforms.Normalize(self.MEAN, self.STD)

    def _load_depth_3ch(self, dp):
        try:
            d = np.load(dp).astype(np.float32) if dp.endswith('.npy') else \
                cv2.imread(dp,cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
            if d.ndim==3: d=d[:,:,0]
        except: d=np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.float32)
        di=Image.fromarray(np.clip(d*255,0,255).astype(np.uint8)).resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR)
        dn=np.array(di,dtype=np.float32)/255.0
        sx=cv2.Sobel(dn,cv2.CV_32F,1,0,ksize=3);sy=cv2.Sobel(dn,cv2.CV_32F,0,1,ksize=3)
        e=np.sqrt(sx**2+sy**2);e/=(e.max()+1e-6)
        l=cv2.Laplacian(dn,cv2.CV_32F,ksize=3);r=l.max()-l.min()
        if r>0: l=(l-l.min())/r
        return torch.from_numpy(np.stack([dn,e,l],axis=0))

    def __len__(self): return len(self.samples)
    @staticmethod
    def _make_boundary(instance_mask):
        """Generate boundary mask from instance mask.
        Boundary = pixels where adjacent pixels belong to different building instances.
        """
        h, w = instance_mask.shape
        boundary = np.zeros((h, w), dtype=np.float32)
        # Check 4-connected neighbors
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            shifted = np.roll(np.roll(instance_mask, dy, axis=0), dx, axis=1)
            # Boundary where: both pixels are buildings (>0) but different instances
            diff = (instance_mask > 0) & (shifted > 0) & (instance_mask != shifted)
            boundary[diff] = 1.0
        # Dilate boundary by 2px to make it easier to learn
        kernel = np.ones((3,3), np.uint8)
        boundary = cv2.dilate(boundary, kernel, iterations=1)
        return boundary

    def __getitem__(self, idx):
        img_path, depth_path, mask_path = self.samples[idx]
        try: img=Image.open(img_path).convert("RGB")
        except: img=Image.new("RGB",(IMG_SIZE,IMG_SIZE))
        try: instance_mask=np.load(mask_path).astype(np.float32)
        except: instance_mask=np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.float32)
        img=img.resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR)
        instance_mask=cv2.resize(instance_mask,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if random.random()>0.5:
                img=img.transpose(Image.FLIP_LEFT_RIGHT); instance_mask=np.fliplr(instance_mask).copy()
            if random.random()>0.3:
                img=transforms.ColorJitter(0.4,0.4,0.3,0.1)(img)
            if random.random()>0.5:
                sc=random.uniform(0.8,1.2); ns=int(IMG_SIZE*sc)
                img=img.resize((ns,ns),Image.BILINEAR); instance_mask=cv2.resize(instance_mask,(ns,ns),interpolation=cv2.INTER_NEAREST)
                if ns>IMG_SIZE:
                    off=random.randint(0,ns-IMG_SIZE)
                    img=img.crop((off,off,off+IMG_SIZE,off+IMG_SIZE)); instance_mask=instance_mask[off:off+IMG_SIZE,off:off+IMG_SIZE]
                else:
                    img=img.resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR)
                    instance_mask=cv2.resize(instance_mask,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_NEAREST)
        # Binary building mask (any instance > 0)
        building_mask = (instance_mask > 0).astype(np.float32)
        # Boundary mask (edges between different instances)
        boundary_mask = self._make_boundary(instance_mask)
        return (self.norm(transforms.ToTensor()(img)),
                self._load_depth_3ch(depth_path),
                torch.from_numpy(building_mask).unsqueeze(0),
                torch.from_numpy(boundary_mask).unsqueeze(0))


def build_datasets():
    samples=[]
    for p in sorted(glob.glob(os.path.join(CURRICULUM,"**","*.jpg"),recursive=True)):
        rel=os.path.relpath(p,CURRICULUM)
        mp=os.path.join(MASK_DIR,os.path.splitext(rel)[0]+".npy")
        dp=os.path.join(DEPTH_ROOT,os.path.splitext(rel)[0]+".npy")
        if not os.path.exists(mp): continue
        if not os.path.exists(dp): dp=os.path.join(DEPTH_ROOT,os.path.splitext(rel)[0]+".png")
        samples.append((p,dp,mp))
    random.shuffle(samples); n=int(len(samples)*0.9)
    print(f"[DATA] {len(samples)} images | Train: {n} | Val: {len(samples)-n}")
    return BuildingSegDataset(samples[:n],True), BuildingSegDataset(samples[n:],False)


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def dice_loss(pred, target, smooth=1.0):
    pf=torch.sigmoid(pred).view(-1); tf=target.view(-1)
    return 1-(2*(pf*tf).sum()+smooth)/(pf.sum()+tf.sum()+smooth)

def multi_scale_loss(preds, mask_gt, boundary_gt, pos_weight):
    l512,l256,l128,boundary_pred = preds
    m512=mask_gt
    m256=F.interpolate(mask_gt,size=l256.shape[-2:],mode="nearest")
    m128=F.interpolate(mask_gt,size=l128.shape[-2:],mode="nearest")
    # Building loss
    bce512=F.binary_cross_entropy_with_logits(l512,m512,pos_weight=pos_weight)
    bce256=F.binary_cross_entropy_with_logits(l256,m256,pos_weight=pos_weight)
    bce128=F.binary_cross_entropy_with_logits(l128,m128,pos_weight=pos_weight)
    bld_loss = (bce512+dice_loss(l512,m512))+0.5*(bce256+dice_loss(l256,m256))+0.25*(bce128+dice_loss(l128,m128))
    # Boundary loss (heavily weighted — boundaries are rare pixels)
    bnd_weight = torch.tensor([10.0]).to(boundary_pred.device)
    bnd_loss = F.binary_cross_entropy_with_logits(boundary_pred, boundary_gt, pos_weight=bnd_weight)
    bnd_loss += dice_loss(boundary_pred, boundary_gt)
    return bld_loss + 0.5 * bnd_loss

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ti=tu=0; bnd_ti=bnd_tu=0
    for rgb,depth,mask,boundary in loader:
        rgb,depth,mask,boundary=rgb.to(DEVICE),depth.to(DEVICE),mask.to(DEVICE),boundary.to(DEVICE)
        l512,_,_,bnd_pred=model(rgb,depth)
        pred=(torch.sigmoid(l512)>0.5).float()
        ti+=(pred*mask).sum().item(); tu+=((pred+mask)>0).float().sum().item()
        # Boundary IoU
        bnd_p=(torch.sigmoid(bnd_pred)>0.5).float()
        bnd_ti+=(bnd_p*boundary).sum().item(); bnd_tu+=((bnd_p+boundary)>0).float().sum().item()
    bld_iou = ti/(tu+1e-6)
    bnd_iou = bnd_ti/(bnd_tu+1e-6)
    return bld_iou, bnd_iou

def main():
    train_ds,val_ds=build_datasets()
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)

    model=BuildingSegModel(DINOV3_CKPT).to(DEVICE)
    bb_p=[p for n,p in model.named_parameters() if "backbone" in n and p.requires_grad]
    dec_p=[p for n,p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt=torch.optim.AdamW([{"params":bb_p,"lr":LR_BACKBONE},{"params":dec_p,"lr":LR_DECODER}],weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2)
    scaler=torch.amp.GradScaler("cuda")
    pw=torch.tensor([2.0]).to(DEVICE)

    best_iou=0; t0=time.time()
    print(f"\n{'='*60}")
    print(f"BUILDING SEGMENTATION (SAM3 masks)")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    print(f"  LR bb: {LR_BACKBONE} | LR dec: {LR_DECODER}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"{'='*60}")

    for epoch in range(EPOCHS):
        model.train(); running=0; opt.zero_grad()
        for step,(rgb,depth,mask,boundary) in enumerate(train_loader):
            rgb,depth,mask,boundary=rgb.to(DEVICE),depth.to(DEVICE),mask.to(DEVICE),boundary.to(DEVICE)
            with torch.amp.autocast("cuda"):
                loss=multi_scale_loss(model(rgb,depth),mask,boundary,pw)/GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step+1)%GRAD_ACCUM==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            running+=loss.item()*GRAD_ACCUM
            if step%LOG_INTERVAL==0:
                elapsed=time.time()-t0; total=EPOCHS*len(train_loader); done=epoch*len(train_loader)+step+1
                print(f"  [E{epoch+1:03d}/{EPOCHS} S{step:04d}/{len(train_loader)}] loss={loss.item()*GRAD_ACCUM:.4f} ETA={elapsed/done*(total-done)/60:.0f}min")
        sched.step()
        print(f"[Epoch {epoch+1}] loss={running/len(train_loader):.4f}")
        if (epoch+1)%EVAL_INTERVAL==0:
            bld_iou, bnd_iou=evaluate(model,val_loader)
            print(f"  [EVAL] Building IoU={bld_iou:.4f} | Boundary IoU={bnd_iou:.4f}")
            if bld_iou>best_iou:
                best_iou=bld_iou
                torch.save({"epoch":epoch+1,"model_state_dict":model.state_dict(),
                            "best_iou":best_iou,"boundary_iou":bnd_iou},
                           os.path.join(CKPT_DIR,"best.pth"))
                print(f"  -> New best IoU: {best_iou:.4f}")
    print(f"\n{'='*60}\nTRAINING COMPLETE in {(time.time()-t0)/60:.0f} min\n  Best IoU: {best_iou:.4f}\n{'='*60}")

if __name__ == "__main__":
    main()
