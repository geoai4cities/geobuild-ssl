# Pretrained weights

Weights are **not** committed to git. Place the files here in the layout below.

```
final_models/
├── ssl_model/dinov3_rgbd_ssl.pth                       # ours      [Drive]
├── detection_model/best.pth                            # ours      [Drive]
├── classification_model/v5_best.pth                    # ours      [Drive]
└── dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth   # third-party [Meta]
```

## 1. Third-party backbones — download from the official source

| File | Put it in | Source |
|---|---|---|
| `dinov3_vitl16_pretrain_lvd1689m-*.pth` | `dinov3_weights/` | [Meta DINOv3](https://github.com/facebookresearch/dinov3) · [HF (gated)](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| CLIP `ViT-B/32` | *(auto)* | downloads automatically via `clip.load("ViT-B/32")` |
| Depth Anything V2 Large | *(auto)* | downloads automatically via `transformers` on first run |

## 2. GeoBuild-SSL trained weights — Google Drive

The **[Google Drive folder](https://drive.google.com/drive/folders/1AUfRtd3mhPr-d3afXLTwBFB_oL_kp0uM)**
mirrors this layout — it already contains the `ssl_model/`, `detection_model/`,
and `classification_model/` subfolders, so the files land in the right place
with no renaming:

| File (in Drive) | Lands at | Used by |
|---|---|---|
| `ssl_model/dinov3_rgbd_ssl.pth` | `ssl_model/` | SSL backbone (detection + classifier) |
| `detection_model/best.pth` | `detection_model/` | building detector / segmentation |
| `classification_model/v5_best.pth` | `classification_model/` | usage classifier (depth-map FiLM) |

Pull the whole folder with `gdown` (recreates the subfolders for you):

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1AUfRtd3mhPr-d3afXLTwBFB_oL_kp0uM" -O final_models/
```

After placing the files, verify each loads:

```bash
python -c "import torch; torch.load('final_models/ssl_model/dinov3_rgbd_ssl.pth', map_location='cpu'); print('OK')"
```
