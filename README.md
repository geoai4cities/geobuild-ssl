# GeoBuild-SSL

**Building Instance Usage Mapping in Cross-Cultural Urban Scenes from Street View Images using Multimodal Self-Supervised Deep Learning**

GeoBuild-SSL predicts the usage class of an individual building (*residential,
commercial, industrial,* or *institutional*) from a single street-view image,
and attributes the prediction to its OpenStreetMap footprint for GIS export. It
fuses three complementary streams: visual features (DINOv3), a geometric prior
from monocular depth (injected via FiLM), and multilingual signage (PaddleOCR +
a frozen CLIP text encoder), trained with a yaw-based curriculum across six
cities in India (Mumbai, Delhi, Chennai, Bengaluru) and Japan (Tokyo, Osaka).

📄 **Paper:** ACM SIGSPATIAL '26 — [doi:10.1145/3841645.3843286](https://doi.org/10.1145/3841645.3843286)

<p align="center">
  <img src="assets/architecture.png" width="98%"><br>
  <em>The GeoBuild-SSL framework: RGB, depth, and multilingual signage streams
  fused into a four-way usage classifier and exported to GIS.</em>
</p>

## Installation

Requires **Python 3.10**, plus **CUDA 12.1** for GPU training (inference also
runs on CPU/Apple-MPS).

```bash
conda create -n buildingssl python=3.10 -y && conda activate buildingssl

# PyTorch (GPU, CUDA 12.1):
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
# …or CPU / Apple-MPS:
# pip install torch torchvision

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git        # OpenAI CLIP
```

Verify the install:

```bash
python -c "import torch, timm, clip; print('torch', torch.__version__, 'OK')"
```

## Pretrained weights

Weights are distributed through **Google Drive** rather than tracked in git.
After cloning, place them under `final_models/` in the layout below. The
inference code resolves these paths relative to the repo root:

```
final_models/
├── ssl_model/dinov3_rgbd_ssl.pth          # SSL backbone      (≈1.2 GB)
├── detection_model/best.pth               # building detector (≈1.2 GB)
└── classification_model/v5_best.pth       # usage classifier  (≈1.2 GB)
```

The **[Google Drive folder](https://drive.google.com/drive/folders/1AUfRtd3mhPr-d3afXLTwBFB_oL_kp0uM)**
mirrors this layout, so [`gdown`](https://github.com/wkentaro/gdown) drops
everything into place:

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1AUfRtd3mhPr-d3afXLTwBFB_oL_kp0uM" -O final_models/
python -c "import torch; torch.load('final_models/classification_model/v5_best.pth', map_location='cpu'); print('OK')"
```

### Third-party models

These are **fetched automatically on first run** from the Hugging Face Hub — no
manual download needed:

| Model | Hub ID | Used by |
|---|---|---|
| Depth Anything V2 Large | `depth-anything/Depth-Anything-V2-Large-hf` | depth estimation (inference + training) |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | signage text embedding |
| SAM 3 | `facebook/sam3` | annotation pipeline only |

`facebook/sam3` is a gated repo: accept the terms on its model page, then
authenticate once with `huggingface-cli login`.

**DINOv3 ViT-L/16** must be downloaded manually, because Meta distributes it
under a license you have to accept on the
[DINOv3 repo](https://github.com/facebookresearch/dinov3). It is required
**only** to re-run SSL pretraining or the annotation pipeline — plain inference
does not need it. Request `dinov3_vitl16_pretrain_lvd1689m`, then place the file
at exactly this path and filename:

```
final_models/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

If that file is absent, the annotation pipeline falls back to DINOv2 ViT-L/14
via `torch.hub` (`facebookresearch/dinov2`), which gives compatible features.

## Inference

The full pipeline (detecting buildings, estimating depth, reading signage, and
classifying) runs on a single street-level image or a folder of images:

```bash
python -m src.inference.geobuild_inference --image  path/to/street_view.jpg --output results/
python -m src.inference.geobuild_inference --folder path/to/images/         --output results/
```

To attribute predictions to OpenStreetMap footprints and export a usage map:

```bash
python -m src.inference.gis_mapping
```

<p align="center">
  <img src="assets/gis_maps.jpg" width="78%"><br>
  <em>Building-usage maps for Delhi, Tokyo, Chennai, and Osaka: ground truth
  (left), RGB-only (centre), RGB+depth (right).</em>
</p>

## Training

```bash
# 1. RGB-D self-supervised pretraining of the DINOv3 + depth backbone
python -m src.ssl.train_ssl_rgbd_dinov3

# 2. Five-stage automatic annotation pipeline (generates training labels)
python -m src.annotation.annotate

# 3. Usage classifier with depth-map FiLM and the yaw curriculum
#    --stage 0/1/2 = Easy (15 ep) / Easy+Medium (15 ep) / all bands (20 ep)
python -m src.classification.train_v5_curriculum \
    --stage     0 \
    --train_csv data/annotations/clean_train_labels.csv \
    --val_csv   data/annotations/clean_val_labels.csv \
    --out_dir   outputs/checkpoints
```

Configuration defaults are in [`src/configs/`](src/configs/).

## Evaluation

```bash
python -m src.evaluation.clean_retrain_eval
```

Reports weighted-F₁, macro-F₁, per-class F₁ and per-country F₁ on the
panorama-level clean split (`data/annotations/clean_test_labels.csv`), in which
no panorama or building instance is shared between training and test.

| Config. | weighted F₁ | macro F₁ | Res. | Com. | Ind. | Inst. |
|---|---|---|---|---|---|---|
| RGB                | 0.40 | 0.31 | 0.19 | 0.53 | 0.35 | 0.18 |
| RGB + depth        | 0.65 | 0.49 | 0.46 | 0.83 | 0.43 | 0.24 |
| **RGB + FiLM**     | **0.67** | **0.52** | **0.53** | **0.84** | **0.45** | **0.24** |
| RGB + FiLM + text  | 0.66 | 0.50 | 0.49 | 0.83 | 0.43 | 0.26 |

<p align="center">
  <img src="assets/qualitative.png" width="68%"><br>
  <em>Qualitative predictions across the four usage classes and several cities.
  Green borders mark correct predictions, red borders mark failure cases.</em>
</p>

## Data

Released label CSVs are in [`data/annotations/`](data/annotations/), split at the
panorama level so no panorama or building instance is shared across splits:

| Split | Instances | File |
|---|---|---|
| Train | 22,091 | `clean_train_labels.csv` |
| Val   | 2,616  | `clean_val_labels.csv` |
| Test  | 2,899  | `clean_test_labels.csv` |

Derived from 118 Google Street View panoramas, each rendered into 16 perspective
views. The test split is human-verified. The underlying imagery is not
redistributed here.

## Repository structure

```
geobuild-ssl/
├── assets/                        Figures used in this README
├── data/
│   └── annotations/               Released label CSVs
├── final_models/                  Weight layout (download targets; weights not in git)
│   ├── ssl_model/
│   ├── detection_model/
│   ├── classification_model/
│   └── dinov3_weights/
├── src/
│   ├── ssl/                       RGB-D DINOv3 self-supervised pretraining (SimCLR)
│   ├── annotation/                Five-stage automatic label pipeline (annotate.py)
│   ├── classification/            Usage classifier: depth-map FiLM, yaw curriculum
│   ├── depth/                     Monocular depth (Depth Anything V2) utilities
│   ├── detection/                 Building detection / segmentation heads (FPN + FCOS)
│   ├── ocr/                       Multilingual signage OCR (PaddleOCR) + CLIP text
│   ├── evaluation/                Clean-split retraining and per-class / per-country evaluation
│   ├── inference/                 End-to-end pipeline, OSM attribution, GIS map export
│   ├── configs/                   Experiment configs
│   └── utils/                     Shared helpers
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

```bibtex
@inproceedings{kachhi2026geobuildssl,
  author    = {Kachhi, Shailesh and Kumar, Vaibhav and Arya, Deeksha},
  title     = {Building Instance Usage Mapping in Cross-Cultural Urban Scenes
               from Street View Images using Multimodal Self-Supervised Deep Learning},
  booktitle = {The 34th ACM International Conference on Advances in Geographic
               Information Systems (SIGSPATIAL '26)},
  year      = {2026},
  address   = {Riverside, CA, USA},
  publisher = {ACM},
  doi       = {10.1145/3841645.3843286}
}
```

## Acknowledgements

Supported by the PUSHPAK National Mission on Drone Technology
(No. 23 (1)/2024-ESDA), MeitY, Government of India, and ANRF Project
No. CRD/2024/000883.

Built on a number of open models and datasets, whose authors and maintainers we
gratefully acknowledge: DINOv3 and SimCLR (self-supervised visual representation),
Depth Anything V2 (monocular depth), the Segment Anything Model (SAM 3, instance
masks), FCOS (detection), PaddleOCR (multilingual text), and CLIP (shared text
embedding). We also acknowledge OpenStreetMap and its contributors for the
building footprints, and the India Driving Dataset, Mapillary Vistas, and Google
Street View for the underlying street-level imagery. Developed at the IISER
Bhopal GeoAI4Cities Lab.

## License

Released under the [Apache License 2.0](LICENSE). The released model weights are
intended for research use. The annotations are derived from third-party imagery
whose terms govern any redistribution of the underlying images.
