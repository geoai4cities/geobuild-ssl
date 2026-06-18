import numpy as np
import torch
import cv2
from PIL import Image


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2 Large."""

    def __init__(self, device, model_id="depth-anything/Depth-Anything-V2-Large-hf"):
        self.device = device
        self.model = None
        self.processor = None
        self._load(model_id)

    def _load(self, model_id):
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.model = self.model.to(self.device).eval()
            print(f"  [OK] Depth Anything V2 loaded ({model_id})")
        except Exception as e:
            print(f"  [WARN] Depth model unavailable: {e}")
            print("         Will use gradient-based fallback depth")

    @torch.no_grad()
    def estimate(self, image_pil):
        """
        Estimate depth from a PIL image.

        Returns:
            metric_depth: np.ndarray (H, W) float32, values in approximate
                          metres [3, 35] calibrated to match training data.
        """
        w, h = image_pil.size

        if self.model is not None and self.processor is not None:
            inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
            raw = self.model(**inputs).predicted_depth          # (1, Hm, Wm)
            raw = torch.nn.functional.interpolate(
                raw.unsqueeze(1), size=(h, w),
                mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()                           # (H, W)
            return self._raw_to_metric(raw)

        # Fallback: linear gradient (matches rough training distribution)
        y = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)
        return (15.0 + (y / h) * 15.0).astype(np.float32)     # 15m top → 30m bottom

    @staticmethod
    def _raw_to_metric(raw_depth):
        """
        Convert DA V2 raw inverse-depth to metric depth [3, 35] m.

        DA V2 outputs inverse depth (high = near, low = far).  We normalise
        with robust percentiles and linearly map to [3, 35]m which matches
        the training annotation distribution (mean_depth_m 17-32 m).
        """
        raw = np.clip(raw_depth, 0, None)

        p_hi = np.percentile(raw, 99.5)     # near reference
        p_lo = np.percentile(raw, 0.5)      # far reference
        raw_norm = np.clip((raw - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)

        # Invert: 0 = nearest, 1 = farthest
        inv = 1.0 - raw_norm

        # Linear map to metres
        metric = 3.0 + inv * 32.0           # near → 3 m, far → 35 m
        return metric.astype(np.float32)
