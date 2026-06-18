"""
Usage:
    python src/utils/generate_views.py \
        --input_dir data/raw/panoramas/ \
        --output_dir data/curriculum_views/ \
        --stage 2 --size 512

"""

import argparse, math, os, glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Difficulty assignment by yaw angle (used in curriculum split)
YAW_DIFFICULTY = {
    "easy":   {67.5, 90.0, 112.5, 247.5, 270.0, 292.5},
    "medium": {22.5, 45.0, 135.0, 225.0, 315.0, 337.5},
    "hard":   {0.0,  157.5, 180.0, 202.5},
}

STAGE_YAW_STEPS = {0: 90.0, 1: 45.0, 2: 22.5}


def equirectangular_to_perspective(pano: np.ndarray, yaw_deg: float,
                                    pitch_deg: float, fov_deg: float,
                                    out_w: int, out_h: int) -> np.ndarray:
    """Convert equirectangular panorama to perspective view.

    Args:
        pano:      H×W×3 equirectangular image
        yaw_deg:   horizontal rotation (0=front, 90=right, 180=back, 270=left)
        pitch_deg: vertical tilt (0=level, positive=up)
        fov_deg:   horizontal field of view in degrees
        out_w/h:   output image dimensions

    Returns:
        out_h×out_w×3 perspective image
    """
    H, W = pano.shape[:2]
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    fov   = math.radians(fov_deg)

    f = (out_w / 2) / math.tan(fov / 2)

    # Build pixel grid
    u = np.arange(out_w) - out_w / 2
    v = np.arange(out_h) - out_h / 2
    uu, vv = np.meshgrid(u, v)

    # Ray directions in camera space
    x = uu
    y = -vv
    z = np.full_like(uu, f)

    # Normalise
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm

    # Rotate by pitch around X axis
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    y2 =  cos_p * y - sin_p * z
    z2 =  sin_p * y + cos_p * z
    y, z = y2, z2

    # Rotate by yaw around Y axis
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    x2 =  cos_y * x + sin_y * z
    z2 = -sin_y * x + cos_y * z
    x, z = x2, z2

    # Convert to spherical
    lon = np.arctan2(x, z)                  # [-π, π]
    lat = np.arcsin(np.clip(y, -1, 1))      # [-π/2, π/2]

    # Map to panorama pixel coordinates
    px = ((lon / (2 * math.pi)) + 0.5) * W
    py = (0.5 - lat / math.pi) * H

    map_x = px.astype(np.float32)
    map_y = py.astype(np.float32)

    return cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def get_yaw_angles(stage: int) -> list:
    step = STAGE_YAW_STEPS[stage]
    n    = int(360 / step)
    return [round(i * step, 1) for i in range(n)]


def yaw_to_filename_part(yaw: float, pitch: float) -> str:
    """Format as yaw_090p00_pitch_000p00"""
    def fmt(v):
        deg = int(v)
        dec = round((abs(v) - abs(deg)) * 100)
        return f"{abs(deg):03d}p{dec:02d}"
    return f"yaw_{fmt(yaw)}_pitch_{fmt(pitch)}"


def get_difficulty(yaw: float) -> str:
    for diff, angles in YAW_DIFFICULTY.items():
        if yaw in angles:
            return diff
    return "medium"  # fallback


def process_panorama(img_path: Path, output_dir: Path, stage: int,
                     fov: int, size: int, pitch: float = 0.0) -> list:
    """Extract all perspective views from one panorama. Returns list of output paths."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read: {img_path}")
        return []

    stem   = img_path.stem
    yaws   = get_yaw_angles(stage)
    written = []

    for yaw in yaws:
        part    = yaw_to_filename_part(yaw, pitch)
        out_name = f"{stem}_{part}.jpg"
        out_path = output_dir / out_name

        if out_path.exists():
            written.append(out_path)
            continue

        view = equirectangular_to_perspective(img, yaw, pitch, fov, size, size)
        cv2.imwrite(str(out_path), view, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(out_path)

    return written


def main():
    parser = argparse.ArgumentParser(description="Generate perspective views from panoramas")
    parser.add_argument("--input_dir",  required=True, help="Root dir containing panoramas (country/city/*.jpg)")
    parser.add_argument("--output_dir", required=True, help="Output root dir (mirrors input structure)")
    parser.add_argument("--stage", type=int, default=2, choices=[0, 1, 2],
                        help="View density: 0=4views, 1=8views, 2=16views")
    parser.add_argument("--fov",   type=int, default=90,  help="Horizontal FOV in degrees")
    parser.add_argument("--size",  type=int, default=512, help="Output image size (square)")
    parser.add_argument("--pitch", type=float, default=0.0, help="Vertical tilt in degrees")
    args = parser.parse_args()

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    # Find all panorama images recursively
    patterns = ["**/*.jpg", "**/*.JPG", "**/*.png", "**/*.PNG"]
    all_panos = []
    for pat in patterns:
        all_panos.extend(input_root.glob(pat))
    all_panos = sorted(set(all_panos))

    if not all_panos:
        print(f"No images found in {input_root}")
        return

    print(f"Found {len(all_panos)} panoramas → generating {len(get_yaw_angles(args.stage))} views each")
    print(f"Stage {args.stage}: {get_yaw_angles(args.stage)}")

    total_written = 0
    for pano_path in tqdm(all_panos, desc="Panoramas"):
        # Mirror directory structure under output root
        rel_parent = pano_path.parent.relative_to(input_root)
        out_dir    = output_root / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)

        written = process_panorama(pano_path, out_dir, args.stage,
                                    args.fov, args.size, args.pitch)
        total_written += len(written)

    print(f"\nDone. Total view images: {total_written}")
    yaws = get_yaw_angles(args.stage)
    print(f"Difficulty breakdown per view:")
    for diff, angles in YAW_DIFFICULTY.items():
        count = sum(1 for y in yaws if y in angles)
        print(f"  {diff}: {count} views per panorama → {count * len(all_panos)} total")


if __name__ == "__main__":
    main()
