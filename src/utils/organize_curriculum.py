"""
Usage:
    python src/utils/organize_curriculum.py \
        --views_dir data/curriculum_views/ \
        --output_dir data/curriculum/ \
        --val_fraction 0.10 \
        --test_fraction 0.10 \
        --seed 42
"""

import argparse, shutil, random
from collections import defaultdict
from pathlib import Path

# Must match generate_views.py
YAW_DIFFICULTY = {
    "easy":   {67.5, 90.0, 112.5, 247.5, 270.0, 292.5},
    "medium": {22.5, 45.0, 135.0, 225.0, 315.0, 337.5},
    "hard":   {0.0, 157.5, 180.0, 202.5},
}


def yaw_from_filename(name: str) -> float | None:
    """Parse yaw from filenames like  9_yaw_135p00_pitch_000p00.jpg"""
    try:
        idx = name.index("_yaw_")
        part = name[idx + 5:].split("_pitch_")[0]   # e.g. "135p00"
        deg_str, dec_str = part.split("p")
        return float(deg_str) + float(dec_str) / 100.0
    except (ValueError, IndexError):
        return None


def get_difficulty(yaw: float) -> str:
    for diff, angles in YAW_DIFFICULTY.items():
        if yaw in angles:
            return diff
    return "medium"


def assign_splits(pano_ids: list, val_frac: float, test_frac: float,
                  rng: random.Random) -> dict[str, str]:
    """Return {pano_id: split} for a single city's panoramas."""
    ids = sorted(pano_ids, key=lambda x: int(x) if x.isdigit() else x)
    rng.shuffle(ids)
    n      = len(ids)
    n_test = max(1, round(n * test_frac))
    n_val  = max(1, round(n * val_frac))
    labels = (["test"] * n_test) + (["val"] * n_val) + (["train"] * (n - n_test - n_val))
    return dict(zip(ids, labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views_dir",     required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--val_fraction",  type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    views_root  = Path(args.views_dir)
    output_root = Path(args.output_dir)
    rng = random.Random(args.seed)

    # Collect all view images grouped by (country, city, pano_id)
    by_pano: dict[tuple, list[Path]] = {}
    for img in sorted(views_root.glob("**/*.jpg")):
        rel = img.relative_to(views_root)
        parts = rel.parts          # (country, city, filename)
        if len(parts) != 3:
            continue
        country, city, fname = parts
        pano_id = fname.split("_yaw_")[0]
        key = (country, city, pano_id)
        by_pano.setdefault(key, []).append(img)

    # Assign train/val/test per city (panorama-level split, no leakage)
    panos_by_city: dict[tuple, list] = defaultdict(list)
    for (country, city, pano_id) in by_pano:
        panos_by_city[(country, city)].append(pano_id)

    pano_split: dict[tuple, str] = {}
    for (country, city), pano_ids in sorted(panos_by_city.items()):
        city_splits = assign_splits(pano_ids, args.val_fraction,
                                    args.test_fraction, rng)
        for pid, split in city_splits.items():
            pano_split[(country, city, pid)] = split

    # Copy each view to the correct location
    copied = 0
    for (country, city, pano_id), views in by_pano.items():
        split = pano_split[(country, city, pano_id)]
        for img in views:
            yaw = yaw_from_filename(img.name)
            difficulty = get_difficulty(yaw) if yaw is not None else "medium"
            dest = output_root / split / difficulty / country / city / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(img, dest)
                copied += 1

    counts = defaultdict(int)
    for v in pano_split.values():
        counts[v] += 1
    print(f"Organized {copied} view images → {output_root}")
    print(f"Panoramas: {counts['train']} train | {counts['val']} val | {counts['test']} test")


if __name__ == "__main__":
    main()
