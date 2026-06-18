"""
Annotation pipeline runner — runs all five label stages in sequence.

Usage:
    python src/annotation/annotate.py           # full pipeline
    python src/annotation/annotate.py --from 3  # resume from stage 3

Stages:
    0 — Analyze existing labels (report + plots)
    1 — SAM3 building detection
    2 — Multi-source classification (CLIP + OCR + geometry)
    3 — Label fusion + confidence filtering
    4 — DINOv3 prototype consistency check
    5 — Rare class mining + final label assembly
    6 — Annotation visualization (40 sample images)
"""

import argparse, subprocess, sys
from pathlib import Path

STAGES = [
    (1, "SAM3 building detection",
     ["python", "src/annotation/sam3_detect.py",
      "--images", "data/curriculum/",
      "--depth",  "data/curriculum_depths/",
      "--output", "data/annotations/sam3_detections.csv",
      "--resume"]),
    (2, "Multi-source classification (CLIP + OCR + geometry)",
     ["python", "src/annotation/multi_source_classify.py",
      "--detections", "data/annotations/sam3_detections.csv",
      "--output",     "data/annotations/multi_source_labels.csv",
      "--resume"]),
    (3, "Label fusion + confidence weighting",
     ["python", "src/annotation/fuse_labels.py",
      "--input",  "data/annotations/multi_source_labels.csv",
      "--output", "data/annotations/fused_labels.csv"]),
    (4, "DINOv3 prototype consistency check",
     ["python", "src/annotation/prototype_consistency.py",
      "--labels",  "data/annotations/fused_labels.csv",
      "--output",  "data/annotations/prototype_checked_labels.csv",
      "--weights", "final_models/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
      "--k", "10"]),
    (5, "Rare class mining + final label assembly",
     ["python", "src/annotation/mine_and_assemble.py",
      "--checked",  "data/annotations/prototype_checked_labels.csv",
      "--original", "data/curriculum/",
      "--output",   "data/annotations/final_labels.csv",
      "--target",   "1500"]),
]


def run_stage(stage_num: int, name: str, cmd: list[str]) -> bool:
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num}: {name}")
    print(f"{'='*60}")
    print(f"CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Stage {stage_num} failed (return code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",  dest="from_stage", type=int, default=1,
                        help="Start from this stage number (1-5)")
    parser.add_argument("--only",  type=int, default=None,
                        help="Run only this stage number")
    args = parser.parse_args()

    stages_to_run = STAGES
    if args.only is not None:
        stages_to_run = [s for s in STAGES if s[0] == args.only]
    else:
        stages_to_run = [s for s in STAGES if s[0] >= args.from_stage]

    print("GeoBuild-SSL: Annotation Pipeline")
    print(f"Running stages: {[s[0] for s in stages_to_run]}")

    for stage_num, name, cmd in stages_to_run:
        ok = run_stage(stage_num, name, cmd)
        if not ok:
            print(f"\nStopped at stage {stage_num}. Fix errors and re-run with:")
            print(f"  python src/annotation/annotate.py --from {stage_num}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("Annotation pipeline COMPLETE")
    print(f"{'='*60}")
    print("Final labels: data/annotations/final_labels.csv")


if __name__ == "__main__":
    main()
