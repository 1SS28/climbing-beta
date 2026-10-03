"""Run a trained detector over photos and save mask overlays.

    .venv/bin/python predict.py runs/v1_960_best.pt data/mygym

Mainly for the domain-shift check: train.py reports a score on ClimbInst's own
distribution, which is not the gym you climb at.

Inference runs at 1600 rather than the 960 it trained at. Confidence tracks
apparent hold size strongly (0.64 for the smallest quarter against 0.94 for the
largest), so resolution buys real recall: on a steep prow it found 188 holds at
1600 against 164 at 960, for 0.16s more per image.
"""

import sys
from pathlib import Path

from ultralytics import YOLO

weights = sys.argv[1] if len(sys.argv) > 1 else "runs/v1_960_best.pt"
source = sys.argv[2] if len(sys.argv) > 2 else "data/yolo/images/test"
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

results = YOLO(weights).predict(
    source,
    imgsz=1600,
    conf=conf,
    device="mps",
    save=True,
    project="runs",
    name="predict",
    exist_ok=True,
    # ~100 holds per photo, so boxes and labels would cover the masks.
    show_labels=False,
    show_boxes=False,
)

counts = [len(r.masks) if r.masks is not None else 0 for r in results]
print(f"\n{len(results)} images, {sum(counts)} holds ({min(counts)}-{max(counts)} per image)")
print(f"overlays -> {Path(results[0].save_dir)}")
