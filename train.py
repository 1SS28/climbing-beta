"""V1: fine-tune YOLO11-seg on ClimbInst.

    uv run python train.py --model yolo11n-seg.pt --epochs 15 --imgsz 640 --name smoke

Ultralytics is AGPL-3.0 and ClimbInst is CC BY-NC-SA 4.0 — both bind only on
commercial release, which would mean retraining on our own data anyway.
"""

import argparse

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolo11s-seg.pt")
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--imgsz", type=int, default=960)  # holds are small and dense; 640 loses the far ones
p.add_argument("--batch", type=int, default=8)
p.add_argument("--name", default="v1")
a = p.parse_args()

results = YOLO(a.model).train(
    data="data/yolo/climbinst.yaml",
    epochs=a.epochs,
    imgsz=a.imgsz,
    batch=a.batch,
    device="mps",
    workers=4,
    project="runs",
    name=a.name,
    exist_ok=True,
    patience=15,
    # A wall shot from the other side is still a wall, but it is never upside
    # down — and hold colour carries route membership, so leave hue nearly alone.
    fliplr=0.5,
    flipud=0.0,
    degrees=5,
    hsv_h=0.015,
    scale=0.4,
)

print(f"\n{a.name} mask mAP50-95: {results.seg.map:.4f}   mAP50: {results.seg.map50:.4f}")
