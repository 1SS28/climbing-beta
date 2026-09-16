"""V1: fine-tune YOLO11-seg on ClimbInst.

    uv run python train.py --model yolo11n-seg.pt --epochs 15 --imgsz 640 --name smoke

See THIRD_PARTY.md for the Ultralytics and ClimbInst terms, which bind only on
distribution.
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
    # Mirroring is fine, flipping is not, and hue carries route membership.
    fliplr=0.5,
    flipud=0.0,
    degrees=5,
    hsv_h=0.015,
    scale=0.4,
)

print(f"\n{a.name} mask mAP50-95: {results.seg.map:.4f}   mAP50: {results.seg.map50:.4f}")
