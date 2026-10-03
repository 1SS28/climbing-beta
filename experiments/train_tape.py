"""Train a tape detector on the pseudo-labels.

    .venv/bin/python train_tape.py [epochs] [imgsz]

Boxes, not masks: a route only needs the tape's colour and roughly where it is,
and a box is enough to sample colour from. Detection also trains faster and
needs fewer examples than segmentation, which matters when the labels are
pseudo-labels and there are only a few hundred.

Small model on purpose. The target is one rigid, tiny, high-contrast object, so
capacity is not the constraint; the label count is.

NOT VALIDATED. Written, never run end to end, and parked deliberately. Only
about 5 percent of ClimbInst is taped, so the labels would come from roughly 60
images, and by construction a model trained on them can only learn tape the
rules already find. It might generalise to dimmer strips; it cannot learn what
the rules systematically miss, which is the actual problem.

The likelier diagnosis is that one wide photograph of a whole wall simply does
not put enough pixels on each strip. Shot from 3 to 4 m instead of 8, a strip
covers several times the area and the existing rules should find most of them.
Test that before training anything.
"""

import sys

from ultralytics import YOLO

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 80
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 1280

results = YOLO("yolo11n.pt").train(
    data="data/tape/tape.yaml",
    epochs=epochs,
    imgsz=imgsz,
    batch=8,
    device="mps",
    workers=4,
    project="runs",
    name="tape",
    exist_ok=True,
    patience=20,
    # Tape is defined by its colour, so hue must stay put; everything else about
    # it is arbitrary, so allow plenty of geometric variation.
    hsv_h=0.005,
    hsv_s=0.5,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.0,
    degrees=12,
    scale=0.6,
    translate=0.2,
)
print(f"\ntape mAP50: {results.box.map50:.4f}  mAP50-95: {results.box.map:.4f}")
