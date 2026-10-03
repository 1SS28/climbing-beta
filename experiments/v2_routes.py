"""V2: group detected holds into routes by colour.

    uv run python v2_routes.py <image> [weights] [out.png]

Each colour group is drawn in its own outline colour, with group sizes printed.
Route membership is never labelled in the data, so this is unsupervised: the
gym encodes the route in the hold colour, and the job is to read it back.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.colour import group, hold_colour

src = sys.argv[1]
weights = sys.argv[2] if len(sys.argv) > 2 else "runs/v1_960_best.pt"
dst = sys.argv[3] if len(sys.argv) > 3 else "out/v2_routes.png"
threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0

from ultralytics import YOLO  # noqa: E402  (slow import, keep it after arg parsing)

result = YOLO(weights).predict(src, imgsz=1600, conf=0.15, device="mps", verbose=False)[0]
img = Image.open(src).convert("RGB")
rgb = np.array(img)

colours, kept = [], []
# masks.xy, not masks.data: .data is in the model's letterboxed frame, so
# rescaling it to the image size shifts every mask off its hold and samples the
# wall instead. .xy is already in original-image coordinates.
for poly in result.masks.xy:
    if len(poly) < 3:
        continue
    m = Image.new("1", img.size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=1)
    m = np.array(m, dtype=bool)
    px = rgb[m]
    if len(px) < 20:  # too few pixels for a stable median
        continue
    colours.append(hold_colour(px))
    kept.append(m)

labels = group(colours, threshold)
palette = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128)]

draw = ImageDraw.Draw(img, "RGBA")
for m, lab in zip(kept, labels):
    ys, xs = np.where(m)
    c = palette[lab % len(palette)]
    draw.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=c + (255,), width=4)

Path(dst).parent.mkdir(parents=True, exist_ok=True)
img.save(dst)

sizes = np.bincount(labels)
arr = np.array(colours)
print(f"{len(kept)} holds -> {len(sizes)} colour groups")
for i in np.argsort(-sizes)[:8]:
    L, a, b = arr[labels == i].mean(axis=0)  # centroid, not an arbitrary member
    print(f"  group {i:2d}: {sizes[i]:3d} holds   Lab ({L:5.1f},{a:6.1f},{b:6.1f})")
print(f"-> {dst}")
