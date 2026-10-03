"""Scan the dataset for walls that actually carry colour-coded routes.

    uv run python find_routes.py [n_images]

Most ClimbInst walls are boards or competition walls where every hold is in
play, so no colour group forms a line. This finds the exceptions, which are the
usable test cases.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.colour import group as colour_group
from experiments.colour import hold_colour
from experiments.routes import best_group

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 120

from ultralytics import YOLO  # noqa: E402

model = YOLO("runs/v1_960_best.pt")
files = sorted(Path("data/climbinst/train/images").glob("*.jpg"))[:limit]

rows = []
for n, f in enumerate(files):
    try:
        result = model.predict(str(f), imgsz=1600, conf=0.35, device="mps", verbose=False)[0]
        if result.masks is None:
            continue
        img = Image.open(f).convert("RGB")
        rgb = np.array(img)

        pts, cols = [], []
        for poly in result.masks.xy:
            if len(poly) < 3:
                continue
            m = Image.new("1", img.size, 0)
            ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=1)
            px = rgb[np.array(m, dtype=bool)]
            if len(px) < 20:
                continue
            pts.append(np.asarray(poly).mean(axis=0))
            cols.append(hold_colour(px))
        if len(pts) < 6:
            continue

        pts = np.array(pts)
        labels = colour_group(cols, 15.0)
        by_group = {g: pts[labels == g] for g in range(labels.max() + 1)}
        centroids = {g: np.array(cols)[labels == g].mean(axis=0) for g in by_group}
        g, s = best_group(by_group, centroids, img.height, img.width)
        if g is not None:
            rows.append((s, f.name, g, int((labels == g).sum()), len(pts)))
    except Exception as exc:
        print(f"  skip {f.name}: {exc}")
    if (n + 1) % 25 == 0:
        print(f"  ...{n + 1}/{len(files)}")

rows.sort(reverse=True)
print(f"\nscanned {len(files)}, {len(rows)} with a route-like group\n")
print(f"{'score':>6}  {'group':>5} {'holds':>5} {'total':>5}  image")
for s, name, g, k, tot in rows[:15]:
    print(f"{s:6.3f}  {g:5d} {k:5d} {tot:5d}  {name}")
