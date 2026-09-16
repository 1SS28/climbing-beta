"""V3 end to end: photo -> holds -> route -> beta.

    .venv/bin/python v3_beta.py <image> [group] [y_top,y_bot,metres] [height_m]

`group` of -1 picks the most route-like cluster. The scale reference matters:
the metric model depends on it.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from beta import Climber, describe, find_start, search
from colour import group as colour_group
from colour import hold_colour
from routes import best_group
from scene import from_polygons, scale_from_reference

src = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else -1
# Scale reference, as "y_top,y_bottom,metres": two image rows a known distance
# apart, usually the top of the wall and the ground line. Scale cannot be
# recovered from one photo, so it is an input, not a guess.
ref = sys.argv[3] if len(sys.argv) > 3 else None
climber_h = float(sys.argv[4]) if len(sys.argv) > 4 else 1.68

from ultralytics import YOLO  # noqa: E402

result = YOLO("runs/v1_960_best.pt").predict(src, imgsz=960, conf=0.3, device="mps", verbose=False)[0]
img = Image.open(src).convert("RGB")
rgb = np.array(img)

polys, colours = [], []
for poly in result.masks.xy:
    if len(poly) < 3:
        continue
    m = Image.new("1", img.size, 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=1)
    px = rgb[np.array(m, dtype=bool)]
    if len(px) < 20:
        continue
    polys.append(poly)
    colours.append(hold_colour(px))

labels = colour_group(colours, 15.0)
sizes = np.bincount(labels)

if want >= 0:
    pick = want
else:
    # Not the largest group, which is usually the neutral greys shared by
    # every route. Pick the most route-like one instead.
    centres = np.array([np.asarray(p).mean(axis=0) for p in polys])
    by_group = {g: centres[labels == g] for g in range(len(sizes))}
    centroids = {g: np.array(colours)[labels == g].mean(axis=0) for g in by_group}
    pick, sc = best_group(by_group, centroids, img.height)
    if pick is None:
        print("no group looks like a route: nothing rises, chains and is saturated")
        sys.exit(1)
    print(f"route-likeness {sc:.3f}")

route = [i for i, l in enumerate(labels) if l == pick]
print(f"{len(polys)} holds, {len(sizes)} groups; climbing group {pick} ({len(route)} holds)")

if ref:
    y_top, y_bot, metres = (float(v) for v in ref.split(","))
    mpp = scale_from_reference((0, y_top), (0, y_bot), metres)
    print(f"scale: {metres} m over {abs(y_bot - y_top):.0f} px -> {mpp * 1000:.2f} mm/px")
else:
    mpp = 4.5 / img.height
    print(f"scale: ASSUMED 4.5 m over the full frame -> {mpp * 1000:.2f} mm/px (pass a reference)")

# Hands are limited to the route, feet may use any hold, as in most gyms.
holds = from_polygons(polys, img.height, mpp)
finish = max(route, key=lambda i: holds[i].y)
climber = Climber(height=climber_h)

start = find_start(holds, climber, hands=route)
if start is None:
    print("no feasible start stance: holds too far apart for this climber")
    sys.exit(1)

path, info = search(holds, start, finish, c=climber, hands=route)
if path is None:
    print(f"no beta found: {info}")
    sys.exit(1)

print(f"start {start} -> finish hold {finish}: {len(path) - 1} moves, cost {info:.2f}")
for line in describe(path, holds):
    print("  ", line)

# Ring the route holds, then draw each move as the moving limb's track.
draw = ImageDraw.Draw(img, "RGBA")
centres = [np.array(p).mean(axis=0) for p in polys]
for i in route:
    c = centres[i]
    draw.ellipse([c[0] - 18, c[1] - 18, c[0] + 18, c[1] + 18], outline=(255, 255, 255, 255), width=4)
for a, b in zip(path, path[1:]):
    limb = next(i for i in range(4) if a[i] != b[i])
    p, q = centres[a[limb]], centres[b[limb]]
    hand = limb < 2
    draw.line([tuple(p), tuple(q)], fill=(255, 60, 60, 220) if hand else (60, 160, 255, 220), width=5)

out = Path("out/v3_beta.png")
out.parent.mkdir(exist_ok=True)
img.save(out)
print(f"-> {out}   (red = hand move, blue = foot move)")
