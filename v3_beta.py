"""V3 end to end: photo -> holds -> route -> beta.

    uv run python v3_beta.py <image> [group] [wall_height_m]

`group` picks which colour cluster to climb; default is the largest. Pass a
wall height in metres if you know it — the whole metric scale hangs off it.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from beta import Climber, describe, find_start, search
from colour import group as colour_group
from colour import hold_colour
from scene import from_polygons

src = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else -1
wall_h = float(sys.argv[3]) if len(sys.argv) > 3 else 4.5

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
pick = int(np.argmax(sizes)) if want < 0 else want
route = [i for i, l in enumerate(labels) if l == pick]
print(f"{len(polys)} holds, {len(sizes)} groups; climbing group {pick} ({len(route)} holds)")

holds = from_polygons([polys[i] for i in route], img.height, wall_h)
finish = int(np.argmax([h.y for h in holds]))

start = find_start(holds)
if start is None:
    print("no feasible start stance — holds too far apart for this climber")
    sys.exit(1)

path, info = search(holds, start, finish, c=Climber(height=1.75))
if path is None:
    print(f"no beta found: {info}")
    sys.exit(1)

print(f"start {start} -> finish hold {finish}: {len(path) - 1} moves, cost {info:.2f}")
for line in describe(path, holds):
    print("  ", line)

# Draw the route holds with their index, and the moving limb's track per move.
draw = ImageDraw.Draw(img, "RGBA")
centres = [np.array(polys[i]).mean(axis=0) for i in route]
for n, c in enumerate(centres):
    draw.ellipse([c[0] - 16, c[1] - 16, c[0] + 16, c[1] + 16], outline=(255, 255, 255, 255), width=3)
    draw.text((c[0] - 4, c[1] - 6), str(n), fill=(255, 255, 255, 255))
for a, b in zip(path, path[1:]):
    limb = next(i for i in range(4) if a[i] != b[i])
    p, q = centres[a[limb]], centres[b[limb]]
    hand = limb < 2
    draw.line([tuple(p), tuple(q)], fill=(255, 60, 60, 220) if hand else (60, 160, 255, 220), width=5)

out = Path("out/v3_beta.png")
out.parent.mkdir(exist_ok=True)
img.save(out)
print(f"-> {out}   (red = hand move, blue = foot move)")
