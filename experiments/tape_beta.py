"""Beta for a taped route: photo -> holds -> tape -> route -> moves.

    .venv/bin/python tape_beta.py <image> [tape_colour_index] [height_m]

Same pipeline as v3_beta, except the route comes from tape colour rather than
hold colour. On a wall where routes are marked with tape, two holds of the same
colour usually belong to different routes, so clustering holds by their own
colour groups the wrong things together.

Uses a .calib.json beside the image when one exists, so distances are real.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from beta import Climber, describe, find_start, route_chain, search
from experiments.colour import group as colour_group
from scene import from_polygons, from_wall_plane
from experiments.tape import assign, hold_mask, saturated_blobs

src = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else -1
climber_h = float(sys.argv[3]) if len(sys.argv) > 3 else 1.68

from ultralytics import YOLO  # noqa: E402

img = Image.open(src).convert("RGB")
result = YOLO("runs/v1_960_best.pt").predict(src, imgsz=1600, conf=0.15, device="mps", verbose=False)[0]
polys = [np.asarray(p) for p in result.masks.xy if len(p) >= 3]
centres = np.array([p.mean(axis=0) for p in polys])
print(f"{len(polys)} holds")

tapes = saturated_blobs(img, hold_mask(img, polys))
if not tapes:
    print("no route tape found")
    sys.exit(1)
tape_lab = colour_group(np.array([t["colour"] for t in tapes]), 18.0)
owner = assign(tapes, centres, max_dist=0.06 * img.height)

by_colour = {}
for hold_i, tape_i in owner.items():
    by_colour.setdefault(int(tape_lab[tape_i]), []).append(hold_i)

# A colour can appear twice on one wall: gyms reuse tape. Holds sharing a
# colour but sitting far apart are separate routes, not one broken route, so
# split each colour into spatial clusters before climbing anything. On this
# wall the pink tape formed two groups 2.9 m apart, which read as a single
# route with an impossible gap.
SPLIT_PX = 0.18 * img.height
routes, labels_out = {}, []
for g, hs in by_colour.items():
    unplaced = set(hs)
    while unplaced:
        seed = unplaced.pop()
        cluster, frontier = [seed], [seed]
        while frontier:
            i = frontier.pop()
            for j in list(unplaced):
                if np.hypot(*(centres[j] - centres[i])) <= SPLIT_PX:
                    unplaced.discard(j)
                    cluster.append(j)
                    frontier.append(j)
        routes[len(routes)] = cluster
        labels_out.append(g)

cols = np.array([t["colour"] for t in tapes])
print(f"{len(tapes)} tapes -> {len(routes)} routes (colour split by position):")
for k, hs in sorted(routes.items(), key=lambda kv: -len(kv[1])):
    L, a, b = cols[tape_lab == labels_out[k]].mean(axis=0)
    ys = [centres[i][1] for i in hs]
    print(f"  route {k}: {len(hs):2d} holds  Lab({L:4.0f},{a:5.0f},{b:5.0f})  "
          f"y {min(ys):.0f}-{max(ys):.0f} px")

pick = want if want >= 0 else max(routes, key=lambda g: len(routes[g]))
route = routes.get(pick)
if not route or len(route) < 3:
    print(f"tape {pick} has too few holds to climb")
    sys.exit(1)
L, a, b = cols[tape_lab == labels_out[pick]].mean(axis=0)
print(f"\nclimbing route {pick}, tape Lab({L:.0f},{a:.0f},{b:.0f}), {len(route)} holds")

calib = Path(src).with_suffix(".calib.json")
if calib.exists():
    import calibrate as _cal
    cal = _cal.load(calib)
    holds = from_wall_plane(polys, cal["H"])
    print(f"calibrated: {cal['metres_per_pixel']*1000:.2f} mm/px"
          + (f", wall {cal['wall_angle_deg']:+.1f} deg" if "wall_angle_deg" in cal else ""))
else:
    holds = from_polygons(polys, img.height, 4.5 / img.height)
    print("no calibration: assuming 4.5 m over the frame")

climber = Climber(height=climber_h)
chain, why = route_chain(holds, route, climber)
if chain is None:
    print(f"route does not connect: {why}")
    sys.exit(1)
if len(chain) < len(route):
    print(f"chain uses {len(chain)} of {len(route)} holds")

start = find_start(holds, climber, hands=chain)
if start is None:
    print("no feasible start stance")
    sys.exit(1)
path, info = search(holds, start, chain[-1], c=climber, hands=chain)
if path is None:
    print(f"no beta found: {info}")
    sys.exit(1)
if len(path) - 1 < 2:
    # A start stance can already have a hand on the finish when the route is
    # only two or three holds. That is not a climb, it is too little route.
    print(f"route too short to be a climb: {len(chain)} holds gave {len(path)-1} moves")
    sys.exit(1)

print(f"\n{len(path)-1} moves, cost {info:.2f}")
for line in describe(path, holds):
    print("  ", line)

draw = ImageDraw.Draw(img, "RGBA")
for i in chain:
    c = centres[i]
    draw.ellipse([c[0]-22, c[1]-22, c[0]+22, c[1]+22], outline=(255, 255, 255, 255), width=6)
for t in tapes:
    c = t["centre"]
    draw.ellipse([c[0]-14, c[1]-14, c[0]+14, c[1]+14], outline=(255, 255, 0, 200), width=4)
for aa, bb in zip(path, path[1:]):
    limb = next(i for i in range(4) if aa[i] != bb[i])
    p, q = centres[aa[limb]], centres[bb[limb]]
    draw.line([tuple(p), tuple(q)],
              fill=(255, 60, 60, 230) if limb < 2 else (60, 160, 255, 230), width=7)
out = Path("out/tape_beta.png")
out.parent.mkdir(exist_ok=True)
img.save(out)
print(f"-> {out}   (red = hand, blue = foot, yellow = tape)")
