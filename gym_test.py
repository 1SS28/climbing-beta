"""Run the pipeline over a folder of photos and report whether it holds up.

    .venv/bin/python gym_test.py data/mygym

This is the domain-shift check. Every number reported so far comes from
ClimbInst's own distribution, which is not the gym you climb at. For each photo
it reports holds found, how many colour groups came out, and whether any of
them looks like a route. Overlays go to out/gym/ for judging by eye, which is
the part that matters.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from colour import group as colour_group
from colour import hold_colour
from routes import best_group

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/mygym")
weights = sys.argv[2] if len(sys.argv) > 2 else "runs/v1_960_best.pt"
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

from ultralytics import YOLO  # noqa: E402

photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"))
if not photos:
    print(f"no photos in {folder}")
    sys.exit(1)

out = Path("out/gym")
out.mkdir(parents=True, exist_ok=True)
model = YOLO(weights)
rows = []

for path in photos:
    try:
        result = model.predict(str(path), imgsz=960, conf=conf, device="mps", verbose=False)[0]
    except Exception as exc:
        print(f"{path.name}: failed to read ({exc})")
        continue
    if result.masks is None:
        rows.append((path.name, 0, 0, None, 0.0, 0.0))
        continue

    img = Image.open(path).convert("RGB")
    rgb = np.array(img)
    polys, colours, confs = [], [], list(result.boxes.conf.cpu().numpy())
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

    if len(polys) < 4:
        rows.append((path.name, len(polys), 0, None, 0.0, float(np.mean(confs)) if confs else 0.0))
        continue

    labels = colour_group(colours, 15.0)
    centres = np.array([np.asarray(p).mean(axis=0) for p in polys])
    by_group = {g: centres[labels == g] for g in range(labels.max() + 1)}
    centroids = {g: np.array(colours)[labels == g].mean(axis=0) for g in by_group}
    pick, sc = best_group(by_group, centroids, img.height)

    # Scale the overlay to the source resolution: a phone photo is ~5700 px
    # wide, and a fixed line width vanishes once the image is shrunk to view.
    k = max(1, round(img.width / 900))
    draw = ImageDraw.Draw(img, "RGBA")
    for i, poly in enumerate(polys):
        on_route = pick is not None and labels[i] == pick
        draw.polygon(
            [tuple(p) for p in poly],
            outline=(255, 40, 40, 255) if on_route else (90, 220, 255, 190),
            width=5 * k if on_route else 2 * k,
        )
    img.thumbnail((1400, 1400))
    img.save(out / f"{path.stem}.png")

    n_route = int((labels == pick).sum()) if pick is not None else 0
    rows.append((path.name, len(polys), labels.max() + 1, n_route, sc, float(np.mean(confs))))

print(f"\n{'photo':28} {'holds':>6} {'groups':>7} {'route':>6} {'score':>6} {'conf':>6}")
for name, n, g, r, s, c in rows:
    print(f"{name[:28]:28} {n:6d} {g:7d} {str(r if r else '-'):>6} {s:6.3f} {c:6.2f}")

found = [r for r in rows if r[3]]
holds = [r[1] for r in rows]
print(f"\n{len(rows)} photos, median {int(np.median(holds)) if holds else 0} holds each")
print(f"{len(found)}/{len(rows)} have a route-like colour group")
print(f"overlays -> {out}  (red = the group picked as a route)")
print("\nJudge by eye: are the masks on real holds, and is the red group one route?")
