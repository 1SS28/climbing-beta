"""Compare the trained tape detector against the rules that labelled it.

    .venv/bin/python eval_tape.py [weights]

The question is recall. The rules find about 8 strips on a wall carrying many
more, which leaves a route with 3 holds where it needs 8 to 15. A model trained
on those same strips is useful only if it finds ones they missed, so what
matters here is the count and whether the extra detections group into sensible
colours, not agreement with the labels.

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
from pathlib import Path

import numpy as np
from PIL import Image

from colour import group as colour_group
from colour import srgb_to_lab
from tape import hold_mask, saturated_blobs


def model_tapes(model, path, img, conf=0.25, imgsz=1280):
    r = model.predict(str(path), imgsz=imgsz, conf=conf, device="mps", verbose=False)[0]
    rgb = np.array(img)
    out = []
    for b in r.boxes.xyxy.cpu().numpy():
        x0, y0, x1, y1 = [int(v) for v in b]
        patch = rgb[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
        if len(patch) < 10:
            continue
        out.append({"centre": np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0]),
                    "colour": np.median(srgb_to_lab(patch), axis=0)})
    return out


def summarise(tapes, label):
    if not tapes:
        print(f"  {label:10} none")
        return
    cols = np.array([t["colour"] for t in tapes])
    lab = colour_group(cols, 18.0)
    counts = np.bincount(lab)
    usable = int((counts >= 2).sum())
    print(f"  {label:10} {len(tapes):3d} tapes, {usable} colours with 2+ "
          + " ".join(f"{c}x(a{cols[lab==g][:,1].mean():.0f},b{cols[lab==g][:,2].mean():.0f})"
                     for g, c in enumerate(counts) if c >= 2))


if __name__ == "__main__":
    from ultralytics import YOLO

    weights = sys.argv[1] if len(sys.argv) > 1 else "runs/tape/weights/best.pt"
    holds_model = YOLO("runs/v1_960_best.pt")
    tape_model = YOLO(weights)
    for f in sorted(Path("data/mygym").glob("*.jpg")):
        img = Image.open(f).convert("RGB")
        r = holds_model.predict(str(f), imgsz=1600, conf=0.15, device="mps", verbose=False)[0]
        polys = [np.asarray(p) for p in r.masks.xy if len(p) >= 3]
        print(f"{f.name}:")
        summarise(saturated_blobs(img, hold_mask(img, polys)), "rules")
        summarise(model_tapes(tape_model, f, img), "trained")
