"""Pseudo-label route tape across the dataset, to train a detector on.

    .venv/bin/python make_tape_labels.py [limit]

No tape dataset exists, so the labels come from the rule-based detector in
tape.py. That detector has decent precision and poor recall, which is the right
way round for this: every box it produces is probably tape, and a network
trained on them learns what tape looks like rather than inheriting a chroma
cutoff. The hope is precisely that the model then fires on dimmer and smaller
strips the rules reject.

The obvious risk is inheriting the rules' blind spots. Anything they never find
is never labelled, so the model cannot learn it from here.

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

from experiments.tape import hold_mask, saturated_blobs

OUT = Path("data/tape")
PAD = 6  # box padding, px


WORK_PX = 1400  # detect on a downscaled copy; the row loop in saturated_blobs
                # is pure Python, so full-resolution photos take ~6 s each


def boxes_for(img, polys, min_px=30):
    """Boxes in ORIGINAL image pixels, detected on a downscaled copy."""
    k = max(img.size) / float(WORK_PX)
    if k > 1.0:
        small = img.resize((int(img.width / k), int(img.height / k)))
        polys = [np.asarray(p) / k for p in polys]
        min_px = max(12, int(min_px / (k * k)))
    else:
        small, k = img, 1.0
    tapes = saturated_blobs(small, hold_mask(small, polys), min_px=min_px)
    out = []
    for t in tapes:
        # saturated_blobs gives a centre and an area; recover a box from the
        # blob's own extent by re-thresholding locally would be slower, so use
        # a square of equivalent area, padded.
        side = np.sqrt(t["pixels"]) * 1.6
        cx, cy = t["centre"]
        out.append((cx * k, cy * k, (side + PAD) * k, (side + PAD) * k))
    return out


def main():
    from ultralytics import YOLO

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    model = YOLO("runs/v1_960_best.pt")
    sources = sorted(Path("data/yolo/images/train").glob("*.jpg"))[:limit]
    sources += sorted(Path("data/mygym").glob("*.jpg"))

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept = total = 0
    for n, f in enumerate(sources):
        try:
            img = Image.open(f).convert("RGB")
            if min(img.size) < 600:
                continue
            r = model.predict(str(f), imgsz=1280, conf=0.2, device="mps", verbose=False)[0]
            if r.masks is None:
                continue
            polys = [np.asarray(p) for p in r.masks.xy if len(p) >= 3]
            bx = boxes_for(img, polys)
            if len(bx) < 2:          # an image with one blob is probably a false one
                continue
            split = "val" if kept % 8 == 0 else "train"
            link = OUT / "images" / split / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
            W, H = img.size
            lines = [f"0 {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}" for cx, cy, w, h in bx]
            (OUT / "labels" / split / f"{f.stem}.txt").write_text("\n".join(lines) + "\n")
            kept += 1
            total += len(bx)
        except Exception as exc:
            print(f"  skip {f.name}: {exc}", flush=True)
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(sources)} scanned, {kept} images kept, {total} tapes", flush=True)

    (OUT / "tape.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: tape\n")
    print(f"\n{kept} images, {total} pseudo-labelled tapes -> {OUT}")


if __name__ == "__main__":
    main()
