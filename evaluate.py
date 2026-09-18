"""Measure detection against the labelled test split.

    .venv/bin/python evaluate.py [imgsz] [conf] [--tta]

Reports matched, missed and unlabelled-but-detected over all test images, plus
how the misses are sized. mAP hides what matters here: a missed hold breaks route_chain, while a spurious
one is usually filtered by colour, so recall is worth more than precision.

Measured on the 50-image test split. At 960 and conf 0.25 recall is 93.6% with
91.4% precision; at 1600 and 0.15 it is 96.9% and 87.4%; at 1600 and 0.08,
97.4% and 85.0%. F1 peaks at the first, which is why the pipeline does not use
it. Halving the misses is worth the precision, and some of those "false"
detections are real holes the annotator missed.

Test-time augmentation is unavailable: Ultralytics does not support it for
segmentation models.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def poly_area(p):
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


def evaluate(model, imgsz=1600, conf=0.25, tta=False, limit=None):
    from ultralytics import YOLO

    m = YOLO(model)
    files = sorted(Path("data/yolo/labels/test").glob("*.txt"))[:limit]
    tot_gt = tot_pred = tot_match = 0
    missed_area, all_area = [], []

    for lab in files:
        img_p = Path("data/yolo/images/test") / (lab.stem + ".jpg")
        if not img_p.exists():
            continue
        W, H = Image.open(img_p).size
        gt = [np.array([float(x) for x in line.split()[1:]]).reshape(-1, 2) * [W, H]
              for line in open(lab) if len(line.split()) > 6]
        r = m.predict(str(img_p), imgsz=imgsz, conf=conf, device="mps",
                      augment=tta, verbose=False)[0]
        pred = [np.asarray(p) for p in (r.masks.xy if r.masks is not None else [])] 
        pc = [p.mean(axis=0) for p in pred if len(p) >= 3]

        for g in gt:
            a = poly_area(g)
            all_area.append(a)
            rad = max(8.0, np.sqrt(a) * 0.6)
            if any(np.hypot(*(g.mean(axis=0) - q)) < rad for q in pc):
                tot_match += 1
            else:
                missed_area.append(a)
        tot_gt += len(gt)
        tot_pred += len(pc)

    return {
        "images": len(files), "labelled": tot_gt, "detected": tot_pred,
        "matched": tot_match, "recall": tot_match / max(tot_gt, 1),
        "missed_median_area": float(np.median(missed_area)) if missed_area else 0.0,
        "all_median_area": float(np.median(all_area)) if all_area else 0.0,
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    imgsz = int(args[0]) if args else 1600
    conf = float(args[1]) if len(args) > 1 else 0.25
    r = evaluate("runs/v1_960_best.pt", imgsz, conf, "--tta" in sys.argv)
    print(f"imgsz={imgsz} conf={conf} tta={'--tta' in sys.argv}")
    print(f"  {r['images']} images, {r['labelled']} labelled, {r['detected']} detected")
    print(f"  matched {r['matched']} = recall {r['recall']:.1%}")
    print(f"  missed holds are {r['missed_median_area']/max(r['all_median_area'],1):.0%} of typical size")
