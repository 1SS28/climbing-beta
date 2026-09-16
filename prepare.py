"""Convert ClimbInst Labelme polygons into the layout YOLO segmentation expects.

ClimbInst ships train/ and test/ only. The val/ split advertised on the dataset
card is empty on the Hub, so validation is carved out of train here.

    uv run python prepare.py
"""

import hashlib
import json
from pathlib import Path

SRC = Path("data/climbinst")
DST = Path("data/yolo")
VAL_PERCENT = 10


def split_of(name):
    """Stable holdout. Hashing the filename keeps the split fixed when files
    are added, removed or reordered."""
    return "val" if int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100 < VAL_PERCENT else "train"


def polygons(ann):
    """Labelme shapes -> YOLO segment lines, normalised and clamped to the frame."""
    w, h = ann["imageWidth"], ann["imageHeight"]
    for shape in ann["shapes"]:
        pts = shape["points"]
        if len(pts) < 3:  # a degenerate polygon has no area to segment
            continue
        coords = [min(max(v / dim, 0.0), 1.0) for x, y in pts for v, dim in ((x, w), (y, h))]
        yield "0 " + " ".join(f"{c:.6f}" for c in coords)


def main():
    counts = {}
    for src_split in ("train", "test"):
        for ann_path in sorted((SRC / src_split / "annotations").glob("*.json")):
            ann = json.loads(ann_path.read_text())

            img_src = SRC / src_split / "images" / ann["imagePath"]
            if not img_src.exists():  # imagePath sometimes disagrees on extension
                matches = list((SRC / src_split / "images").glob(ann_path.stem + ".*"))
                if not matches:
                    print(f"skip {ann_path.name}: no image")
                    continue
                img_src = matches[0]

            split = split_of(ann_path.name) if src_split == "train" else "test"
            lines = list(polygons(ann))

            (DST / "images" / split).mkdir(parents=True, exist_ok=True)
            (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

            link = DST / "images" / split / img_src.name
            if not link.exists():  # symlink rather than copy; the images are 4 GB
                link.symlink_to(img_src.resolve())
            (DST / "labels" / split / f"{img_src.stem}.txt").write_text("\n".join(lines) + "\n")

            counts[split] = counts.get(split, [0, 0])
            counts[split][0] += 1
            counts[split][1] += len(lines)

    (DST / "climbinst.yaml").write_text(
        f"path: {DST.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: hold\n"
    )
    for split, (imgs, inst) in sorted(counts.items()):
        print(f"{split:6s} {imgs:5d} images  {inst:6d} instances")


if __name__ == "__main__":
    main()
