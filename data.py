"""ClimbInst annotations. One Labelme JSON per image, polygons in image pixels."""

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load(path):
    """Read one annotation. Returns (RGB image, list of (N,2) polygons).

    imageData is null throughout ClimbInst, so the image normally comes from
    ../images/<imagePath>; the base64 branch is just in case a file has it.
    """
    path = Path(path)
    d = json.loads(path.read_text())
    if d.get("imageData"):
        img = Image.open(io.BytesIO(base64.b64decode(d["imageData"])))
    else:
        img = Image.open(path.parent.parent / "images" / d["imagePath"])
    return img.convert("RGB"), [np.array(s["points"], dtype=np.float32) for s in d["shapes"]]


def split(root, name):
    """Every annotation path in a split, sorted."""
    return sorted((Path(root) / name / "annotations").glob("*.json"))
