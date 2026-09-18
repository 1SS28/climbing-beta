"""Calibrate a wall from four bolt holes, and save it for reuse.

    .venv/bin/python calibrate.py <image> x1,y1 x2,y2 x3,y3 x4,y4 --cols N --rows M

The four points are bolt holes at the corners of a block N holes wide and M
holes tall, given in the order top-left, top-right, bottom-left, bottom-right.
Spacing defaults to 8 inches; pass --spacing for a gym that differs.

Accuracy comes from how far apart the corners are, not how many are used: at
ordinary tapping accuracy, corners 2 m apart recover the wall angle to about
2.5 degrees and 3 m apart to 1.3, while four adjacent holes give 35. Pick the
widest block of clean wall in the frame.

All four must lie on ONE flat panel. Do not span a corner, a fold, or the join
between two faces. Four points define a homography exactly, so a set straddling
two planes fits with zero residual and returns a confident wrong answer, with
nothing to warn you. A wall made of several faces needs a calibration each.

Bolt spacing is regular within a panel but not across one. Screw-on holds also
leave holes off the grid; those are outliers a lattice fit discards, but they
are not valid calibration points, so pick holes that clearly sit on the
repeating pattern.

A wall only has to be done once. The result is written as JSON and reused by
any photo taken from the same position.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scene import MAX_WALL_M
from wall import homography, intrinsics, plane_from_homography, to_wall_metres, wall_angle


def calibrate(corners_px, cols, rows, spacing_m, image_size, focal_px=None):
    """Four corner bolt holes -> a wall calibration.

    Returns the homography from wall metres to image pixels, the wall angle
    where a focal length is known, and the scale at the centre of the block.
    """
    corners_px = np.asarray(corners_px, dtype=float)
    if corners_px.shape != (4, 2):
        raise ValueError("need exactly four corner points")
    if cols < 1 or rows < 1:
        raise ValueError("cols and rows count gaps between holes, so both must be >= 1")

    w, h = cols * spacing_m, rows * spacing_m
    wall_xy = np.array([[0.0, h], [w, h], [0.0, 0.0], [w, 0.0]])  # TL, TR, BL, BR
    H = homography(wall_xy, corners_px)

    out = {
        "H": H.tolist(),
        "block_m": [w, h],
        "spacing_m": spacing_m,
        "image_size": list(image_size),
    }

    # Scale at the block's centre, for anything still working in flat pixels.
    mid = to_wall_metres(H, [corners_px.mean(axis=0)])[0]
    step = to_wall_metres(H, [corners_px.mean(axis=0) + [10.0, 0.0]])[0]
    out["metres_per_pixel"] = float(np.linalg.norm(step - mid) / 10.0)

    if focal_px:
        K = intrinsics(focal_px, image_size[0], image_size[1])
        R, t = plane_from_homography(H, K)
        out["focal_px"] = float(focal_px)
        out["wall_angle_deg"] = wall_angle(R)
        out["camera_distance_m"] = float(np.linalg.norm(t))
    return out


def load(path):
    """Read a saved calibration, with H back as an array."""
    d = json.loads(Path(path).read_text())
    d["H"] = np.array(d["H"], dtype=float)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("corners", nargs=4, help="x,y of TL TR BL BR bolt holes")
    ap.add_argument("--cols", type=int, required=True, help="gaps between left and right holes")
    ap.add_argument("--rows", type=int, required=True, help="gaps between bottom and top holes")
    ap.add_argument("--spacing", type=float, default=0.2032, help="metres between holes")
    ap.add_argument("--focal-px", type=float, default=None, help="focal length, for the wall angle")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from PIL import Image

    img = Image.open(a.image)
    pts = [tuple(float(v) for v in c.split(",")) for c in a.corners]
    cal = calibrate(pts, a.cols, a.rows, a.spacing, img.size, a.focal_px)

    span = cal["metres_per_pixel"] * img.height
    if not 1.0 <= span <= MAX_WALL_M * 3:
        print(f"WARNING: this implies the frame spans {span:.1f} m, which is unlikely")

    print(f"block {cal['block_m'][0]:.2f} x {cal['block_m'][1]:.2f} m")
    print(f"scale {cal['metres_per_pixel'] * 1000:.3f} mm/px at the block centre")
    if "wall_angle_deg" in cal:
        print(f"wall angle {cal['wall_angle_deg']:+.1f} deg (0 vertical, + overhanging)")
        print(f"camera {cal['camera_distance_m']:.2f} m from the wall")

    dst = a.out or str(Path(a.image).with_suffix(".calib.json"))
    Path(dst).write_text(json.dumps(cal, indent=2))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
