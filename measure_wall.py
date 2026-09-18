"""Measure a wall's angle from its bolt-hole grid, with no tape measure.

    .venv/bin/python measure_wall.py <image>

The angle is scale-invariant: the rotation recovered from a homography does not
depend on how far apart the holes actually are, only on how the grid projects.
So this runs before anyone has measured a gym, and the spacing is needed later
only to turn the same homography into metres.

The chain is: detect holds and mask them out, find bolt holes by Hough voting
on gradient direction, fit a lattice in a small window where perspective is
near-uniform, grow it outward refitting a homography as it goes, then decompose
that homography with the focal length from EXIF.

STATUS: does not work yet. Detection is fine; the lattice fit is not. Hough
returns ~940 candidates on a real wall, but RANSAC then fits a basis that
merely connects a few of them by coincidence: probing outward from the seed,
the +a and +b directions land exactly (they were built from real points, so
that is circular) while -a and -b miss by 40 to 77 px. Growing therefore dies
at three holes.

Two causes worth fixing before trying again. The seed heuristic picks the
densest 400 px cell, which keeps landing at x=137 and x=197, the frame's left
edge where artefacts cluster rather than bare panel. And RANSAC accepts a basis
on six inliers, which is far too weak to distinguish a grid from a coincidence.

Four tapped corners recover the same wall angle to about 2 degrees and are
already implemented in calibrate.py, so nothing is blocked on this.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from lattice import grow, hough_circles
from wall import homography, intrinsics, plane_from_homography, wall_angle


def focal_from_heic(path):
    """35mm-equivalent focal length via macOS metadata, if the original is around."""
    for cand in (Path(path).with_suffix(".HEIC"), Path.home() / "Downloads" / (Path(path).stem + ".HEIC")):
        if cand.exists():
            out = subprocess.run(["mdls", "-name", "kMDItemFocalLength35mm", str(cand)],
                                 capture_output=True, text=True).stdout
            if "=" in out:
                try:
                    return float(out.split("=")[1].strip())
                except ValueError:
                    pass
    return None


def hold_mask(image_path, img, conf=0.15):
    """Every detected hold, dilated, so their rims do not pose as bolt holes."""
    from ultralytics import YOLO

    r = YOLO("runs/v1_960_best.pt").predict(str(image_path), imgsz=1600, conf=conf,
                                            device="mps", verbose=False)[0]
    m = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(m)
    for poly in (r.masks.xy if r.masks is not None else []):
        if len(poly) >= 3:
            d.polygon([tuple(p) for p in poly], fill=255)
    return np.array(m.filter(ImageFilter.MaxFilter(21))) > 0


def local_basis(points, centre, size=500.0, iters=30000, seed=0):
    """Fit a lattice where perspective is near-uniform.

    Inlier fraction rises sharply as the window shrinks (17% over 2600 px, 65%
    over 500) because one affine basis cannot describe a projective grid. So fit
    small and let grow() carry it across the frame.
    """
    rng = np.random.default_rng(seed)
    sel = points[(np.abs(points[:, 0] - centre[0]) < size / 2) &
                 (np.abs(points[:, 1] - centre[1]) < size / 2)]
    if len(sel) < 8:
        return None
    best = (0, None)
    for _ in range(iters):
        i, j, k = rng.choice(len(sel), 3, replace=False)
        a, b = sel[j] - sel[i], sel[k] - sel[i]
        la, lb = np.hypot(*a), np.hypot(*b)
        if not (40 < la < 110 and 40 < lb < 110):
            continue
        if abs(a[0] * b[1] - a[1] * b[0]) < 0.45 * la * lb:
            continue
        try:
            co = np.linalg.lstsq(np.column_stack([a, b]), (sel - sel[i]).T, rcond=None)[0].T
        except np.linalg.LinAlgError:
            continue
        ok = (np.abs(co - np.round(co)).max(axis=1) < 0.20).sum()
        if ok > best[0]:
            best = (int(ok), (sel[i], a, b))
    return best[1] if best[0] >= 6 else None


def measure(image_path, focal_35mm=None):
    img = Image.open(image_path)
    gray = np.array(img.convert("L"), dtype=float)
    masked = hold_mask(image_path, img)

    pts, _ = hough_circles(gray, vote_min=0.30)
    pts = np.array([p for p in pts if not masked[int(p[1]), int(p[0])]])
    if len(pts) < 20:
        return {"error": f"only {len(pts)} bolt-hole candidates"}

    # Start where the candidates are densest: that is bare, well-lit panel.
    grid = 400
    cells = {}
    for p in pts:
        cells.setdefault((int(p[0] // grid), int(p[1] // grid)), []).append(p)
    centre = np.mean(max(cells.values(), key=len), axis=0)

    basis = local_basis(pts, centre)
    if basis is None:
        return {"error": "no consistent lattice near the densest region"}
    origin, a, b = basis
    seed = int(np.argmin(np.hypot(*(pts - origin).T)))
    confirmed, missing = grow(pts, seed, a, b)
    if len(confirmed) < 8:
        return {"error": f"lattice grew to only {len(confirmed)} holes"}

    cell_xy = np.array(list(confirmed.keys()), dtype=float)   # hole units, not metres
    img_xy = np.array(list(confirmed.values()), dtype=float)
    H = homography(cell_xy, img_xy)

    out = {"candidates": len(pts), "grid_holes": len(confirmed), "occluded": len(missing),
           "basis_px": (float(np.hypot(*a)), float(np.hypot(*b)))}
    focal_35mm = focal_35mm or focal_from_heic(image_path)
    if focal_35mm:
        f_px = (float(focal_35mm) / 36.0) * max(img.size)
        R, _ = plane_from_homography(H, intrinsics(f_px, img.width, img.height))
        out["focal_35mm"] = focal_35mm
        out["wall_angle_deg"] = wall_angle(R)
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        r = measure(path)
        name = Path(path).name
        if "error" in r:
            print(f"{name}: {r['error']}")
        else:
            ang = r.get("wall_angle_deg")
            print(f"{name}: {r['grid_holes']} holes on the grid from {r['candidates']} candidates, "
                  f"{r['occluded']} occluded"
                  + (f", wall {ang:+.1f} deg" if ang is not None else " (no focal length, no angle)"))
