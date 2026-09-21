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

STATUS: produces plausible-looking numbers that are not yet repeatable. Do not
trust its output.

One run on IMG_3495 found 53 holes with a 0.06-hole residual, giving 1.12 mm/px
and a camera 4.28 m away, which agrees with the climber measured at 3.50 m in
the same frame. A later run on the same photo found 81 holes with a 0.19-hole
residual, 0.52 mm/px and a camera at 2.67 m: nearer than the climber standing
in front of the wall, which is impossible. The wall angle moved from +11.9 to
+1.9 degrees between those runs, and IMG_3497, an overhang, came back at -15.9,
i.e. leaning away.

The cause is that RANSAC settles on a different grid each time, and several
pitches fit the candidates about equally well, since Hough finds points between
bolt holes as well as on them. The residual does flag the worse fit, and
plausibility catches the impossible camera distance, so the ingredients for
choosing between runs exist; they are just not yet used. Until they are,
calibrate.py with four tapped corners is the path that gives a trustworthy
answer.

An affine version of this failed for a long time: one basis cannot describe a
grid across a whole frame, and growing from a seed died at three holes because
RANSAC kept accepting bases that merely connected a few points by coincidence.
Fitting the homography directly is what fixed it.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from lattice import hough_circles
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


def grid_score(H, pts, tol=0.25):
    """How many DISTINCT grid cells the candidates occupy under H.

    Counting matched points instead lets a degenerate homography win by
    collapsing the plane: an early version scored 62 "inliers" that turned out
    to sit on 7 cells, because everything lands near an integer once the plane
    is crushed. Distinct cells removes the incentive.
    """
    from wall import to_wall_metres

    try:
        g = to_wall_metres(H, pts)
    except np.linalg.LinAlgError:
        return 0, None
    off = np.abs(g - np.round(g)).max(axis=1)
    ok = (off < tol) & (np.abs(g) < 40).all(axis=1)
    if ok.sum() < 6:
        return 0, None
    return len(np.unique(np.round(g[ok]), axis=0)), ok


def fit_grid(pts, iters=150000, seed=1):
    """Projective RANSAC: sample four candidates as grid corners, fit, score.

    Projective rather than affine because a single basis cannot describe a grid
    across a whole frame; inlier fraction climbs from 17% over 2600 px to 65%
    over 500 as the window shrinks. Fitting the homography directly sidesteps
    that instead of fighting it.
    """
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(iters):
        quad = pts[rng.choice(len(pts), 4, replace=False)]
        c = quad.mean(axis=0)
        tl = quad[np.argmin((quad[:, 0] - c[0]) + (quad[:, 1] - c[1]))]
        br = quad[np.argmax((quad[:, 0] - c[0]) + (quad[:, 1] - c[1]))]
        tr = quad[np.argmax((quad[:, 0] - c[0]) - (quad[:, 1] - c[1]))]
        bl = quad[np.argmin((quad[:, 0] - c[0]) - (quad[:, 1] - c[1]))]
        o = np.array([tl, tr, bl, br])
        if len({tuple(p) for p in o}) != 4:
            continue
        top, left = np.hypot(*(tr - tl)), np.hypot(*(bl - tl))
        bot, right = np.hypot(*(br - bl)), np.hypot(*(br - tr))
        if min(top, left, bot, right) < 300:
            continue
        # Reject collapsed quads: opposite edges of a real grid block stay
        # comparable, and a near-triangle otherwise scores well by accident.
        if max(top, bot) / max(min(top, bot), 1) > 1.8:
            continue
        if max(left, right) / max(min(left, right), 1) > 1.8:
            continue
        for n in (4, 5, 6, 8, 10, 12):
            for m in (4, 5, 6, 8, 10, 12):
                if not (55 < top / n < 115 and 55 < left / m < 115):
                    continue
                H = homography(np.array([[0., m], [n, m], [0., 0.], [n, 0.]]), o)
                s, ok = grid_score(H, pts)
                if s > best[0]:
                    best = (s, (H, ok))
    return best[1]


def measure(image_path, focal_35mm=None, spacing_m=0.1524):
    """Recover a wall's grid, angle and scale from one photo."""
    from wall import to_wall_metres

    img = Image.open(image_path)
    gray = np.array(img.convert("L"), dtype=float)
    masked = hold_mask(image_path, img)

    pts, _ = hough_circles(gray, vote_min=0.30)
    pts = np.array([p for p in pts if not masked[int(p[1]), int(p[0])]])
    if len(pts) < 20:
        return {"error": f"only {len(pts)} bolt-hole candidates"}

    fit = fit_grid(pts)
    if fit is None:
        return {"error": "no consistent grid"}
    H0, ok = fit
    good = pts[ok]
    cells = np.round(to_wall_metres(H0, good))
    _, idx = np.unique(cells, axis=0, return_index=True)
    good, cells = good[idx], cells[idx]
    if len(good) < 12:
        return {"error": f"grid covers only {len(good)} holes"}

    # Half-pitch, because Hough finds points between bolt holes as well as on
    # them. This factor is guessed rather than determined, which is a large part
    # of why runs disagree: several pitches fit the candidates about equally
    # well and nothing here yet picks between them.
    Hm = homography(cells * spacing_m * 0.5, good)
    resid = np.hypot(*(to_wall_metres(homography(cells, good), good) - cells).T)

    mid = good.mean(axis=0)
    mpp = float(np.linalg.norm(to_wall_metres(Hm, [mid + [50., 0.]])[0]
                               - to_wall_metres(Hm, [mid])[0]) / 50.0)
    out = {"holes": len(good), "residual_holes": float(np.median(resid)),
           "metres_per_pixel": mpp, "H": Hm.tolist(),
           "image_size": list(img.size), "spacing_m": spacing_m}

    focal_35mm = focal_35mm or focal_from_heic(image_path)
    if focal_35mm:
        f_px = (float(focal_35mm) / 36.0) * max(img.size)
        R, t = plane_from_homography(Hm, intrinsics(f_px, img.width, img.height))
        out["focal_px"] = f_px
        out["wall_angle_deg"] = wall_angle(R)      # scale-invariant
        out["camera_distance_m"] = float(np.linalg.norm(t))
    return out


if __name__ == "__main__":
    import json

    for path in sys.argv[1:]:
        r = measure(path)
        name = Path(path).name
        if "error" in r:
            print(f"{name}: {r['error']}")
            continue
        ang = r.get("wall_angle_deg")
        print(f"{name}: {r['holes']} holes, residual {r['residual_holes']:.3f} holes, "
              f"{r['metres_per_pixel']*1000:.2f} mm/px"
              + (f", wall {ang:+.1f} deg, camera {r['camera_distance_m']:.2f} m" if ang is not None else ""))
        dst = Path(path).with_suffix(".calib.json")
        dst.write_text(json.dumps(r, indent=2))
        print(f"  -> {dst}")
