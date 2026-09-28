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

STATUS: repeatable, and the scale is self-consistent. The angle is measured
relative to the camera rather than to gravity, so treat it as unverified.

Runs used to disagree because each took whatever one random RANSAC fit
returned: the same photo gave 1.12 mm/px on one run and 0.52 on the next, with
the camera at 4.28 m and then at 2.67 m, the latter nearer than a climber
standing in front of the wall. Now restarts use fixed seeds and the winner is
the lowest-residual grid whose geometry is physically possible, so the same
photo gives the same answer.

The two checks do different jobs, which is what earlier versions conflated. The
residual is in hole units and so is independent of pitch: it says which grid
fit is real. Plausibility says which pitch that grid corresponds to, because a
wrong pitch fits perfectly and merely puts the camera somewhere impossible.

What remains unverified is the angle, for a reason no amount of fitting can
address: see wall_angle in wall.py. It is measured against the camera, and a
photo carries no gravity vector.

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

from lattice import confirm_grid, hough_circles
from scene import MAX_WALL_M
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


def refit(pts, H0, spacing_m, pitch):
    """Refit the homography over every inlier, and report how well it fits.

    The residual is in hole units and so does not depend on the pitch guess,
    which is what lets it choose between grid fits while plausibility chooses
    the pitch. Conflating those two jobs is why earlier runs disagreed.
    """
    from wall import to_wall_metres

    cells = np.round(to_wall_metres(H0, pts))
    _, idx = np.unique(cells, axis=0, return_index=True)
    good, cells = pts[idx], cells[idx]
    if len(good) < 12:
        return None
    H_cells = homography(cells, good)
    resid = float(np.median(np.hypot(*(to_wall_metres(H_cells, good) - cells).T)))
    H_m = homography(cells * spacing_m * pitch, good)
    return {"H": H_m, "cells": cells, "pts": good, "residual_holes": resid}


def plausible_wall(H_m, img_size, focal_px):
    """Reject geometry a bouldering gym cannot produce.

    A wrong pitch shows up here rather than in the residual: the grid still fits
    perfectly, it is merely the wrong size, so the camera ends up somewhere
    impossible. One earlier run put the camera 2.67 m away on a wall
    photographed from behind a climber standing at 3.50 m.
    """
    from wall import to_wall_metres

    w, h = img_size
    R, t = plane_from_homography(H_m, intrinsics(focal_px, w, h))
    mid = np.array([w / 2.0, h / 2.0])
    mpp = float(np.linalg.norm(to_wall_metres(H_m, [mid + [50.0, 0.0]])[0]
                               - to_wall_metres(H_m, [mid])[0]) / 50.0)
    # Distance where the scale was taken, not to the grid's origin corner. On a
    # tilted wall those differ, and quoting one with the other invites a
    # cross-check that fails for no real reason. Metres per pixel at distance D
    # is D / focal, so the two are the same measurement.
    dist = mpp * focal_px
    frame_m = mpp * h
    # A 24mm-equivalent lens at distance D spans about 1.5*D vertically, so a
    # frame much taller than that means the scale is wrong rather than the room
    # being large.
    ok = (2.0 <= dist <= 12.0) and (1.5 <= frame_m <= MAX_WALL_M * 2.5) \
        and (frame_m <= 2.2 * dist)
    return ok, {"camera_distance_m": dist, "metres_per_pixel": mpp,
                "frame_span_m": frame_m, "wall_angle_deg": wall_angle(R)}


def measure(image_path, focal_35mm=None, spacing_m=0.1524, restarts=8, iters=25000):
    """Recover a wall's grid, angle and scale from one photo.

    Deterministic: the restarts use fixed seeds, so the same photo gives the
    same answer. Each restart proposes a grid; the lowest-residual grid whose
    geometry is physically possible wins. Earlier versions took whatever one
    random fit returned, which is why the same photo gave 1.12 mm/px on one run
    and 0.52 on the next.
    """
    img = Image.open(image_path)
    gray = np.array(img.convert("L"), dtype=float)
    masked = hold_mask(image_path, img)

    pts, _ = hough_circles(gray, vote_min=0.30)
    pts = np.array([p for p in pts if not masked[int(p[1]), int(p[0])]])
    if len(pts) < 20:
        return {"error": f"only {len(pts)} bolt-hole candidates"}

    focal_35mm = focal_35mm or focal_from_heic(image_path)
    f_px = (float(focal_35mm) / 36.0) * max(img.size) if focal_35mm else None

    # Hough finds points between bolt holes as well as on them, so the detected
    # lattice can sit at a fraction of the true pitch. Try the candidates rather
    # than assume one.
    PITCHES = (1.0, 0.5, 0.25)
    tried, best = [], None
    for seed in range(restarts):
        fit = fit_grid(pts, iters=iters, seed=seed)
        if fit is None:
            continue
        H0, ok = fit
        for pitch in PITCHES:
            r = refit(pts[ok], H0, spacing_m, pitch)
            if r is None:
                continue
            if f_px is None:
                cand = dict(r, pitch=pitch, plausible=True)
            else:
                good, geo = plausible_wall(r["H"], img.size, f_px)
                cand = dict(r, pitch=pitch, plausible=good, **geo)
            tried.append((cand["residual_holes"], pitch, len(r["pts"]), cand["plausible"]))
            if not cand["plausible"]:
                continue
            # Lowest residual wins; more holes breaks a tie.
            key = (cand["residual_holes"], -len(cand["pts"]))
            if best is None or key < (best["residual_holes"], -len(best["pts"])):
                best = cand

    if best is None:
        near = sorted(tried)[:3]
        return {"error": "no physically plausible grid",
                "rejected": [{"residual_holes": round(r, 3), "pitch": p, "holes": n} for r, p, n, _ in near]}

    out = {"holes": len(best["pts"]), "residual_holes": best["residual_holes"],
           "pitch_of_true_spacing": best["pitch"], "spacing_m": spacing_m,
           "H": best["H"].tolist(), "image_size": list(img.size),
           "restarts": restarts, "candidates_considered": len(tried)}
    for k in ("metres_per_pixel", "wall_angle_deg", "camera_distance_m", "frame_span_m"):
        if k in best:
            out[k] = best[k]
    if f_px:
        out["focal_px"] = f_px
    return out


if __name__ == "__main__":
    import json

    for path in sys.argv[1:]:
        r = measure(path)
        name = Path(path).name
        if "error" in r:
            print(f"{name}: {r['error']}")
            for rej in r.get("rejected", []):
                print(f"    closest rejected: residual {rej['residual_holes']} at pitch "
                      f"{rej['pitch']}, {rej['holes']} holes")
            continue
        ang = r.get("wall_angle_deg")
        print(f"{name}: {r['holes']} holes, residual {r['residual_holes']:.3f}, "
              f"pitch x{r['pitch_of_true_spacing']}, {r['metres_per_pixel']*1000:.2f} mm/px"
              + (f", wall {ang:+.1f} deg, camera {r['camera_distance_m']:.2f} m, "
                 f"frame {r['frame_span_m']:.1f} m" if ang is not None else ""))
        dst = Path(path).with_suffix(".calib.json")
        dst.write_text(json.dumps(r, indent=2))
        print(f"  -> {dst}")
