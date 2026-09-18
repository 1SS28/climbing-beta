"""Recover the wall plane from the T-nut grid.

The bolt holes form a known rectangular lattice lying in the wall. Four of them
pin down a homography between the wall plane and the image, and with the focal
length from EXIF that homography decomposes into the wall's orientation.

What it buys, all from hardware present on every wall in every gym:

- hold positions in true wall-plane metres, with perspective removed, so a hold
  high on the wall is no longer smaller than one at eye level
- distances between holds measured along the wall rather than in the image
- hold sizes in centimetres
- the wall's angle, which is what a flat-wall assumption gets wrong on an
  overhang, and overhangs are where balance decides whether a move works

Four holes is the mathematical minimum and nowhere near enough in practice.
With 2 px of error on each clicked point, a 2x2 block recovers a 25 degree wall
to within 20 degrees, which is worthless; 4x3 gives 3.6 degrees, and 6x5 gives
0.7. The decomposition amplifies small errors when the points are close
together relative to the frame, so the grid has to be both numerous and spread
out. That makes automatic detection of the holes a requirement rather than a
convenience: nobody is tapping thirty bolt holes by hand.

Numpy only, no OpenCV.
"""

import numpy as np


def homography(wall_xy, image_uv):
    """Solve H mapping wall-plane metres to image pixels. Needs 4+ points.

    Direct linear transform: each correspondence contributes two rows, and the
    solution is the null space, which is the smallest singular vector.
    """
    wall_xy = np.asarray(wall_xy, dtype=float)
    image_uv = np.asarray(image_uv, dtype=float)
    if len(wall_xy) < 4:
        raise ValueError("need at least 4 points to fix a homography")

    rows = []
    for (X, Y), (u, v) in zip(wall_xy, image_uv):
        rows.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        rows.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def intrinsics(focal_px, width_px, height_px):
    """Camera matrix, principal point assumed at the image centre."""
    return np.array([[focal_px, 0, width_px / 2.0],
                     [0, focal_px, height_px / 2.0],
                     [0, 0, 1.0]])


def plane_from_homography(H, K):
    """Decompose H into the wall's rotation and translation in camera axes.

    H = K [r1 r2 t] for a plane at Z=0, so K^-1 H recovers two rotation columns
    up to a common scale; the third is their cross product. The result is
    orthonormalised, since noise in the clicked points leaves it slightly off.
    """
    M = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)  # nearest true rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, t


def wall_angle(R):
    """Wall angle in degrees: 0 is vertical, positive overhanging.

    Assumes the camera was held roughly upright, so image-down is world-down.
    A phone's attitude would remove that assumption; EXIF does not carry it.
    """
    normal = R[:, 2]  # plane normal in camera axes
    if normal[2] > 0:
        normal = -normal  # face the camera
    # Tilt of the normal away from horizontal. A vertical wall has a horizontal
    # normal; an overhang tips it downward.
    horizontal = np.hypot(normal[0], normal[2])
    return float(np.degrees(np.arctan2(normal[1], horizontal)))


def to_wall_metres(H, image_uv):
    """Map image points onto the wall plane, in metres, undoing perspective."""
    pts = np.atleast_2d(np.asarray(image_uv, dtype=float))
    ones = np.ones((len(pts), 1))
    hom = np.hstack([pts, ones]) @ np.linalg.inv(H).T
    return hom[:, :2] / hom[:, 2:3]


def grid_points(cols, rows, spacing_m):
    """Wall-plane coordinates of a cols x rows block of bolt holes."""
    return np.array([[c * spacing_m, r * spacing_m] for r in range(rows) for c in range(cols)], dtype=float)
