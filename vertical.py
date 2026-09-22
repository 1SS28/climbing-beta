"""Estimate which way is down, from vertical lines in the scene.

A photo carries no gravity vector, so a wall's angle can only be measured
against the camera (see wall_angle in wall.py). But parallel vertical lines in
the world converge to a vanishing point in the image, and that point is the
world's up direction seen through the lens: back-projecting it through the
camera matrix recovers gravity in camera coordinates.

Gyms are full of usable verticals: wall edges, door frames, pipes, columns, the
mortar lines in blockwork.

STATUS: correct on synthetic scenes, useless on real photographs. Tilts of 10,
25 and 40 degrees come back exactly, and a level camera correctly yields no
finite vanishing point. On the gym photos it returns tilts around 75 degrees
with the vanishing point sitting inside the frame, which would mean the lens
pointed nearly at the ceiling.

The cause is the shortcut below. Treating every edge pixel as its own line
gives a million weakly-constrained lines per photo, and among a million there
are always a few thousand that agree on some meaningless point: the winning fit
explains 0.1 to 0.2 percent of them, unchanged whether near-vertical is defined
as 35 degrees or 8. Spurious agreement at that scale drowns the real signal.

The fix is to extract actual line segments first, long runs of collinear edge
pixels, and let each segment vote once with a weight set by its length. A
genuine wall edge is hundreds of pixels long; noise is not. That is a Hough
line transform or a connected-component fit, and it is the piece this module
skipped.
"""

import numpy as np

from lattice import sobel


def edge_lines(gray, grad_min=25.0, max_tilt_deg=35.0, stride=2):
    """Homogeneous lines through strong, roughly vertical edges.

    One line per pixel, which is the flaw: see the module docstring.

    A line's direction is perpendicular to the gradient, so a vertical edge has
    a near-horizontal gradient. Each pixel gives the line a*x + b*y + c = 0 with
    (a, b) its unit gradient, and any vanishing point v satisfies l . v = 0.
    """
    g = gray[::stride, ::stride]
    gx, gy = sobel(g)
    mag = np.hypot(gx, gy)
    ys, xs = np.nonzero(mag > grad_min)
    if len(xs) == 0:
        return np.zeros((0, 3))
    ux, uy = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs]
    # Vertical line => gradient points sideways => |uy| small.
    keep = np.abs(uy) < np.sin(np.radians(max_tilt_deg))
    xs, ys, ux, uy = xs[keep] * stride, ys[keep] * stride, ux[keep], uy[keep]
    return np.column_stack([ux, uy, -(ux * xs + uy * ys)])


def vanishing_point(lines, iters=4000, tol=3.0, seed=0, max_vp_factor=30.0, img_diag=7000.0):
    """The point most of these lines pass through, by RANSAC then least squares.

    Candidates must be a genuine finite intersection. Scoring l . v = 0 freely
    lets a point at infinity win outright: the edges are pre-filtered to be
    near-vertical, so they share a normal direction, and a vanishing point
    infinitely far up satisfies all of them without describing any convergence.
    Bounding how far the point may sit removes that free lunch, and a camera
    that really is level shows up as no candidate surviving, which the caller
    reads as zero tilt.
    """
    if len(lines) < 50:
        return None, 0
    rng = np.random.default_rng(seed)
    limit = max_vp_factor * img_diag
    best = (0, None)
    for _ in range(iters):
        i, j = rng.choice(len(lines), 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if abs(v[2]) < 1e-12:
            continue
        pt = v[:2] / v[2]
        if not np.isfinite(pt).all() or np.hypot(*pt) > limit:
            continue                       # effectively parallel: no information
        d = np.abs(lines @ v) / abs(v[2])  # pixel distance from line to point
        inl = int((d < tol).sum())
        if inl > best[0]:
            best = (inl, v / np.linalg.norm(v))
    if best[1] is None:
        return None, 0
    v = best[1]
    d = np.abs(lines @ v) / abs(v[2])
    keep = d < tol
    if keep.sum() >= 50:
        # Refit on inliers with the scale fixed, so the solution cannot drift
        # back to infinity: solve a*vx + b*vy = -c.
        A, rhs = lines[keep][:, :2], -lines[keep][:, 2]
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        v = np.array([sol[0], sol[1], 1.0])
        v = v / np.linalg.norm(v)
    return v, int(keep.sum())


def gravity_from_vanishing(v, K):
    """World-up in camera coordinates, from the vertical vanishing point."""
    d = np.linalg.inv(K) @ v
    d = d / np.linalg.norm(d)
    return -d if d[1] > 0 else d              # image y runs down; up is -y


def camera_tilt_deg(v, K):
    """How far the camera is tilted from level, in degrees.

    Zero means the optical axis is horizontal, so image-down is world-down and
    the upright assumption holds. Photographing an overhang from below forces
    this well away from zero, which is exactly when it matters.
    """
    up = gravity_from_vanishing(v, K)
    # Positive means the lens is pointed upward, which is what photographing an
    # overhang from the ground forces you to do.
    return float(np.degrees(np.arcsin(np.clip(up[2], -1.0, 1.0))))


def wall_angle_from_gravity(R, v, K):
    """Wall angle against gravity: 0 vertical, positive overhanging.

    Unlike wall_angle in wall.py this does not assume the camera was level, so
    it stays valid on the steep walls where that assumption breaks.
    """
    up = gravity_from_vanishing(v, K)
    n = R[:, 2]
    if n[2] > 0:
        n = -n                                # face the camera
    return float(np.degrees(np.arcsin(np.clip(np.dot(n, up), -1.0, 1.0))))
