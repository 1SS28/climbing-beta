"""Recover scale from the T-nut grid.

Gym walls are drilled on a regular lattice, so if the bolt holes can be found
the spacing between them is a ruler lying in the image. That would replace the
hardcoded wall height that every metric constraint currently hangs off — the
single weakest assumption in the pipeline.

Whether this survives real photos is the open question; see the notes in
README. Written with numpy only, in keeping with the rest of the project.
"""

import numpy as np

# Common commercial T-nut spacings, metres. Walls vary by manufacturer, so the
# best this can do is match an observed spacing to a plausible standard.
STANDARD_SPACINGS = (0.100, 0.125, 0.150, 0.200, 0.203)  # 203 mm = 8 inches


def box_mean(a, k):
    """Mean over a (2k+1) square, via an integral image. O(n) regardless of k."""
    p = np.pad(a, k + 1, mode="edge")
    integral = p.cumsum(0).cumsum(1)
    h, w = a.shape
    y, x = np.arange(h), np.arange(w)
    y0, y1 = y[:, None], y[:, None] + 2 * k + 1
    x0, x1 = x[None, :], x[None, :] + 2 * k + 1
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return total / (2 * k + 1) ** 2


def local_minima(a, radius):
    """True where a pixel is the darkest in its square neighbourhood.

    A min-filter by repeated shifted minimum — cheaper to write than connected
    components, and a bolt hole yields one minimum, which is all that's wanted.
    """
    out = a.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                out = np.minimum(out, np.roll(np.roll(a, dy, axis=0), dx, axis=1))
    return a <= out


def candidates(gray, radius=3, context=14, depth=8.0):
    """Points that look like bolt holes: small, round, darker than their surroundings."""
    dark = box_mean(gray, context) - gray > depth
    pts = np.argwhere(local_minima(gray, radius) & dark)
    return pts[:, ::-1].astype(float)  # (x, y)


def dominant_spacing(points, lo=8.0, hi=120.0, bins=112):
    """The most common short distance between candidates, in pixels.

    Uses every pair within range rather than nearest neighbours only: on a
    lattice, pairs pile up at the spacing and its multiples, and missing holes
    (hidden behind holds) leave that peak standing where a nearest-neighbour
    statistic would smear it.
    """
    if len(points) < 8:
        return None, 0.0
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    d = d[np.triu_indices(len(points), k=1)]
    d = d[(d > lo) & (d < hi)]
    if len(d) < 20:
        return None, 0.0
    hist, edges = np.histogram(d, bins=bins, range=(lo, hi))
    peak = int(np.argmax(hist))
    spacing = 0.5 * (edges[peak] + edges[peak + 1])
    # How much the peak stands above the typical bin — a flat histogram means
    # there is no lattice here, only scattered specks.
    strength = hist[peak] / max(np.median(hist), 1e-6)
    return spacing, float(strength)


def metres_per_pixel(gray, assume=0.200):
    """Scale from the lattice, assuming a standard spacing. None if no grid found."""
    pts = candidates(gray)
    spacing, strength = dominant_spacing(pts)
    if spacing is None or strength < 3.0:
        return None, {"points": len(pts), "strength": strength}
    return assume / spacing, {"points": len(pts), "spacing_px": spacing, "strength": strength}
