"""Recover scale from the T-nut grid. Does not work; see README.

Gym walls are drilled on a regular lattice, so the bolt holes should be a ruler
lying in the image. Neither method here finds that lattice in real photos, at
these resolutions. Kept because the idea is an obvious one to re-try.
"""

import numpy as np

# Common commercial T-nut spacings, metres. Walls vary by manufacturer.
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

    Min-filter by repeated shifted minimum: cheaper than connected components,
    and one minimum per bolt hole is all that is needed.
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

    Every pair within range, not nearest neighbours only: on a lattice, pairs
    pile up at the spacing and its multiples, and holes hidden behind holds
    leave that peak standing.
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
    # How far the peak stands above a typical bin. Flat means no lattice.
    strength = hist[peak] / max(np.median(hist), 1e-6)
    return spacing, float(strength)


def metres_per_pixel(gray, assume=0.200):
    """Scale from the lattice, assuming a standard spacing. None if no grid found."""
    pts = candidates(gray)
    spacing, strength = dominant_spacing(pts)
    if spacing is None or strength < 3.0:
        return None, {"points": len(pts), "strength": strength}
    return assume / spacing, {"points": len(pts), "spacing_px": spacing, "strength": strength}


def dark_disks(gray, radius=6, thresh=24.0, blur=6):
    """Centres of dark, roughly circular blobs about `radius` px across.

    A bolt hole is a dark disk, so score each pixel by how much darker its own
    neighbourhood is than the ring around it, then take local maxima.

    The blur matters more than the threshold. Panels carry a dense scatter of
    tiny specks a few pixels across, and without smoothing they outnumber the
    real holes about fifty to one, which buries the grid: the spacing histogram
    then peaks on noise at 16 px instead of the true 145. Blurring first erases
    the specks and leaves the holes, cutting candidates from ~3000 to ~120.
    """
    gray = box_mean(gray, blur)
    resp = box_mean(gray, radius * 3) - box_mean(gray, radius)
    peak = resp.copy()
    sep = radius * 2
    for dy in range(-sep, sep + 1):
        for dx in range(-sep, sep + 1):
            if dy or dx:
                peak = np.maximum(peak, np.roll(np.roll(resp, dy, 0), dx, 1))
    return np.argwhere((resp >= peak) & (resp > thresh))[:, ::-1].astype(float)


def lattice_basis(points, max_px=400.0, bins=80):
    """The two shortest repeating displacement vectors among the candidates.

    Displacements between neighbouring bolt holes pile up at the grid spacing;
    displacements involving junk scatter. Taking the two strongest peaks that
    are not parallel gives the grid's own axes, without knowing them in advance.
    """
    if len(points) < 8:
        return None
    d = points[:, None, :] - points[None, :, :]
    d = d.reshape(-1, 2)
    d = d[(np.hypot(d[:, 0], d[:, 1]) > 8) & (np.hypot(d[:, 0], d[:, 1]) < max_px)]
    if len(d) < 20:
        return None

    hist, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=bins,
                                  range=[[-max_px, max_px], [-max_px, max_px]])
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    idx = np.dstack(np.meshgrid(xc, yc, indexing="ij"))
    flat = [(hist[i, j], idx[i, j]) for i in range(bins) for j in range(bins) if hist[i, j] > 0]
    flat.sort(key=lambda t: -t[0])

    first = None
    for _, v in flat:
        if np.hypot(*v) < 15:
            continue
        if first is None:
            first = v
            continue
        # second axis: not parallel to the first
        cross = abs(first[0] * v[1] - first[1] * v[0])
        if cross > 0.35 * np.hypot(*first) * np.hypot(*v):
            return np.array(first), np.array(v)
    return None


def on_lattice(points, a, b, tol=0.22):
    """Keep candidates that sit near an integer combination of the basis.

    This is the filter: bolt holes land on the grid, hair does not. It works on
    a real wall but under-selects, because one fixed basis is an affine model of
    what perspective makes projective. Holes further from the camera sit closer
    together in the image, so a single spacing fits only part of the frame and
    genuine holes elsewhere get rejected.

    The fix is to seed a homography from a local patch where the spacing is
    near-constant, then re-assign every candidate under that homography and
    refit. Not yet implemented.
    """
    B = np.column_stack([a, b])
    origin = points[len(points) // 2]
    coords = np.linalg.lstsq(B, (points - origin).T, rcond=None)[0].T
    off = np.abs(coords - np.round(coords))
    keep = (off.max(axis=1) < tol)
    return points[keep], coords[keep]
