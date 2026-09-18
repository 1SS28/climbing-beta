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
    displacements involving junk scatter. The strongest peak alone is not
    enough, because a two-hole step generates nearly as many pairs as a one-hole
    step, so the peak is often a harmonic: on a real wall that gave 130 px for a
    grid whose true spacing is 65. Taking the shortest peak instead lands in the
    noise floor. So take the strongest, then walk down its own halves and thirds
    for as long as they are still supported.
    """
    if len(points) < 8:
        return None
    d = (points[:, None, :] - points[None, :, :]).reshape(-1, 2)
    r = np.hypot(d[:, 0], d[:, 1])
    d = d[(r > 8) & (r < max_px)]
    if len(d) < 20:
        return None

    hist, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=bins,
                                  range=[[-max_px, max_px], [-max_px, max_px]])

    def support(v):
        """How many displacements land in the bin containing v."""
        i = np.searchsorted(xe, v[0]) - 1
        j = np.searchsorted(ye, v[1]) - 1
        if not (0 <= i < bins and 0 <= j < bins):
            return 0.0
        return float(hist[max(0,i-1):i+2, max(0,j-1):j+2].sum())

    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    peaks = [(hist[i, j], np.array([xc[i], yc[j]]))
             for i in range(bins) for j in range(bins)
             if hist[i, j] > 0 and np.hypot(xc[i], yc[j]) > 12]
    if not peaks:
        return None
    peaks.sort(key=lambda t: -t[0])

    def fundamental(v):
        """Step down to v/2 or v/3 while the shorter vector is still supported."""
        for _ in range(3):
            for k in (2, 3):
                w = v / k
                if np.hypot(*w) > 12 and support(w) >= 0.30 * support(v):
                    v = w
                    break
            else:
                return v
        return v

    a = fundamental(peaks[0][1])
    la = np.hypot(*a)
    for _, v in peaks:
        v = fundamental(v)
        lb = np.hypot(*v)
        if abs(a[0]*v[1] - a[1]*v[0]) > 0.35 * la * lb:
            return a, v
    return None


def seed_patch(points, a, b, tol=0.18, min_hits=6, darkness=None):
    """The most grid-consistent neighbourhood, to start growing from.

    Growth propagates whatever the seed believes, so it must start somewhere the
    evidence is strong: the point whose neighbours best match the basis.
    """
    best, best_n = None, 0
    for i, p in enumerate(points):
        if darkness is not None and darkness[i] < np.median(darkness):
            continue  # seed on a convincing hole, not a faint mark or hold edge
        d = points - p
        coords = np.linalg.lstsq(np.column_stack([a, b]), d.T, rcond=None)[0].T
        near = coords[(np.abs(coords) <= 2.5).all(axis=1)]
        hits = int((np.abs(near - np.round(near)).max(axis=1) < tol).sum())
        if hits > best_n:
            best, best_n = i, hits
    return (best, best_n) if best_n >= min_hits else (None, best_n)


def grow(points, seed, a, b, snap=0.35, refit_every=6):
    """Grow the lattice hole by hole from a seed.

    Each step predicts where the next hole should be from the geometry already
    confirmed, then looks for a candidate there. Once four holes are known the
    prediction comes from a homography refitted as it goes, so perspective is
    measured rather than assumed, and the affine basis is only ever used to get
    started.

    Returns (confirmed, missing): confirmed maps integer grid coordinates to
    image points; missing lists grid coordinates that were predicted, sit
    inside the explored area, and had no hole. Those are the interesting ones,
    since something is covering them.
    """
    from wall import homography

    pts = np.asarray(points, dtype=float)
    confirmed = {(0, 0): pts[seed]}
    frontier = [(0, 0)]
    missing, H, tried = [], None, {(0, 0)}
    step = min(np.hypot(*a), np.hypot(*b))

    while frontier:
        cell = frontier.pop(0)
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + d[0], cell[1] + d[1])
            if nxt in tried:
                continue
            tried.add(nxt)

            if H is not None:  # projective prediction, valid across the frame
                q = np.array([nxt[0], nxt[1], 1.0]) @ H.T
                pred = q[:2] / q[2]
            else:              # affine fallback, only until four holes are known
                pred = confirmed[cell] + d[0] * a + d[1] * b

            dist = np.hypot(*(pts - pred).T)
            j = int(np.argmin(dist))
            if dist[j] < snap * step:
                confirmed[nxt] = pts[j]
                frontier.append(nxt)
                if len(confirmed) >= 4 and len(confirmed) % refit_every == 0:
                    cells = np.array(list(confirmed.keys()), dtype=float)
                    H = homography(cells, np.array(list(confirmed.values())))
            elif abs(nxt[0]) <= 12 and abs(nxt[1]) <= 12:
                missing.append(nxt)   # expected a hole, found none: occluded
    return confirmed, missing
