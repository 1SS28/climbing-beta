"""Recover scale from the T-nut grid. Does not work; see README.

Gym walls are drilled on a regular lattice, so the bolt holes should be a ruler
lying in the image. Neither method here finds that lattice in real photos, at
these resolutions. Kept because the idea is an obvious one to re-try.
"""

import numpy as np

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


def sobel(g):
    """Image gradients, by shifts rather than a convolution library."""
    up, dn = np.roll(g, -1, 0), np.roll(g, 1, 0)
    lf, rt = np.roll(g, -1, 1), np.roll(g, 1, 1)
    ul, ur = np.roll(up, -1, 1), np.roll(up, 1, 1)
    dl, dr = np.roll(dn, -1, 1), np.roll(dn, 1, 1)
    gx = (ur + 2 * rt + dr) - (ul + 2 * lf + dl)
    gy = (dl + 2 * dn + dr) - (ul + 2 * up + ur)
    return gx, gy


def hough_circles(gray, radii=(6, 7, 8, 9), grad_min=18.0, vote_min=0.45):
    """Circle centres by Hough voting along gradient directions.

    A bolt hole is not merely dark, it is a dark disk with a circular rim, and
    only the rim's gradients agree on where the centre is. Chalk, wood grain and
    panel seams are dark too but their edges point every which way, so they
    scatter votes instead of piling them up. That is the part plain blob
    detection could not separate.
    """
    gx, gy = sobel(gray)
    mag = np.hypot(gx, gy)
    ys, xs = np.nonzero(mag > grad_min)
    if len(ys) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    ux, uy = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs]

    h, w = gray.shape
    best_acc = np.zeros((h, w), dtype=np.float32)
    for r in radii:
        acc = np.zeros((h, w), dtype=np.float32)
        for sign in (1, -1):  # dark-on-light or light-on-dark
            cx = np.clip((xs + sign * r * ux).astype(int), 0, w - 1)
            cy = np.clip((ys + sign * r * uy).astype(int), 0, h - 1)
            np.add.at(acc, (cy, cx), 1.0)
        acc /= (2 * np.pi * r)          # normalise: bigger circles gather more votes
        best_acc = np.maximum(best_acc, acc)

    # Separable max filter. A square max is the max over rows of the max over
    # columns, so this is 2*(2r+1) passes rather than (2r+1) squared: on a
    # 24-megapixel image that is 38 rolls instead of 361, and each roll
    # allocates ~96 MB, so the naive version thrashed memory for minutes.
    sep = max(radii)
    tmp = best_acc.copy()
    for dy in range(-sep, sep + 1):
        if dy:
            tmp = np.maximum(tmp, np.roll(best_acc, dy, 0))
    peak = tmp.copy()
    for dx in range(-sep, sep + 1):
        if dx:
            peak = np.maximum(peak, np.roll(tmp, dx, 1))
    hits = np.argwhere((best_acc >= peak) & (best_acc > vote_min))
    return hits[:, ::-1].astype(float), best_acc[hits[:, 0], hits[:, 1]]


def confirm_grid(gray, H, cells_x=(-12, 12), cells_y=(-12, 12), radius=7, depth=18.0):
    """Check a fitted grid against the image: do the predicted holes exist?

    The complement of scoring by inliers, and the more informative half. A grid
    at twice the true pitch still explains every detected hole perfectly, since
    every real hole sits on it; it simply also predicts a hole halfway between
    each pair, where there is bare wall. Fitting to detections cannot see that.
    Asking whether each predicted position has something dark at it can.

    Returns (confirmed fraction, predicted count). The depth threshold is
    calibrated against holes found by hough_circles: at 18 it fires on 61% of
    real holes and 12% of random points on the same wall, about five to one. A
    lower bar is useless here. At 6 it takes 82% of holes but also 38% of random
    points, and every grid then scores alike regardless of pitch, which is what
    made a first attempt at this look like it did not work.
    """
    from wall import to_wall_metres

    h, w = gray.shape
    smooth = box_mean(gray, 8)
    context = box_mean(gray, 26)
    hits = total = 0
    for gx in range(cells_x[0], cells_x[1] + 1):
        for gy in range(cells_y[0], cells_y[1] + 1):
            q = np.array([gx, gy, 1.0]) @ H.T
            if abs(q[2]) < 1e-9:
                continue
            x, y = q[0] / q[2], q[1] / q[2]
            if not (radius < x < w - radius and radius < y < h - radius):
                continue
            total += 1
            xi, yi = int(x), int(y)
            patch = smooth[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1]
            if patch.size and (context[yi, xi] - patch.min()) > depth:
                hits += 1
    return (hits / total if total else 0.0), total
