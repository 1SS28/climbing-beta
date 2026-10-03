"""Hold colour, and which holds share one."""

import numpy as np

# sRGB D65 -> XYZ, and the CIE epsilon/kappa thresholds for the L* transfer.
_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """(N,3) uint8 sRGB -> (N,3) CIELAB.

    Euclidean distance in CIELAB tracks perceived difference; in RGB it does not.
    """
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)  # undo gamma
    xyz = (c @ _M.T) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def hold_colour(pixels):
    """One colour for a hold, from the pixels inside its mask.

    Median rather than mean: masks leak wall at the edges and chalk in the
    middle, and a mean drags toward both.
    """
    return np.median(srgb_to_lab(pixels), axis=0)


def group(labs, threshold=18.0):
    """Cluster holds by colour. Returns a label per hold.

    Distance uses a* and b* only. Dropping L* means shadow does not split a
    route, since a shaded hold keeps its hue.

    Complete linkage, because single linkage chains through the continuum from
    orange to yellow to pink: it merged 43 of 76 distinct holds at a threshold
    of 5. Threshold-based rather than k-means, since the number of routes on a
    wall is unknown.
    """
    labs = np.asarray(labs)
    if len(labs) == 0:
        return np.array([], dtype=int)

    ab = labs[:, 1:]
    dist = np.linalg.norm(ab[:, None, :] - ab[None, :, :], axis=2)

    clusters = [[i] for i in range(len(labs))]
    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                worst = dist[np.ix_(clusters[i], clusters[j])].max()
                if worst <= threshold and (best is None or worst < best[0]):
                    best = (worst, i, j)
        if best is None:
            break
        _, i, j = best
        clusters[i] += clusters.pop(j)

    out = np.empty(len(labs), dtype=int)
    for k, members in enumerate(clusters):
        out[members] = k
    return out
