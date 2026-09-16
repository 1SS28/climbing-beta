"""V2: what colour is each hold, and which holds share a colour.

Written out by hand rather than pulled from a colour library — the conversion is
short, and the project is partly an excuse to know how it works.
"""

import numpy as np

# sRGB D65 -> XYZ, and the CIE epsilon/kappa thresholds for the L* transfer.
_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb):
    """(N,3) uint8 sRGB -> (N,3) CIELAB.

    CIELAB because Euclidean distance in it tracks perceived colour difference,
    which plain RGB does not: two RGB triples the same distance apart can look
    identical or obviously different.
    """
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)  # undo gamma
    xyz = (c @ _M.T) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def hold_colour(pixels):
    """One robust colour for a hold, from the pixels inside its mask.

    The median, not the mean: a mask leaks a little wall at its edges and picks
    up chalk and specular highlights in its middle, and a mean drags toward
    those while a median ignores them.
    """
    return np.median(srgb_to_lab(pixels), axis=0)


def group(labs, threshold=18.0):
    """Cluster holds by colour, complete linkage. Returns a label per hold.

    Distance is taken on a* and b* only, dropping L*: the same hold reads much
    darker in shadow or under a light, but its hue barely moves.

    Complete linkage, after single linkage failed outright. Hold colours form a
    continuum — orange runs into yellow runs into pink — so joining clusters
    whose *nearest* members are close chains straight through those gaps: on a
    76-hold wall with obviously distinct routes it merged 43 of them even at a
    threshold of 5. Requiring every pair in a merged cluster to be within the
    threshold gives the gap no foothold.

    The number of routes on a wall is unknown, which is why this is threshold-
    based rather than k-means.
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
                worst = dist[np.ix_(clusters[i], clusters[j])].max()  # complete linkage
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
