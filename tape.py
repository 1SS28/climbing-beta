"""Find route tape, and use it rather than hold colour to define a route.

Many gyms, including the one these photos come from, mark a route with a strip
of coloured tape beside each hold rather than by setting it in one colour. Two
purple holds under pink tape are the same route; a green hold under blue tape
is a different one. Clustering hold colours cannot recover that, and on a taped
wall it produces groups that are not routes at all.

Tape is an easier target than a hold in principle. It is small, strongly
saturated, roughly rectangular, sits on bare wall rather than on a hold, and
gyms choose the colours to be told apart at a glance.

STATUS: finds tape, but not only tape. On IMG_3495 it returns 13 candidates in
6 colours, and the two strongest match the pink and blue markers visible by
eye: Lab(46, 51, 0) and Lab(38, 25, -62). It also fires on the coloured rims of
holds, because the segmentation masks sit slightly inside a hold's true edge
and the dilation here does not cover the difference, and it misses some real
strips entirely.

Worth fixing before relying on it: dilate the hold mask further, and use the
fact that tape is an oriented strip of near-uniform colour, which a hold rim is
not, since a rim curves and shades.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from colour import srgb_to_lab


def hold_mask(img, polys, dilate=9):
    """Pixels belonging to detected holds, slightly grown."""
    m = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(m)
    for poly in polys:
        if len(poly) >= 3:
            d.polygon([tuple(p) for p in poly], fill=255)
    return np.array(m.filter(ImageFilter.MaxFilter(dilate))) > 0


def saturated_blobs(img, holds, min_chroma=28.0, min_px=40, max_frac=0.0006, stride=1):
    """Connected runs of strongly coloured pixels that are not holds.

    Two-pass labelling by row runs, since scipy is not a dependency here.
    """
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    lab = srgb_to_lab(rgb.reshape(-1, 3)).reshape(h, w, 3)
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    mask = (chroma > min_chroma) & (~holds)

    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    labels = np.zeros((h, w), dtype=np.int32)
    nxt = 1
    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        xs = np.flatnonzero(row)
        breaks = np.flatnonzero(np.diff(xs) > 1)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [len(xs) - 1]])
        for s, e in zip(starts, ends):
            x0, x1 = xs[s], xs[e]
            above = labels[y - 1, x0:x1 + 1] if y else np.zeros(0, dtype=np.int32)
            touching = np.unique(above[above > 0])
            if len(touching) == 0:
                lid = nxt
                parent[lid] = lid
                nxt += 1
            else:
                lid = touching.min()
                for o in touching[1:]:
                    union(lid, o)
            labels[y, x0:x1 + 1] = lid

    if nxt == 1:
        return []
    flat = labels.ravel()
    nz = flat > 0
    roots = np.array([find(v) if v in parent else 0 for v in range(nxt)])
    flat[nz] = roots[flat[nz]]
    labels = flat.reshape(h, w)

    out = []
    max_px = max_frac * h * w
    for lid in np.unique(labels):
        if lid == 0:
            continue
        ys, xs = np.nonzero(labels == lid)
        if not (min_px <= len(ys) <= max_px):
            continue
        # Tape is a strip: longer than it is wide, and it fills its own box.
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        elong = max(bh, bw) / max(min(bh, bw), 1)
        fill = len(ys) / float(bh * bw)
        if elong < 1.8 or fill < 0.45:
            continue
        px = rgb[ys, xs]
        out.append({"centre": np.array([xs.mean(), ys.mean()]),
                    "colour": np.median(srgb_to_lab(px), axis=0),
                    "pixels": int(len(ys)), "elongation": float(elong)})
    return out


def assign(tapes, hold_centres, max_dist):
    """Give each hold the colour of its nearest tape, if one is close enough.

    Tape sits beside or just below its hold, so nearest wins. Holds with no
    tape nearby are left unassigned rather than guessed at: on a taped wall
    they are usually another route's holds, or wall furniture.
    """
    if not tapes:
        return {}
    tc = np.array([t["centre"] for t in tapes])
    owner = {}
    for i, c in enumerate(hold_centres):
        d = np.hypot(*(tc - c).T)
        j = int(np.argmin(d))
        if d[j] <= max_dist:
            owner[i] = j
    return owner
