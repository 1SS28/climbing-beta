"""Which colour group is actually a route.

The largest group is usually the neutral greys shared by every route. A real
route rises, chains together within reach, and is marked in a saturated colour.

Scored in pixels and fractions so it works before scale is known.
"""

import numpy as np


def chain_fraction(points, reach):
    """Largest fraction of the group reachable by hopping hold to hold."""
    if len(points) < 2:
        return 0.0
    best = 0
    unvisited = set(range(len(points)))
    while unvisited:
        seed = unvisited.pop()
        seen, frontier = {seed}, [seed]
        while frontier:
            i = frontier.pop()
            for j in list(unvisited):
                if np.linalg.norm(points[j] - points[i]) <= reach:
                    unvisited.discard(j)
                    seen.add(j)
                    frontier.append(j)
        best = max(best, len(seen))
    return best / len(points)


def score(points, lab, image_height, min_holds=5):
    """How route-like a colour group is. Zero means it is not one.

    Three signals multiplied, so failing any one disqualifies the group.
    """
    n = len(points)
    if n < min_holds:
        return 0.0

    rise = (points[:, 1].max() - points[:, 1].min()) / image_height
    chain = chain_fraction(points, reach=0.20 * image_height)  # ~1 m on a 4.5 m wall
    chroma = float(np.hypot(lab[1], lab[2]))
    # Greys sit near zero chroma. Saturated so a strong colour cannot dominate.
    colourful = min(chroma / 30.0, 1.0)

    return float(rise * chain * colourful)


def best_group(points_by_group, labs, image_height):
    """(group index, score) of the most route-like group, or (None, 0)."""
    scored = [
        (g, score(np.asarray(p), labs[g], image_height))
        for g, p in points_by_group.items()
    ]
    scored = [s for s in scored if s[1] > 0]
    if not scored:
        return None, 0.0
    return max(scored, key=lambda s: s[1])
