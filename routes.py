"""Which colour group is actually a route?

Picking the largest group is wrong: on most walls that is the neutral greys and
blacks, which are shared furniture rather than a line to climb. A route has
shape — it rises, its holds chain together within reach, and gyms mark it in a
saturated colour precisely so it reads apart from the wall.

Scored in pixels and fractions, not metres, so this works before scale is known.
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
    """How much this colour group looks like a route. 0 means it is not one.

    Three signals, multiplied so that failing any one disqualifies the group:
    it must rise up the wall, hang together within reach, and be a colour a
    setter would choose to mark a line with.
    """
    n = len(points)
    if n < min_holds:
        return 0.0

    rise = (points[:, 1].max() - points[:, 1].min()) / image_height
    chain = chain_fraction(points, reach=0.20 * image_height)  # ~1 m on a 4.5 m wall
    chroma = float(np.hypot(lab[1], lab[2]))
    # Greys and near-blacks sit near zero chroma and are almost never a route on
    # their own; saturate the term so a strong colour does not dominate.
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
