"""V3: find a plausible sequence of moves through a route.

A stance is which hold each of the four limbs is on. A move changes exactly one
limb. That makes the problem a shortest path over stances, so it is Dijkstra
with a heuristic — no training data, and every move can be explained by the cost
that chose it, which matters more than the move being the one a strong climber
would pick. There is no ground truth for beta; an interpretable answer is worth
more than a confident one.
"""

import heapq
from dataclasses import dataclass

import numpy as np

LIMBS = ("LH", "RH", "LF", "RF")


@dataclass(frozen=True)
class Climber:
    """Reach limits, in metres, derived from height. Beta is personal — a 155 cm
    climber and a 190 cm climber genuinely do different moves — so this is the
    one knob that has to exist from the start."""

    height: float = 1.75
    ape: float = 1.0  # arm span / height

    @property
    def span(self):
        return self.height * self.ape

    @property
    def body(self):
        return 0.9 * self.height  # furthest a hand and a foot can sensibly be

    @property
    def torso_min(self):
        return 0.34 * self.height  # hands this far above feet, at least

    @property
    def step_max(self):
        return 0.55 * self.height  # the longest single limb move, ~1 m


@dataclass
class Costs:
    """Weights on the four terms. Exposed because tuning these *is* the model —
    there is nothing learned here to hide behind."""

    travel: float = 1.0
    strain: float = 2.0
    balance: float = 1.5
    quality: float = 0.3


def feasible(stance, holds, c):
    """Pairwise limits only — no invented torso.

    A jointed body model would need shoulder, hip and torso parameters that
    cannot be recovered from a photo, and each invented number would be one more
    thing quietly deciding the beta. Distances between the four contact points
    are the most that can honestly be constrained.
    """
    # At most two limbs share a hold — matching hands, or a hand and a foot.
    # Without this the search finds a degenerate "solution" that stacks all four
    # limbs on one hold and shuffles up the wall, since coincident contact points
    # drive both strain and balance to zero.
    counts = {}
    for i in stance:
        counts[i] = counts.get(i, 0) + 1
        if counts[i] > 2:
            return False

    p = [holds[i].pos for i in stance]
    hands, feet = p[:2], p[2:]
    if np.linalg.norm(hands[0] - hands[1]) > c.span:
        return False
    if np.linalg.norm(feet[0] - feet[1]) > 0.8 * c.span:
        return False

    # The torso has a length: hands are well above feet, and not further than
    # the body can stretch. This is what stops the collapse above.
    torso = min(h[1] for h in hands) - max(f[1] for f in feet)
    if not c.torso_min <= torso <= 0.95 * c.height:
        return False

    return all(np.linalg.norm(h - f) <= c.body for h in hands for f in feet)


def stance_cost(stance, holds, c, w):
    """Static cost of standing in a stance, split into named parts."""
    p = np.array([holds[i].pos for i in stance])
    hands, feet = p[:2], p[2:]

    # How extended the body is, as a fraction of its limits. Squared so that
    # being near the limit hurts sharply rather than linearly.
    strain = (np.linalg.norm(hands[0] - hands[1]) / c.span) ** 2
    strain += max(np.linalg.norm(h - f) / c.body for h in hands for f in feet) ** 2

    # Balance: how far the centre of mass sits outside the feet. On a slab this
    # is what actually decides whether a move works.
    com = p.mean(axis=0)
    balance = abs(com[0] - feet[:, 0].mean())

    quality = sum(0.05 / max(holds[i].size, 0.01) for i in stance) / 4
    return w.strain * strain + w.balance * balance + w.quality * quality


def find_start(holds, c=Climber()):
    """Lowest stance the body actually fits into. Gyms mark the start holds and
    we cannot detect that marking, so the convention here is simply: start as
    low as possible. Left/right are assigned by x so the limbs are not crossed."""
    order = sorted(range(len(holds)), key=lambda i: holds[i].y)
    for fi in range(len(order)):
        for fj in range(fi + 1, len(order)):
            feet = sorted((order[fi], order[fj]), key=lambda i: holds[i].x)
            for hi in range(len(order)):
                for hj in range(hi + 1, len(order)):
                    hands = sorted((order[hi], order[hj]), key=lambda i: holds[i].x)
                    stance = (hands[0], hands[1], feet[0], feet[1])
                    if feasible(stance, holds, c):
                        return stance
    return None


def search(holds, start, finish, c=Climber(), w=Costs(), max_expansions=200_000):
    """Dijkstra over stances. Returns (path, cost) or (None, reason)."""
    if not feasible(start, holds, c):
        return None, "start stance is not physically reachable"

    reach = c.step_max  # one move repositions one limb; it is not a teleport
    dist = {start: 0.0}
    prev = {}
    queue = [(0.0, start)]
    seen = set()
    expansions = 0

    while queue:
        d, s = heapq.heappop(queue)
        if s in seen:
            continue
        seen.add(s)

        if s[0] == finish or s[1] == finish:  # a hand on the finish hold ends it
            path = [s]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1], d

        expansions += 1
        if expansions > max_expansions:
            return None, f"gave up after {max_expansions} stances"

        for limb in range(4):
            here = holds[s[limb]].pos
            for j in range(len(holds)):
                if j == s[limb]:
                    continue
                step = float(np.linalg.norm(holds[j].pos - here))
                if step > reach:  # prune early; most holds are out of range
                    continue
                nxt = tuple(j if k == limb else s[k] for k in range(4))
                if not feasible(nxt, holds, c):
                    continue
                nd = d + w.travel * step + stance_cost(nxt, holds, c, w)
                if nd < dist.get(nxt, np.inf):
                    dist[nxt] = nd
                    prev[nxt] = s
                    heapq.heappush(queue, (nd, nxt))

    return None, "no sequence reaches the finish hold. " + reach_gap(holds, start, finish, c)


def reach_gap(holds, start, finish, c):
    """Explain a failure: how far apart the route actually is.

    "No sequence" is useless on its own — it cannot distinguish a route that is
    merely hard from a colour grouping that has swept up unrelated holds, which
    is the far more common cause.
    """
    pos = [h.pos for h in holds]
    reached, frontier = set(start), list(start)
    while frontier:  # holds chainable from the start within one limb move
        i = frontier.pop()
        for j in range(len(holds)):
            if j not in reached and np.linalg.norm(pos[j] - pos[i]) <= c.step_max:
                reached.add(j)
                frontier.append(j)

    if finish in reached:
        return "holds are chainable, so the body constraints are what block it."
    outside = [j for j in range(len(holds)) if j not in reached]
    gap = min(np.linalg.norm(pos[j] - pos[i]) for i in reached for j in outside)
    return (
        f"the route splits: {len(reached)}/{len(holds)} holds chainable from the start, "
        f"and the nearest gap across is {gap:.2f} m against a {c.step_max:.2f} m limit."
    )


def describe(path, holds, c=Climber(), w=Costs()):
    """One line per move, naming the limb and what the move cost."""
    out = []
    for a, b in zip(path, path[1:]):
        limb = next(i for i in range(4) if a[i] != b[i])
        step = float(np.linalg.norm(holds[b[limb]].pos - holds[a[limb]].pos))
        out.append(
            f"{LIMBS[limb]} -> hold {b[limb]:2d}  "
            f"move {step:4.2f}m  stance {stance_cost(b, holds, c, w):5.2f}"
        )
    return out
