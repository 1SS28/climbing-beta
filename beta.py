"""Find a sequence of moves through a route.

A stance is which hold each limb is on; a move changes one limb. Finding a
sequence is therefore a shortest path over stances.
"""

import heapq
from dataclasses import dataclass

import numpy as np

LIMBS = ("LH", "RH", "LF", "RF")


@dataclass(frozen=True)
class Climber:
    """Reach limits in metres, derived from height."""

    height: float = 1.68  # 5'6"
    ape: float = 1.0  # arm span / height

    @property
    def span(self):
        return self.height * self.ape

    @property
    def body(self):
        # Foot to fingertip, fully extended. Values below ~1.2x height deadlock
        # real routes: hands cannot rise until feet do, and vice versa.
        return 1.25 * self.height

    @property
    def torso_min(self):
        return 0.25 * self.height  # crouched, hands still above feet

    @property
    def step_max(self):
        return 0.55 * self.height  # the longest single limb move, ~1 m


@dataclass
class Costs:
    """Weights on the four cost terms."""

    travel: float = 1.0
    strain: float = 2.0
    balance: float = 1.5
    quality: float = 0.3


def feasible(stance, holds, c):
    """Pairwise limits between the four contact points.

    No jointed body model: shoulder, hip and torso parameters cannot be
    recovered from a photo, so only contact-point distances are constrained.
    """
    # Cap of two limbs per hold. Without it the search stacks all four limbs on
    # one hold, which drives strain and balance to zero.
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

    # Torso length bounds, which also prevent the collapse above.
    torso = min(h[1] for h in hands) - max(f[1] for f in feet)
    if not c.torso_min <= torso <= 0.95 * c.height:
        return False

    return all(np.linalg.norm(h - f) <= c.body for h in hands for f in feet)


def stance_cost(stance, holds, c, w):
    """Static cost of standing in a stance, split into named parts."""
    p = np.array([holds[i].pos for i in stance])
    hands, feet = p[:2], p[2:]

    # Extension as a fraction of limits, squared so the limit hurts sharply.
    strain = (np.linalg.norm(hands[0] - hands[1]) / c.span) ** 2
    strain += max(np.linalg.norm(h - f) / c.body for h in hands for f in feet) ** 2

    # How far the centre of mass sits outside the feet.
    com = p.mean(axis=0)
    balance = abs(com[0] - feet[:, 0].mean())

    quality = sum(0.05 / max(holds[i].size, 0.01) for i in stance) / 4
    return w.strain * strain + w.balance * balance + w.quality * quality


def eligible(holds, hands=None, feet=None):
    """Which holds each limb may use. Hands take the route, feet take the wall.

    Restricting all four limbs to a sparse route makes most routes unclimbable.
    """
    everything = list(range(len(holds)))
    return (list(hands) if hands is not None else everything,
            list(feet) if feet is not None else everything)


def find_start(holds, c=Climber(), hands=None, feet=None):
    """Lowest stance the body fits into. Start markings are not detectable, so
    the convention is to start as low as possible."""
    hand_ok, foot_ok = eligible(holds, hands, feet)
    by_height = lambda s: sorted(s, key=lambda i: holds[i].y)  # noqa: E731
    hs, fs = by_height(hand_ok), by_height(foot_ok)
    for fi in range(len(fs)):
        for fj in range(fi + 1, len(fs)):
            ft = sorted((fs[fi], fs[fj]), key=lambda i: holds[i].x)
            for hi in range(len(hs)):
                for hj in range(hi + 1, len(hs)):
                    hd = sorted((hs[hi], hs[hj]), key=lambda i: holds[i].x)
                    stance = (hd[0], hd[1], ft[0], ft[1])
                    if feasible(stance, holds, c):
                        return stance
    return None


def search(holds, start, finish, c=Climber(), w=Costs(), hands=None, feet=None,
           max_expansions=200_000):
    """Dijkstra over stances. Returns (path, cost) or (None, reason)."""
    if not feasible(start, holds, c):
        return None, "start stance is not physically reachable"
    hand_ok, foot_ok = eligible(holds, hands, feet)
    allowed = (hand_ok, hand_ok, foot_ok, foot_ok)  # LH, RH, LF, RF

    reach = c.step_max
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
            for j in allowed[limb]:
                if j == s[limb]:
                    continue
                step = float(np.linalg.norm(holds[j].pos - here))
                if step > reach:
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
    """Explain a failure: whether the route splits, and by how much.

    Distinguishes a hard route from a colour grouping that swept up unrelated
    holds, which is the more common cause.
    """
    pos = [h.pos for h in holds]
    reached, frontier = set(start), list(start)
    while frontier:
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
