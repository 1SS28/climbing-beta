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

    # Torso runs shoulders to hips, so compare the hand and foot midpoints
    # rather than their extremes. Requiring every hand above every foot forbids
    # high steps, heel hooks and underclings, which are ordinary climbing, and
    # it blocked 7 of 10 real routes.
    torso = np.mean([h[1] for h in hands]) - np.mean([f[1] for f in feet])
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


def route_chain(holds, route, c=Climber()):
    """The holds forming a path from the lowest hold of a route to the highest.

    A route is a line from the ground to the top, so the top hold is the goal,
    not a candidate for removal. Breadth-first from the lowest hold over steps
    the climber can make: holds off the path (strays the colour grouping swept
    in from elsewhere) are simply never visited.

    Returns (chain, reason). chain is None when nothing connects the bottom to
    the top, which is the honest answer. Taking the largest connected component
    instead discards the top hold whenever the gap is near it, then reports a
    beta for a truncated route.
    """
    if len(route) < 2:
        return (list(route), "") if route else (None, "route is empty")

    lo = min(route, key=lambda i: holds[i].y)
    hi = max(route, key=lambda i: holds[i].y)

    prev, frontier = {lo: None}, [lo]
    while frontier:
        nxt = []
        for i in frontier:
            for j in route:
                if j not in prev and np.linalg.norm(holds[j].pos - holds[i].pos) <= c.step_max:
                    prev[j] = i
                    nxt.append(j)
        frontier = nxt

    if hi not in prev:
        reached = list(prev)
        gap = min(
            np.linalg.norm(holds[j].pos - holds[i].pos)
            for i in reached
            for j in route
            if j not in prev
        )
        return None, (
            f"no path from the lowest hold to the highest: {len(reached)}/{len(route)} "
            f"reachable, nearest gap {gap:.2f} m against a {c.step_max:.2f} m limit"
        )

    chain, node = [], hi
    while node is not None:
        chain.append(node)
        node = prev[node]
    return chain[::-1], ""


def eligible(holds, hands=None, feet=None):
    """Which holds each limb may use. Hands take the route, feet take the wall.

    Restricting all four limbs to a sparse route makes most routes unclimbable.
    """
    everything = list(range(len(holds)))
    return (list(hands) if hands is not None else everything,
            list(feet) if feet is not None else everything)


def find_start(holds, c=Climber(), hands=None, feet=None):
    """Feet low, hands as high as the body allows.

    Not the lowest stance overall, which is what this used to return and which
    is nearly immobile. On a real wall that gave a crouch with a hand and a foot
    sharing a hold at ankle height: 18 stances were reachable from it, the
    highest hand got 1.59 m up a 4.13 m wall, and 81 holds sat above that. A
    climber pulls on with hands high and feet low, and starting that way leaves
    the search somewhere to go.
    """
    hand_ok, foot_ok = eligible(holds, hands, feet)
    low = sorted(foot_ok, key=lambda i: holds[i].y)
    high = sorted(hand_ok, key=lambda i: -holds[i].y)
    for fi in range(len(low)):
        for fj in range(fi + 1, len(low)):
            ft = sorted((low[fi], low[fj]), key=lambda i: holds[i].x)
            for hi in range(len(high)):
                for hj in range(hi + 1, len(high)):
                    hd = sorted((high[hi], high[hj]), key=lambda i: holds[i].x)
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

    return None, "no sequence reaches the finish hold. " + reach_gap(holds, start, finish, c, hand_ok)


def reach_gap(holds, start, finish, c, hand_ok=None):
    """Explain a failure: whether the route splits, and by how much.

    Chains over hand-eligible holds only. Chaining over every hold answers a
    different question and reports a connected route when the one the hands may
    actually use is severed.
    """
    pool = list(hand_ok) if hand_ok is not None else list(range(len(holds)))
    pos = [h.pos for h in holds]
    reached = {i for i in start if i in pool} or {start[0]}
    frontier = list(reached)
    while frontier:
        i = frontier.pop()
        for j in pool:
            if j not in reached and np.linalg.norm(pos[j] - pos[i]) <= c.step_max:
                reached.add(j)
                frontier.append(j)

    if finish in reached:
        return "route holds are chainable, so the body constraints are what block it."
    outside = [j for j in pool if j not in reached]
    if not outside:
        return "every route hold is chainable; the finish hold is not among them."
    gap = min(np.linalg.norm(pos[j] - pos[i]) for i in reached for j in outside)
    return (
        f"the route splits: {len(reached)}/{len(pool)} route holds chainable from the start, "
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
