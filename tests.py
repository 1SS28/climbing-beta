"""Invariants worth keeping. Run with: .venv/bin/python tests.py

No framework, no dependencies beyond the project's own. These exist because
several bugs in this project survived a long time by being plausible: a body
span that silently deadlocked every route, a reach check that chained over the
wrong set of holds, a scale that swung route connectivity from 9/10 to 2/10.
Each test below pins something that was actually wrong at some point.
"""

import numpy as np

from beta import Climber, feasible, find_start, route_chain, search
from colour import group, srgb_to_lab
from lattice import box_mean
from scene import Hold, from_polygons, from_wall_plane, polygon_area, scale_from_person, scale_from_tnuts
from wall import homography, intrinsics, plane_from_homography, to_wall_metres, wall_angle

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILS.append(name)


def ladder(n=12, step=0.35):
    return [Hold(x=0.35 if i % 2 == 0 else 0.75, y=0.3 + step * i, size=0.06) for i in range(n)]


print("colour")
lab = srgb_to_lab([[255, 255, 255], [0, 0, 0], [255, 0, 0]])
check("white is L=100, neutral", abs(lab[0][0] - 100) < 0.5 and abs(lab[0][1]) < 1 and abs(lab[0][2]) < 1)
check("black is L=0", abs(lab[1][0]) < 0.5)
check("red is warm", lab[2][1] > 50 and lab[2][2] > 30)
# Complete linkage must not chain across a continuum; single linkage did.
ramp = np.array([[50, v, 0] for v in range(0, 60, 6)], dtype=float)
check("clustering does not chain a continuum", len(np.bincount(group(ramp, 10.0))) > 1,
      "one cluster means it chained end to end")

print("\ngeometry")
check("shoelace area", abs(polygon_area([[0, 0], [2, 0], [2, 3], [0, 3]]) - 6.0) < 1e-9)
check("box_mean of a constant is constant", abs(box_mean(np.full((20, 20), 7.0), 3) - 7.0).max() < 1e-9)
check("person scale", abs(scale_from_person(1680, 1.68) - 0.001) < 1e-9)
check("t-nut scale spans holes", abs(scale_from_tnuts((0, 0), (200, 0), holes_apart=2, spacing_m=0.2) - 0.002) < 1e-9)

print("\nwall plane")
K = intrinsics(3808.0, 4284, 5712)
for tilt in (0.0, 20.0, 35.0):
    a = np.radians(tilt)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    t = np.array([-1.0, -1.5, 5.0])
    wall_xy = np.array([[0, 0], [2.0, 0], [0, 1.6], [2.0, 1.6]])
    img = np.array([(K @ (R @ np.array([X, Y, 0.0]) + t))[:2] / (K @ (R @ np.array([X, Y, 0.0]) + t))[2]
                    for X, Y in wall_xy])
    H = homography(wall_xy, img)
    got = wall_angle(plane_from_homography(H, K)[0])
    check(f"wall angle {tilt:.0f} deg", abs(got - tilt) < 0.1, f"got {got:.2f}")
    back = to_wall_metres(H, img)
    check(f"round-trip to metres at {tilt:.0f} deg", np.abs(back - wall_xy).max() < 1e-6)

print("\nbody model")
c = Climber(1.68)
check("hands above feet by a torso", c.torso_min < c.body)
# The bug that deadlocked every real route: too small a hand-to-foot span.
check("hand-foot span exceeds height", c.body > c.height,
      "a climber on a foothold reaching overhead spans more than their height")
h = ladder()
stacked = (3, 3, 3, 3)
check("all four limbs on one hold is rejected", not feasible(stacked, h, c))
s = find_start(h, c)
check("a start stance exists on a ladder", s is not None)
# Midpoints, not extremes: requiring every hand above every foot outlaws high
# steps, heel hooks and underclings, which are ordinary climbing.
if s is not None:
    torso = np.mean([h[s[0]].y, h[s[1]].y]) - np.mean([h[s[2]].y, h[s[3]].y])
    check("start torso is within reach limits", c.torso_min <= torso <= 0.95 * c.height, f"torso {torso:.2f} m")
    check("a high foot does not invalidate a stance",
          feasible((0, 1, 2, 3), [Hold(x=0.4, y=2.0, size=0.06), Hold(x=0.9, y=2.1, size=0.06),
                                  Hold(x=0.4, y=0.7, size=0.06), Hold(x=0.9, y=1.5, size=0.06)], c),
          "one foot high should still be climbable")

print("\nbeta search")
path, info = search(h, s, len(h) - 1, c=c)
check("ladder is climbable", path is not None, str(info))
if path:
    check("every step moves exactly one limb",
          all(sum(1 for k in range(4) if a[k] != b[k]) == 1 for a, b in zip(path, path[1:])))
    check("finishes with a hand on the top hold", len(h) - 1 in (path[-1][0], path[-1][1]))
    check("cost is positive", info > 0)
# A route whose top is out of reach must report, not truncate.
severed = [Hold(x=0.4, y=0.3), Hold(x=0.5, y=0.9), Hold(x=0.45, y=1.5), Hold(x=0.5, y=4.5)]
chain, why = route_chain(severed, [0, 1, 2, 3], c)
check("severed route returns None, not a stump", chain is None, "silently truncating is the bug")
chain2, _ = route_chain(severed[:3], [0, 1, 2], c)
check("connected route returns its holds", chain2 is not None and len(chain2) == 3)
# A stray hold off to the side must not join the chain.
stray = severed[:3] + [Hold(x=3.5, y=1.2)]
chain3, _ = route_chain(stray, [0, 1, 2, 3], c)
check("a stray hold is left off the chain", chain3 is not None and 3 not in chain3)

print("\nscene")
poly = [np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)]
hold = from_polygons(poly, 1000, 0.002)[0]
check("flat mapping puts y up", abs(hold.y - 1.9) < 1e-6 and abs(hold.x - 0.1) < 1e-6)
check("flat mapping sizes the hold", abs(hold.size - 0.2) < 1e-6)
Hm = homography(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float),
                np.array([[0, 100], [100, 100], [0, 0], [100, 0]], dtype=float))
wh = from_wall_plane([np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=float)], Hm)[0]
check("wall-plane mapping returns metres", 0 < wh.x < 1 and 0 < wh.y < 1)

print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
raise SystemExit(1 if FAILS else 0)
