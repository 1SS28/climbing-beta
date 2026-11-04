"""Is the animation the beta the search chose? Run: .venv/bin/python -m unittest test_figure

An overlay that looks right and is not is worse than none, so each property a
viewer would take for granted is checked rather than assumed.
"""
import unittest

import numpy as np

from beta import Climber, find_start, search
from figure import BEYOND, Body, LIMBS, RIGID, frames, on_photo
from measurement import Grid, wall_mapping
from scene import Hold

MM = 1e-4   # a tenth of a millimetre


def ladder(n=12, step=.35):
    return [Hold(x=.35 if i % 2 == 0 else .75, y=.3 + step * i, size=.06) for i in range(n)]


def climb(holds, height=1.68):
    c = Climber(height)
    path, _ = search(holds, find_start(holds, c), len(holds) - 1, c=c)
    return path, frames(path, holds, Body(height))


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.holds = ladder()
        cls.path, cls.frames = climb(cls.holds)

    def test_every_stance_is_held_exactly(self):
        keys = [f for f in self.frames if f["key"]]
        self.assertEqual(len(keys), len(self.path))
        for f, stance in zip(keys, self.path):
            for limb, i in zip(LIMBS, stance):
                np.testing.assert_allclose(f["joints"][limb], self.holds[i].pos[:2], atol=1e-12)

    def test_only_the_moving_limb_moves(self):
        for k, (a, b) in enumerate(zip(self.path, self.path[1:]), start=1):
            moving = next(LIMBS[i] for i in range(4) if a[i] != b[i])
            for f in (f for f in self.frames if f["move"] == k):
                self.assertEqual(f["moving"], moving)
                for limb, i in zip(LIMBS, a):
                    if limb != moving:
                        np.testing.assert_allclose(f["joints"][limb], self.holds[i].pos[:2], atol=1e-12)

    def test_bones_keep_their_length(self):
        # In three dimensions: elbows and knees bend partly out of the wall.
        def at(f, n):
            return np.append(f["joints"][n], f["depth"].get(n, 0.0))
        for a, b in RIGID:
            lengths = [np.linalg.norm(at(f, a) - at(f, b)) for f in self.frames]
            self.assertLess(max(lengths) - min(lengths), MM, (a, b))

    def test_bends_go_partly_into_depth(self):
        # Flat, every bend was drawn sideways; the bent limbs must leave the wall.
        bent = [z for f in self.frames for z in f["depth"].values() if z > 0]
        self.assertGreater(len(bent), len(self.frames))

    def test_shrug_and_toe_point_stay_within_their_limits(self):
        body = Body(1.68)
        rest = np.hypot(body.torso, body.shoulder - body.hip)   # hip to shoulder, standing still
        for f in self.frames:
            for side, foot in (("l", "LF"), ("r", "RF")):
                toe = np.linalg.norm(f["joints"][f"ankle_{side}"] - f["joints"][foot])
                self.assertLessEqual(toe, body.slack(foot) + MM)
                flank = np.linalg.norm(f["joints"][f"shoulder_{side}"] - f["joints"][f"hip_{side}"])
                self.assertLessEqual(abs(flank - rest), body.slack("LH") + MM)

    def test_a_climbable_ladder_is_shown_within_reach(self):
        self.assertLessEqual(max(max(f["reach"].values()) for f in self.frames), BEYOND)

    def test_a_stance_no_body_can_hold_is_flagged_not_hidden(self):
        # Hands 2.6 m above the feet: past anything a 1.68 m climber can span.
        holds = [Hold(x=.3, y=0), Hold(x=.7, y=0), Hold(x=.3, y=2.6), Hold(x=.7, y=2.6)]
        f = frames([(2, 3, 0, 1)], holds, Body(1.68))[0]
        self.assertGreater(max(f["reach"].values()), BEYOND)
        for limb, i in zip(LIMBS, (2, 3, 0, 1)):
            np.testing.assert_allclose(f["joints"][limb], holds[i].pos[:2], atol=1e-12)

    def test_the_body_moves_without_jumping(self):
        hips = np.array([(f["joints"]["hip_l"] + f["joints"]["hip_r"]) / 2 for f in self.frames])
        self.assertLess(np.linalg.norm(np.diff(hips, axis=0), axis=1).max(), .1)
        for f in self.frames:   # upright enough that the head stays above the hips
            self.assertGreater(f["joints"]["head"][1], (f["joints"]["hip_l"][1] + f["joints"]["hip_r"][1]) / 2)

    def test_the_same_beta_always_animates_the_same_way(self):
        _, again = climb(self.holds)
        for a, b in zip(self.frames, again):
            for n in a["joints"]:
                np.testing.assert_array_equal(a["joints"][n], b["joints"][n])

    def test_drawn_with_the_perspective_of_the_holds(self):
        # A wall seen off-square: the figure must land on the holds as drawn.
        H = np.array([[300, 50, 100], [20, -300, 800], [.15, .05, 1]])
        def project(p):
            q = np.c_[p, np.ones(len(p))] @ H.T
            return (q[:, :2] / q[:, 2:]).tolist()
        mapping = wall_mapping(Grid(method="grid", corners=project(np.array([[0, 2], [2, 2], [0, 0], [2, 0]])),
                                    cols=10, rows=10, spacing_m=.2), (1000, 1000))
        holds = ladder(n=6)
        path, wall = climb(holds)
        photo = on_photo(wall, mapping.to_pixels, Body(1.68))
        for f, w in zip(photo, wall):
            for n, px in f["joints"].items():
                np.testing.assert_allclose(mapping.to_metres([px])[0], w["joints"][n], atol=MM)
        for f, stance in zip((f for f in photo if f["key"]), path):
            for limb, i in zip(LIMBS, stance):
                np.testing.assert_allclose(f["joints"][limb], mapping.to_pixels(holds[i].pos[:2])[0], atol=.01)


if __name__ == "__main__":
    unittest.main()
