"""Does the video checker catch a video that leaves the beta? Run: .venv/bin/python -m unittest test_video_check

The checker is only worth trusting if it fails a bad video, so each test hands
it a video's worth of limb positions with one known fault and checks that the
report finds it. The positions stand in for what the pose model would see;
the model itself is not what is under test.
"""
import unittest

import numpy as np

from beta import Climber, find_start, search
from figure import Body, frames, on_photo
from measurement import Grid, wall_mapping
from test_figure import ladder
from video_check import JOINT, LIMBS, ON_HOLD_M, SWAP, lands, score

SIZE = (1200, 1600)
GRID = {"method": "grid", "corners": [[100, 100], [1100, 150], [80, 1550], [1120, 1500]],
        "cols": 10, "rows": 25, "spacing_m": 0.2}
FPS = 30


def transcript():
    """A beta as the app would save it, timed the way the player plays it."""
    holds = ladder()
    c = Climber(1.68)
    path, _ = search(holds, find_start(holds, c), len(holds) - 1, c=c)
    mapping = wall_mapping(Grid(**GRID), SIZE)
    shown = on_photo(frames(path, holds, Body(1.68)), mapping.to_pixels, Body(1.68))
    t = [0.0, 0.3]
    for f in shown[1:-1]:
        t.append(t[-1] + (0.55 if f["key"] else 0.045))
    for f, s in zip(shown, t):
        f["t"] = s
    moves = []
    for k in range(1, len(path)):
        mine = [f for f in shown if f["move"] == k]
        moves.append({"n": k, "limb": mine[0]["moving"], "start_s": mine[0]["t"], "end_s": mine[-1]["t"],
                      "text": f"move {k}"})
    return {"app": "climbing-beta", "kind": "beta-transcript", "version": 1,
            "video": {"width": SIZE[0], "height": SIZE[1], "duration_s": t[-1] + 1.0},
            "size": list(SIZE), "calibration": GRID, "moves": moves, "frames": shown}, mapping


def video(tr, speed=1.0, scale=1.0):
    """Where a perfect video would show each limb, sampled like a real one."""
    times = np.array([f["t"] for f in tr["frames"]])
    out = []
    for t in np.arange(0, tr["video"]["duration_s"] * speed, 1 / FPS):
        f = tr["frames"][np.searchsorted(times, t / speed, side="right") - 1]
        out.append((t, np.array([f["joints"][JOINT[k]] for k in LIMBS]) * scale))
    return out


def stance_times(tr):
    """When each stance is reached and when the next move leaves it."""
    keys = [f["t"] for f in tr["frames"] if f["key"]]
    return list(zip(keys, [m["start_s"] for m in tr["moves"]] + [tr["video"]["duration_s"]]))


class VideoCheckTests(unittest.TestCase):
    def setUp(self):
        self.tr, self.mapping = transcript()
        self.samples = video(self.tr)

    def check(self, samples, size=SIZE, duration=None):
        return score(self.tr, samples, size, duration=duration)

    def test_a_faithful_video_lands_every_stance(self):
        report = self.check(self.samples)
        self.assertTrue(all(lands(s) for s in report["stances"]))
        self.assertLess(np.nanmax(report["error_m"]), 1e-6)
        self.assertEqual(report["swapped"], 0)

    def test_an_offset_is_measured_on_the_wall(self):
        shifted = [(t, self.mapping.to_pixels(self.mapping.to_metres(p) + [0.10, 0])) for t, p in self.samples]
        err = self.check(shifted)["error_m"]
        np.testing.assert_allclose(err, 0.10, atol=1e-6)

    def test_a_hand_that_never_arrives_is_a_miss(self):
        moves = self.tr["moves"]
        k = next(i for i, m in enumerate(moves, start=1) if m["limb"] == "RH" and i > 1)
        j = LIMBS.index("RH")
        before = self.tr["frames"][[i for i, f in enumerate(self.tr["frames"]) if f["key"]][k - 1]]
        stay = np.array(before["joints"]["RH"])
        start, end = stance_times(self.tr)[k]
        lagging = []
        for t, p in self.samples:
            p = p.copy()
            if moves[k - 1]["start_s"] <= t < end:
                p[j] = stay
            lagging.append((t, p))
        verdicts = [lands(s) for s in self.check(lagging)["stances"]]
        self.assertFalse(verdicts[k])
        self.assertTrue(all(v for i, v in enumerate(verdicts) if i != k))

    def test_left_and_right_swapped_are_caught(self):
        report = self.check([(t, p[SWAP]) for t, p in self.samples])
        self.assertGreater(report["swapped"], len(self.samples) // 2)
        # Swapped, a limb still lands where its pair shares the hold, as both
        # hands do on the finish; everywhere else it misses.
        keys = [f for f in self.tr["frames"] if f["key"]]
        for f, s in zip(keys, report["stances"]):
            at = self.mapping.to_metres(np.array([f["joints"][JOINT[k]] for k in LIMBS]))
            apart = np.linalg.norm(at - at[SWAP], axis=1)
            limbs = [LIMBS.index(s["limb"])] if s["limb"] else range(4)
            if max(apart[j] for j in limbs) > 2 * ON_HOLD_M:
                self.assertFalse(lands(s), s["move"])
        self.assertFalse(lands(report["stances"][0]))

    def test_a_still_limb_that_slides_is_reported(self):
        k = 3
        move = self.tr["moves"][k - 1]
        j = next(i for i, limb in enumerate(LIMBS) if limb != move["limb"])
        slid = []
        for t, p in self.samples:
            p = p.copy()
            if move["start_s"] <= t < move["end_s"]:
                p[j] = self.mapping.to_pixels(self.mapping.to_metres(p[j:j + 1]) + [0, 0.3])[0]
            slid.append((t, p))
        stances = self.check(slid)["stances"]
        self.assertGreater(stances[k]["drift_m"], 0.29)
        self.assertLess(max(s["drift_m"] for i, s in enumerate(stances) if i and i != k), 1e-6)

    def test_a_slower_smaller_video_is_fitted(self):
        speed, scale = 1.5, 0.5
        samples = video(self.tr, speed=speed, scale=scale)
        report = self.check(samples, size=(SIZE[0] * scale, SIZE[1] * scale),
                            duration=self.tr["video"]["duration_s"] * speed)
        self.assertAlmostEqual(report["stretch"], speed)
        self.assertTrue(all(lands(s) for s in report["stances"]))

    def test_frames_without_a_climber_are_not_counted_as_landing(self):
        start, end = stance_times(self.tr)[2]
        gone = [(t, None if start <= t < end else p) for t, p in self.samples]
        report = self.check(gone)
        self.assertLess(report["found"], len(self.samples))
        self.assertIsNone(lands(report["stances"][2]))
        self.assertTrue(lands(report["stances"][1]))

    def test_the_tolerance_is_a_hand_and_a_hold(self):
        # Inside it a stance lands; past it, it does not.
        for off, expected in ((ON_HOLD_M * 0.9, True), (ON_HOLD_M * 1.1, False)):
            moved = [(t, self.mapping.to_pixels(self.mapping.to_metres(p) + [0, off])) for t, p in self.samples]
            self.assertEqual({lands(s) for s in self.check(moved)["stances"]}, {expected})


if __name__ == "__main__":
    unittest.main()
