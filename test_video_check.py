"""Does the video checker catch a video that leaves the beta? Run: .venv/bin/python -m unittest test_video_check

The checker is only worth trusting if it fails a bad video, so each test hands
it a video's worth of limb positions with one known fault and checks that the
report finds it. The positions stand in for what the pose model would see;
the model itself is not what is under test.
"""
import unittest

import cv2
import numpy as np

from beta import Climber, find_start, search
from figure import Body, frames, on_photo
from measurement import Grid, wall_mapping
from test_figure import ladder
from video_check import ANKLE_TO_TOE_M, JOINT, LIMBS, ON_HOLD_M, SWAP, Register, lands, match, project, score

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


def video(tr, speed=1.0):
    """Where a perfect video would show each limb, sampled like a real one."""
    times = np.array([f["t"] for f in tr["frames"]])
    out = []
    for t in np.arange(0, tr["video"]["duration_s"] * speed, 1 / FPS):
        f = tr["frames"][np.searchsorted(times, t / speed, side="right") - 1]
        out.append((t, np.array([f["joints"][JOINT[k]] for k in LIMBS])))
    return out


def stance_times(tr):
    """When each stance is reached and when the next move leaves it."""
    keys = [f["t"] for f in tr["frames"] if f["key"]]
    return list(zip(keys, [m["start_s"] for m in tr["moves"]] + [tr["video"]["duration_s"]]))


class VideoCheckTests(unittest.TestCase):
    def setUp(self):
        self.tr, self.mapping = transcript()
        self.samples = video(self.tr)

    def check(self, samples, duration=None):
        return score(self.tr, samples, duration=duration)

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

    def test_a_slower_video_is_fitted(self):
        speed = 1.5
        report = self.check(video(self.tr, speed=speed), duration=self.tr["video"]["duration_s"] * speed)
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


    def test_someone_else_in_frame_is_not_taken_for_the_climber(self):
        floor = np.array([[500, 1550], [560, 1550], [510, 1590], [550, 1590]], float)
        report = self.check([(t, [floor, p]) for t, p in self.samples])
        self.assertTrue(all(lands(s) for s in report["stances"]))


def climbing(tr, mapping, seed=0, change=None):
    """A person climbing the beta at their own pace: out of shot, then each
    stance held for a while, with moves of their own length between."""
    rng = np.random.default_rng(seed)
    keys = [f for f in tr["frames"] if f["key"]]
    stances = [mapping.to_metres(np.array([f["joints"][k] for k in LIMBS], float)) for f in keys]
    if change:
        stances = change([s.copy() for s in stances])
    out, held, t = [], [], 0.0
    for _ in range(20):
        out.append((t, None))
        t += .1
    for k, s in enumerate(stances):
        start = t
        for _ in range(rng.integers(5, 40)):
            out.append((t, s + rng.normal(0, .015, s.shape)))
            t += .1
        held.append((start, t))
        if k + 1 < len(stances):
            for a in np.linspace(0, 1, rng.integers(3, 20), endpoint=False)[1:]:
                out.append((t, s + (stances[k + 1] - s) * a + rng.normal(0, .015, s.shape)))
                t += .1
    return [(t, None if p is None else mapping.to_pixels(p)) for t, p in out], held


class OwnPaceTests(unittest.TestCase):
    def setUp(self):
        self.tr, self.mapping = transcript()
        self.limbs = [None] + [m["limb"] for m in self.tr["moves"]]

    def test_every_stance_is_found_in_order_whatever_the_pace(self):
        for seed in range(5):
            samples, held = climbing(self.tr, self.mapping, seed)
            stances = match(self.tr, samples)["stances"]
            self.assertTrue(all(lands(s) for s in stances), seed)
            for s, (a, b) in zip(stances, held):
                self.assertTrue(a - .3 <= s["t"] <= b + .3, (seed, s["move"], s["t"], a, b))

    def test_a_different_foothold_is_reported_where_it_was(self):
        k = next(i for i, limb in enumerate(self.limbs) if limb and limb.endswith("F") and i > 1)
        j = LIMBS.index(self.limbs[k])

        def elsewhere(stances):
            beta = stances[k][j].copy()
            for s in stances[k:]:
                if not np.allclose(s[j], beta):
                    break
                s[j] += [.4, 0]
            return stances
        samples, _ = climbing(self.tr, self.mapping, change=elsewhere)
        stances = match(self.tr, samples)["stances"]
        self.assertFalse(lands(stances[k]))
        np.testing.assert_allclose(stances[k]["offset_m"][LIMBS[j]], [.4, 0], atol=.03)
        self.assertTrue(all(lands(s) for s in stances[:k]))

    def test_moves_made_in_another_order_miss_the_stance_between(self):
        k = next(i for i in range(1, len(self.limbs) - 1) if self.limbs[i] != self.limbs[i + 1])
        j = LIMBS.index(self.limbs[k + 1])

        def swapped(stances):
            between = stances[k - 1].copy()
            between[j] = stances[k + 1][j]
            stances[k] = between
            return stances
        samples, _ = climbing(self.tr, self.mapping, change=swapped)
        verdicts = [lands(s) for s in match(self.tr, samples)["stances"]]
        self.assertFalse(verdicts[k])
        self.assertTrue(all(v for i, v in enumerate(verdicts) if i != k))

    def test_someone_on_the_mat_is_not_taken_for_the_climber(self):
        samples, _ = climbing(self.tr, self.mapping)
        floor = self.mapping.to_pixels(np.array([[1.4, -.2], [1.6, -.2], [1.45, -.9], [1.55, -.9]]))
        crowded = [(t, [floor] + ([] if p is None else [p])) for t, p in samples]
        self.assertTrue(all(lands(s) for s in match(self.tr, crowded)["stances"]))

    def test_stances_reached_out_of_order_are_not_all_counted(self):
        k = 4
        samples, _ = climbing(self.tr, self.mapping,
                              change=lambda st: st[:k] + [st[k + 1], st[k]] + st[k + 2:])
        stances = match(self.tr, samples)["stances"]
        self.assertFalse(lands(stances[k]) and lands(stances[k + 1]))
        self.assertEqual([s["t"] for s in stances], sorted(s["t"] for s in stances))

    def test_limbs_the_pose_model_missed_do_not_stop_it(self):
        # It loses a wrist behind the body or an ankle past the frame often.
        rng = np.random.default_rng(3)
        samples, _ = climbing(self.tr, self.mapping)
        holes = []
        for t, p in samples:
            if p is not None:
                p = p.copy()
                p[rng.random(4) < .25] = np.nan
            holes.append((t, p))
        self.assertTrue(all(lands(s) for s in match(self.tr, holes)["stances"]))

    def test_wrists_and_ankles_are_allowed_their_distance_from_the_hold(self):
        for hand, foot, expected in ((.9, .9, True), (1.2, .9, False), (.9, 1.2, False)):
            off = np.array([[0, ON_HOLD_M * hand]] * 2 + [[0, -(ON_HOLD_M + ANKLE_TO_TOE_M) * foot]] * 2)
            samples, _ = climbing(self.tr, self.mapping, change=lambda st: [s + off for s in st])
            self.assertEqual({lands(s) for s in match(self.tr, samples)["stances"]}, {expected}, (hand, foot))


class RegisterTests(unittest.TestCase):
    @staticmethod
    def wall(seed):
        rng = np.random.default_rng(seed)
        img = np.full((900, 1200), 128, np.uint8)
        for _ in range(500):
            x, y, r = int(rng.integers(0, 1200)), int(rng.integers(0, 900)), int(rng.integers(4, 30))
            if rng.random() < .5:
                cv2.circle(img, (x, y), r, int(rng.integers(0, 256)), -1)
            else:
                cv2.rectangle(img, (x, y), (x + r, y + int(rng.integers(4, 30))), int(rng.integers(0, 256)), -1)
        return cv2.GaussianBlur(img, (3, 3), 0)

    def test_a_frame_from_nearby_lines_up_with_the_photo(self):
        photo = self.wall(1)
        # Another framing and aspect from a step to the side: photo to video.
        H = np.array([[.95, .06, -60], [-.03, .92, -40], [2e-5, 3e-5, 1]])
        frame = cv2.warpPerspective(photo, H, (1280, 720))
        found = Register(photo)(frame)
        pts = np.array([[300, 300], [800, 250], [600, 600], [400, 650]], float)
        np.testing.assert_allclose(project(found, project(H, pts)), pts, atol=1.0)

    def test_a_different_wall_does_not_line_up(self):
        self.assertIsNone(Register(self.wall(1))(self.wall(2)))

    def test_a_limb_not_seen_stays_not_seen(self):
        out = project(np.eye(3), np.array([[1, 2], [np.nan, np.nan]]))
        self.assertTrue(np.isnan(out[1]).all())
        np.testing.assert_allclose(out[0], [1, 2])


if __name__ == "__main__":
    unittest.main()
