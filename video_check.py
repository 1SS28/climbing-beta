"""Does a video keep to the beta?

    .venv/bin/python video_check.py climbing-beta-*.json video.mp4 [--own-pace] [--draw out.mp4]

Save as video in the app writes two files: the video, and a transcript of the
beta with the photo it was found on. This finds the climber in a video of
that beta and reports, in centimetres on the wall, how far each hand and foot
was from where the beta put it, and whether each stance was reached.

Two kinds of video need two ways of comparing. One a video model made from
the app's video keeps its timing, so each moment is held to the same moment of
the beta. One of a person climbing does not, so --own-pace finds, in order, the
moment that best fits each stance of the beta, however long the climber took
between them.

Each frame is lined up with the photo by matching the wall between them, so
the video only has to be filmed from about the same spot, not framed the same.
The climber is found with an Ultralytics pose model, which gives wrists and
ankles: a hand is judged at the wrist, a few centimetres off the grip, and a
foot at the ankle, further again from the toe on the hold.
"""

import argparse
import base64
import json
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pydantic import TypeAdapter

from measurement import Calibration, wall_mapping

LIMBS = ("LH", "RH", "LF", "RF")
NAME = {"LH": "left hand", "RH": "right hand", "LF": "left foot", "RF": "right foot"}
JOINT = {"LH": "LH", "RH": "RH", "LF": "ankle_l", "RF": "ankle_r"}   # what the figure has to compare
KEYPOINT = {"LH": 9, "RH": 10, "LF": 15, "RF": 16}                    # COCO wrists and ankles
SWAP = [1, 0, 3, 2]
SEEN = 0.5          # pose confidence below which a limb counts as not seen
ON_HOLD_M = 0.15    # wrist to grip plus a hold's size
ANKLE_TO_TOE_M = 0.10
RETIME = 0.02       # a video this much longer or shorter than the transcript is stretched to fit
CAP_M = 0.5         # past this, how far off a limb is no longer tells stances apart
SMOOTH = 5          # samples in the running median that steadies the pose model
STILL = 0.3         # cost per m/s of the fastest limb: a stance is a moment the limbs are placed
OWN_PACE_FPS = 10
REGISTER_S = 1.0    # how often a handheld video is lined up with the photo again
MIN_MATCHES = 30


def load(path):
    tr = json.loads(Path(path).read_text())
    if tr.get("app") != "climbing-beta" or tr.get("kind") != "beta-transcript" or tr.get("version") != 1:
        raise ValueError(f"{path}: not a transcript saved with a beta video")
    if any(f["t"] is None for f in tr["frames"]):
        raise ValueError(f"{path}: some frames were never shown while recording")
    return tr


class Wall(NamedTuple):
    to_metres: object
    to_pixels: object


def mapping_of(tr):
    """The transcript's calibration, for limbs anywhere in the video.

    The app's own mapping refuses points it cannot place, which is right for
    a tap but not for a wrist the pose model did not see, or one past the
    edge of the photo in a video framed wider. Both calibrations are a
    homography between photo and wall, so it is rebuilt here as one.
    """
    m = wall_mapping(TypeAdapter(Calibration).validate_python(tr["calibration"]), tuple(tr["size"]))
    wall = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], float)
    H = homography(m.to_pixels(wall), wall)
    return Wall(lambda pts: project(H, pts), m.to_pixels)


def homography(src, dst):
    """The exact homography taking four points to four others."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.append(np.linalg.solve(np.array(A), np.array(b)), 1).reshape(3, 3)


def people(entry):
    """A sample's climbers: None for nobody, one (4, 2) array, or a list of them."""
    if entry is None:
        return []
    if isinstance(entry, np.ndarray) and entry.ndim == 2:
        return [entry]
    return list(entry)


def project(H, pts):
    """Points through a homography, keeping NaN for limbs not seen."""
    pts = np.atleast_2d(np.asarray(pts, float))
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:]


def mean_off(a, b):
    d = np.linalg.norm(a - b, axis=-1)
    return np.nanmean(np.where(np.isfinite(d), np.minimum(d, CAP_M), CAP_M), axis=-1)


def score(tr, samples, duration=None):
    """Hold each sample to the same moment of the beta.

    samples are (time, climbers) with each climber a (4, 2) array of LH, RH,
    LF, RF in photo pixels, NaN where a limb was not seen.
    """
    mapping = mapping_of(tr)
    times = np.array([f["t"] for f in tr["frames"]])
    ref = np.array([[f["joints"][JOINT[k]] for k in LIMBS] for f in tr["frames"]], float)
    ref_m = mapping.to_metres(ref.reshape(-1, 2)).reshape(ref.shape)

    # A video model may not keep the length; spread the beta over what it made.
    length = tr["video"]["duration_s"]
    stretch = duration / length if duration and abs(duration / length - 1) > RETIME else 1.0

    t = np.array([s[0] for s in samples]) / stretch
    which = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1)
    err = np.full((len(samples), 4), np.nan)
    swapped = np.full(len(samples), False)
    found = np.full(len(samples), False)
    for i, (_, entry) in enumerate(samples):
        want = ref_m[which[i]]
        seen = [m for m in (mapping.to_metres(np.asarray(p, float)) for p in people(entry)) if np.isfinite(m).any()]
        if not seen:
            continue
        got = min(seen, key=lambda m: mean_off(m, want))   # whoever is nearest the beta
        found[i] = True
        err[i] = np.linalg.norm(got - want, axis=1)
        crossed = np.linalg.norm(got[SWAP] - want, axis=1)
        both = np.isfinite(err[i]) & np.isfinite(crossed)
        swapped[i] = both.any() and crossed[both].mean() + 0.1 < err[i][both].mean()

    # A stance is judged while it is held: from the frame it lands on to the
    # start of the next move.
    keys = [i for i, f in enumerate(tr["frames"]) if f["key"]]
    starts = [m["start_s"] for m in tr["moves"]] + [length]
    stances = []
    for k, key in enumerate(keys):
        held = (t >= times[key]) & (t < starts[k]) & found
        median = [float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else None
                  for e in (err[held, j] for j in range(4))]
        moving = tr["moves"][k - 1]["limb"] if k else None
        moved = (t >= tr["moves"][k - 1]["start_s"]) & (t < times[key]) & found if k else None
        drift = None
        if k:
            still = [err[moved, j] for j, limb in enumerate(LIMBS) if limb != moving]
            seen = [np.median(e[np.isfinite(e)]) for e in still if np.isfinite(e).any()]
            drift = float(max(seen)) if seen else None
        stances.append({"move": k, "limb": moving, "error_m": dict(zip(LIMBS, median)),
                        "judge": [moving] if moving else list(LIMBS),
                        "tolerance": dict.fromkeys(LIMBS, ON_HOLD_M), "drift_m": drift})

    return {"samples": len(samples), "found": int(found.sum()), "stretch": stretch,
            "error_m": err, "swapped": int(swapped.sum()), "stances": stances}


def smooth(track):
    """A running median over time, per limb, ignoring frames it was not seen."""
    half = SMOOTH // 2
    padded = np.pad(track, ((half, half), (0, 0), (0, 0)), constant_values=np.nan)
    windows = np.stack([padded[i:i + len(track)] for i in range(SMOOTH)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=0)


def match(tr, samples):
    """Find each stance of the beta, in order, at the climber's own pace.

    A person climbing pauses, rests and moves at their own speed, so the time
    of a stance says nothing. What is kept is order: every stance is given the
    sample that fits it best, later than the stance before, and the whole set
    is chosen together so one good fit cannot steal another's moment.
    """
    mapping = mapping_of(tr)
    keys = [f for f in tr["frames"] if f["key"]]
    want = mapping.to_metres(np.array([[f["joints"][k] for k in LIMBS] for f in keys], float)
                             .reshape(-1, 2)).reshape(len(keys), 4, 2)
    if len(samples) < len(keys):
        raise ValueError(f"{len(samples)} frames is too few for {len(keys)} stances")

    # Of everyone in a frame, the climber is whoever looks most like a stance
    # of this beta; someone standing on the mat looks like none of them.
    track = np.full((len(samples), 4, 2), np.nan)
    found = np.full(len(samples), False)
    for i, (_, entry) in enumerate(samples):
        seen = [m for m in (mapping.to_metres(np.asarray(p, float)) for p in people(entry)) if np.isfinite(m).any()]
        if seen:
            track[i] = min(seen, key=lambda m: mean_off(m[None], want).min())
            found[i] = True
    track = smooth(track)

    err = np.linalg.norm(track[:, None] - want[None], axis=-1)                # sample, stance, limb
    cost = np.where(np.isfinite(err), np.minimum(err, CAP_M), CAP_M).mean(-1)
    # A foot swinging past the beta's hold on its way to another is not that
    # stance, so moving limbs count against a moment.
    t = np.array([s[0] for s in samples])
    step = np.gradient(track, t, axis=0) if len(t) > 1 else np.zeros_like(track)
    speed = np.nan_to_num(np.linalg.norm(step, axis=-1)).max(-1)
    cost += STILL * np.minimum(speed, 1.0)[:, None]

    # Cheapest increasing assignment of samples to stances.
    n, K = cost.shape
    best = np.full((K, n), np.inf)
    back = np.zeros((K, n), int)
    best[0] = cost[:, 0]
    idx = np.arange(n)
    for k in range(1, K):
        low = np.minimum.accumulate(best[k - 1])
        at = np.maximum.accumulate(np.where(best[k - 1] <= low, idx, 0))
        best[k, 1:] = cost[1:, k] + low[:-1]
        back[k, 1:] = at[:-1]
    chosen = [int(np.argmin(best[-1]))]
    for k in range(K - 1, 0, -1):
        chosen.append(int(back[k, chosen[-1]]))
    chosen.reverse()

    tolerance = {k: ON_HOLD_M + (ANKLE_TO_TOE_M if k.endswith("F") else 0) for k in LIMBS}
    stances = []
    for k, i in enumerate(chosen):
        e = err[i, k]
        stances.append({"move": k, "limb": tr["moves"][k - 1]["limb"] if k else None, "t": float(t[i]),
                        "error_m": {limb: float(v) if np.isfinite(v) else None for limb, v in zip(LIMBS, e)},
                        "offset_m": {limb: (track[i, j] - want[k, j]).tolist() for j, limb in enumerate(LIMBS)},
                        "judge": list(LIMBS), "tolerance": tolerance, "drift_m": None})
    return {"samples": len(samples), "found": int(found.sum()), "stances": stances, "track": track}


def lands(stance):
    """Whether every limb judged is on its hold; None where one was not seen."""
    errors = [stance["error_m"][k] for k in stance["judge"]]
    if None in errors:
        return None
    return all(e <= stance["tolerance"][k] for k, e in zip(stance["judge"], errors))


class Register:
    """Homographies from video frames to the photo, found by matching the wall."""

    LONG = 1600

    def __init__(self, photo):
        import cv2
        self.cv2 = cv2
        self.sift = cv2.SIFT_create(4000)
        self.kp, self.des = self.sift.detectAndCompute(photo, None)
        self.matcher = cv2.BFMatcher()

    def __call__(self, frame):
        cv2 = self.cv2
        s = min(1.0, self.LONG / max(frame.shape[:2]))
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or self.des is None or len(kp) < MIN_MATCHES:
            return None
        pairs = [p for p in self.matcher.knnMatch(des, self.des, k=2) if len(p) == 2]
        good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
        if len(good) < MIN_MATCHES:
            return None
        src = np.float32([kp[m.queryIdx].pt for m in good]) / s
        dst = np.float32([self.kp[m.trainIdx].pt for m in good])
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inliers.sum() < MIN_MATCHES:
            return None
        return H


def photo_of(tr):
    import cv2
    data = (tr.get("photo") or {}).get("jpeg")
    if not data:
        return None
    raw = np.frombuffer(base64.b64decode(data.split(",", 1)[1]), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def detect(video, model, tr, every=None, own_pace=False):
    """Everyone the pose model finds in each sampled frame, in photo pixels."""
    import cv2
    from ultralytics import YOLO

    pose = YOLO(model)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"{video}: cannot open")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if every is None:
        every = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / OWN_PACE_FPS)) if own_pace else 1
    photo = photo_of(tr)
    register = Register(photo) if photo is not None else None

    raw, frames, homs, n, end, tried, lined = [], [], [], 0, 0.0, 0, 0
    H, last = None, -np.inf
    while True:
        ok, image = cap.read()
        if not ok:
            break
        # A recorded canvas has no fixed frame rate, so take each frame's own time.
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        end = max(end, t)
        if n % every == 0:
            if register and t - last >= REGISTER_S:
                last, tried = t, tried + 1
                found = register(image)
                if found is not None:
                    H, lined = found, lined + 1
            result = pose(image, verbose=False)[0]
            found_people = []
            if result.keypoints is not None:
                for xy, conf in zip(result.keypoints.xy.cpu().numpy(), result.keypoints.conf.cpu().numpy()):
                    found_people.append(np.array([xy[KEYPOINT[k]] if conf[KEYPOINT[k]] >= SEEN else [np.nan, np.nan]
                                                  for k in LIMBS]))
            raw.append((t, found_people))
            homs.append(H)
            frames.append(n)
        n += 1
    cap.release()

    # Frames before the first good line-up take the first one; with none at
    # all, the video is assumed to be framed like the photo.
    first = next((h for h in homs if h is not None), None)
    if first is None:
        sx, sy = np.array(tr["size"], float) / size
        first = np.diag([sx, sy, 1.0])
        how = "not lined up with the photo" + (", which the transcript lacks" if photo is None else "") + \
              "; assumed framed the same"
    else:
        how = f"lined up with the photo on {lined} of {tried} tries"
    homs = [h if h is not None else first for h in homs]
    samples = [(t, [project(H, p) for p in ps]) for (t, ps), H in zip(raw, homs)]
    return samples, frames, homs, size, end, how


def draw(video, out, expected, samples, frames, homs, size, chosen):
    """The video again, with where the beta wants each limb in green and what was found in red."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    by_frame = {f: i for i, f in enumerate(frames)}
    n = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if n in by_frame:
            i = by_frame[n]
            back = np.linalg.inv(homs[i])
            want = expected(samples[i][0])
            got = chosen(i)
            for j in range(4):
                if want is not None:
                    w = tuple(int(v) for v in project(back, want[j])[0])
                    cv2.circle(image, w, 10, (90, 220, 60), 3)
                if got is not None and np.isfinite(got[j]).all():
                    g = tuple(int(v) for v in project(back, got[j])[0])
                    cv2.circle(image, g, 6, (60, 60, 255), -1)
                    if want is not None:
                        cv2.line(image, w, g, (60, 60, 255), 2)
        writer.write(image)
        n += 1
    cap.release()
    writer.release()


def cm(v):
    return "  -  " if v is None or not np.isfinite(v) else f"{v * 100:4.0f} cm"


def offset_words(dx, dy):
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return "not seen"
    return f"{abs(dx) * 100:.0f} cm {'right' if dx > 0 else 'left'}, {abs(dy) * 100:.0f} cm {'higher' if dy > 0 else 'lower'}"


def clock(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def stance_name(tr, s):
    return "start" if not s["move"] else f"move {s['move']}: {tr['moves'][s['move'] - 1]['text']}"


def show(tr, report, name, how):
    print(name)
    print(f"  {how}")
    found = report["found"] / max(report["samples"], 1)
    print(f"  climber found in {report['found']} of {report['samples']} frames ({found:.0%})")
    if report["stretch"] != 1.0:
        print(f"  video is {report['stretch']:.2f}x the transcript's length; the beta was stretched to fit")
    err = report["error_m"]
    print("  error on the wall, median and 90th percentile:")
    for j, limb in enumerate(LIMBS):
        seen = err[:, j][np.isfinite(err[:, j])]
        if len(seen):
            print(f"    {NAME[limb]:<10} {cm(np.median(seen))}  {cm(np.percentile(seen, 90))}  ({len(seen)} frames)")
        else:
            print(f"    {NAME[limb]:<10} never seen")
    if report["swapped"]:
        print(f"  left and right swapped in {report['swapped']} frames")
    print(f"  stance by stance (a limb lands within {ON_HOLD_M * 100:.0f} cm of its hold):")
    landed = 0
    for s in report["stances"]:
        verdict = {True: "lands", False: "misses", None: "not seen"}[lands(s)]
        landed += verdict == "lands"
        detail = "  ".join(f"{k} {cm(s['error_m'][k])}" for k in s["judge"])
        drift = f", others off by up to {cm(s['drift_m']).strip()}" if s["drift_m"] is not None else ""
        print(f"    {stance_name(tr, s)}\n      {detail}  {verdict}{drift}")
    print(f"  {landed} of {len(report['stances'])} stances land")


def show_matched(tr, report, name, how):
    print(name)
    print(f"  {how}")
    found = report["found"] / max(report["samples"], 1)
    print(f"  climber found in {report['found']} of {report['samples']} frames ({found:.0%})")
    if not report["found"]:
        print("  with no climber, no stance can be matched")
        return
    tol = report["stances"][0]["tolerance"]
    print(f"  stance by stance, in order, at the climber's own pace (on a hold: hands within "
          f"{tol['LH'] * 100:.0f} cm at the wrist, feet {tol['LF'] * 100:.0f} cm at the ankle):")
    reached = 0
    for s in report["stances"]:
        verdict = lands(s)
        reached += verdict is True
        print(f"    {clock(s['t'])}  {stance_name(tr, s)}")
        print("      " + "  ".join(f"{k} {cm(s['error_m'][k])}" for k in LIMBS) + "  " +
              {True: "reached", False: "not reached", None: "not seen"}[verdict])
        for k in LIMBS:
            e = s["error_m"][k]
            if e is not None and e > s["tolerance"][k]:
                print(f"      {NAME[k]} was {offset_words(*s['offset_m'][k])} than the beta's hold")
    print(f"  {reached} of {len(report['stances'])} stances reached")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("transcript")
    ap.add_argument("video")
    ap.add_argument("--own-pace", action="store_true",
                    help="a person climbing at their own speed: match stances in order, not by time")
    ap.add_argument("--model", default="yolo11s-pose.pt", help="an Ultralytics pose model")
    ap.add_argument("--every", type=int, help="check every nth frame (default: all, or about 10 a second with --own-pace)")
    ap.add_argument("--draw", help="write the video with the beta and the found limbs drawn on it")
    args = ap.parse_args()

    tr = load(args.transcript)
    samples, frames, homs, size, end, how = detect(args.video, args.model, tr, args.every, args.own_pace)
    if not samples:
        raise SystemExit(f"{args.video}: no frames")
    name = f"{Path(args.transcript).name} against {Path(args.video).name}"
    mapping = mapping_of(tr)
    if args.own_pace:
        report = match(tr, samples)
        show_matched(tr, report, name, how)
        keys = [f for f in tr["frames"] if f["key"]]
        times = [s["t"] for s in report["stances"]]
        def expected(t):
            k = np.searchsorted(times, t, side="right") - 1
            return None if k < 0 else np.array([keys[k]["joints"][limb] for limb in LIMBS])
        def chosen(i):
            return mapping.to_pixels(report["track"][i]) if np.isfinite(report["track"][i]).any() else None
    else:
        report = score(tr, samples, duration=end)
        show(tr, report, name, how)
        times = np.array([f["t"] for f in tr["frames"]])
        def expected(t):
            f = tr["frames"][max(0, np.searchsorted(times, t / report["stretch"], side="right") - 1)]
            return np.array([f["joints"][JOINT[limb]] for limb in LIMBS])
        def chosen(i):
            want = mapping.to_metres(expected(samples[i][0]))
            seen = [p for p in people(samples[i][1]) if np.isfinite(p).any()]
            return min(seen, key=lambda p: mean_off(mapping.to_metres(p), want)) if seen else None
    if args.draw:
        draw(args.video, args.draw, expected, samples, frames, homs, size, chosen)
        print(f"  drawn to {args.draw}")


if __name__ == "__main__":
    main()
