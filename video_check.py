"""Does a video keep to the beta?

    .venv/bin/python video_check.py climbing-beta-*.json video.mp4 [--draw out.mp4]

Save as video in the app writes two files: the video, and a transcript saying
where every hand and foot is at every moment of it. This finds the climber in
any video of that beta, the app's own or one a video model made from it, and
reports how far each hand and foot is from where the beta put it, in
centimetres on the wall, and whether each move lands.

The climber is found with an Ultralytics pose model. It gives wrists and
ankles, so hands are compared at the wrist, which sits a few centimetres off
the grip, and feet at the ankle, which the figure has as well. ON_HOLD_M allows
for that and for the size of a hold.
"""

import argparse
import json
from pathlib import Path

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
RETIME = 0.02       # a video this much longer or shorter than the transcript is stretched to fit


def load(path):
    tr = json.loads(Path(path).read_text())
    if tr.get("app") != "climbing-beta" or tr.get("kind") != "beta-transcript" or tr.get("version") != 1:
        raise ValueError(f"{path}: not a transcript saved with a beta video")
    if any(f["t"] is None for f in tr["frames"]):
        raise ValueError(f"{path}: some frames were never shown while recording")
    return tr


def score(tr, samples, video_size, duration=None):
    """Compare (time, points) samples from a video with the transcript.

    points is a (4, 2) array of LH, RH, LF, RF in video pixels, NaN where a
    limb was not seen, or None where no climber was found.
    """
    mapping = wall_mapping(TypeAdapter(Calibration).validate_python(tr["calibration"]), tuple(tr["size"]))
    to_video = np.array(tr["size"], float) / np.asarray(video_size, float)
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
    found = np.array([s[1] is not None for s in samples])
    for i, (_, pts) in enumerate(samples):
        if pts is None:
            continue
        got = mapping.to_metres(np.asarray(pts, float) * to_video)
        want = ref_m[which[i]]
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
                        "drift_m": drift, "frames": int(held.sum())})

    return {"samples": len(samples), "found": int(found.sum()), "stretch": stretch,
            "error_m": err, "swapped": int(swapped.sum()), "stances": stances}


def lands(stance):
    """Whether the limb that moved is on its hold, or for the start, every limb."""
    limbs = [stance["limb"]] if stance["limb"] else LIMBS
    errors = [stance["error_m"][k] for k in limbs]
    return None if None in errors else max(errors) <= ON_HOLD_M


def detect(video, model, every=1):
    """(time, points) for each sampled frame, with points in video pixels."""
    import cv2
    from ultralytics import YOLO

    pose = YOLO(model)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"{video}: cannot open")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    samples, frames, n, end = [], [], 0, 0.0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        # A recorded canvas has no fixed frame rate, so take each frame's own time.
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        end = max(end, t)
        if n % every == 0:
            result = pose(image, verbose=False)[0]
            pts = None
            if result.keypoints is not None and len(result.boxes):
                best = int(result.boxes.conf.argmax())
                xy = result.keypoints.xy[best].cpu().numpy()
                conf = result.keypoints.conf[best].cpu().numpy()
                pts = np.array([xy[KEYPOINT[k]] if conf[KEYPOINT[k]] >= SEEN else [np.nan, np.nan]
                                for k in LIMBS])
            samples.append((t, pts))
            frames.append(n)
        n += 1
    cap.release()
    return samples, frames, size, end


def draw(video, out, tr, samples, frames, size, report):
    """The video again, with the beta's limbs in green and the found ones in red."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    times = np.array([f["t"] for f in tr["frames"]])
    scale = np.asarray(size, float) / np.array(tr["size"], float)
    by_frame = dict(zip(frames, samples))
    n = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if n in by_frame:
            t, pts = by_frame[n]
            f = tr["frames"][max(0, np.searchsorted(times, t / report["stretch"], side="right") - 1)]
            for j, limb in enumerate(LIMBS):
                want = tuple(int(v) for v in np.array(f["joints"][JOINT[limb]]) * scale)
                cv2.circle(image, want, 9, (90, 220, 60), 3)
                if pts is not None and np.isfinite(pts[j]).all():
                    got = tuple(int(v) for v in pts[j])
                    cv2.line(image, want, got, (60, 60, 255), 2)
                    cv2.circle(image, got, 6, (60, 60, 255), -1)
        writer.write(image)
        n += 1
    cap.release()
    writer.release()


def cm(v):
    return "  -  " if v is None or not np.isfinite(v) else f"{v * 100:4.0f} cm"


def show(tr, report, name):
    print(name)
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
        what = "start" if not s["move"] else f"move {s['move']}: {tr['moves'][s['move'] - 1]['text']}"
        limbs = [s["limb"]] if s["limb"] else LIMBS
        detail = "  ".join(f"{k} {cm(s['error_m'][k])}" for k in limbs)
        drift = f", others off by up to {cm(s['drift_m']).strip()}" if s["drift_m"] is not None else ""
        print(f"    {what}\n      {detail}  {verdict}{drift}")
    print(f"  {landed} of {len(report['stances'])} stances land")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("transcript")
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11s-pose.pt", help="an Ultralytics pose model")
    ap.add_argument("--every", type=int, default=1, help="check every nth frame")
    ap.add_argument("--draw", help="write the video with the beta and the found limbs drawn on it")
    args = ap.parse_args()

    tr = load(args.transcript)
    samples, frames, size, end = detect(args.video, args.model, args.every)
    if not samples:
        raise SystemExit(f"{args.video}: no frames")
    report = score(tr, samples, size, duration=end)
    show(tr, report, f"{Path(args.transcript).name} against {Path(args.video).name}")
    if args.draw:
        draw(args.video, args.draw, tr, samples, frames, size, report)
        print(f"  drawn to {args.draw}")


if __name__ == "__main__":
    main()
