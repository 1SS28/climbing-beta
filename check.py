"""Score tape-measure checks saved from the hosted app.

    .venv/bin/python check.py hold-distance-*.json [--photos]

The app saves one file per check: the photo it measured, the taps, the
calibration, what it measured and what the tape read. Each measurement is rerun
here through the same code the API uses, so a file that does not reproduce is
caught before its numbers are trusted.

Two kinds of error look alike in a single gap and come apart over several. A
wrong bolt spacing or a miscounted grid scales every distance by one factor, so
tape over measured is the same everywhere; tapping error scatters around 1. When
the ratio holds steady, the scale is the cause and the implied spacing says what
it should have been. --photos writes each embedded photo out beside its file.
"""

import argparse
import base64
import json
import statistics
from pathlib import Path

from measurement import Span, measure_points
from scene import COMMON_TNUT_SPACING

REPRODUCE_M = 0.0015    # the API rounds to the millimetre
MIN_FOR_SCALE = 3       # fewer gaps than this cannot separate scale from scatter


def load(path):
    rec = json.loads(Path(path).read_text())
    if rec.get("app") != "hold-distance" or rec.get("version") != 1:
        raise ValueError(f"{path}: not a check saved by the app")
    return rec


def rerun(rec):
    span = Span(size=(rec["photo"]["width"], rec["photo"]["height"]),
                calibration=rec["calibration"],
                points=[p["xy"] for p in rec["points"]])
    return [g["metres"] for g in measure_points(span)["gaps"]]


def score(records):
    """records: [(name, rec)] -> per-gap rows, files that failed, and a verdict."""
    rows, failed = [], []
    for name, rec in records:
        again = rerun(rec)
        saved = [g["metres"] for g in rec["result"]["gaps"]]
        if len(again) != len(saved) or any(abs(a - b) > REPRODUCE_M for a, b in zip(again, saved)):
            failed.append(name)
            continue
        for t, m in zip(rec["tape"], again):
            if t.get("tape_m"):
                rows.append({"file": name, "from": t["from"], "to": t["to"], "measured": m,
                             "tape": t["tape_m"], "pct": (m - t["tape_m"]) / t["tape_m"] * 100,
                             "spacing": rec["calibration"].get("spacing_m")})

    verdict = {"n": len(rows)}
    if not rows:
        return rows, failed, verdict
    pct = [r["pct"] for r in rows]
    verdict.update(mean_abs=statistics.fmean(abs(p) for p in pct), worst=max(abs(p) for p in pct),
                   mean_signed=statistics.fmean(pct))
    if len(rows) < MIN_FOR_SCALE:
        return rows, failed, verdict

    factor = statistics.median(r["tape"] / r["measured"] for r in rows)
    after = statistics.fmean(abs(r["measured"] * factor - r["tape"]) / r["tape"] * 100 for r in rows)
    verdict.update(factor=factor, mean_abs_after=after,
                   scale_error=abs(factor - 1) > 0.03 and after < verdict["mean_abs"] / 2)
    spacings = {r["spacing"] for r in rows}
    if verdict["scale_error"] and len(spacings) == 1 and None not in spacings:
        implied = spacings.pop() * factor
        name, value = min(COMMON_TNUT_SPACING.items(), key=lambda kv: abs(kv[1] - implied))
        verdict.update(implied_spacing=implied, nearest=name if abs(value - implied) / value < 0.03 else None)
    return rows, failed, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--photos", action="store_true", help="write each embedded photo beside its file")
    a = ap.parse_args()

    records = [(Path(f).name, load(f)) for f in a.files]
    if a.photos:
        for f, (_, rec) in zip(a.files, records):
            out = Path(f).with_suffix(".jpg")
            out.write_bytes(base64.b64decode(rec["photo"]["jpeg"].split(",", 1)[1]))
            print(f"photo -> {out}")

    rows, failed, v = score(records)
    for name in failed:
        print(f"{name}: does not reproduce through the current API; its numbers are not used")
    for r in rows:
        print(f"{r['file']}  {r['from'] + 1}->{r['to'] + 1}  measured {r['measured']:.3f}  "
              f"tape {r['tape']:.3f}  {(r['measured'] - r['tape']) * 100:+.1f} cm  {r['pct']:+.1f}%")
    if not v["n"]:
        print("no gaps with a tape reading")
        return
    print(f"\n{v['n']} gaps: mean error {v['mean_abs']:.1f}%, worst {v['worst']:.1f}%, "
          f"mean signed {v['mean_signed']:+.1f}%")
    if "factor" not in v:
        print(f"too few gaps to tell a scale error from tapping error (need {MIN_FOR_SCALE})")
    elif v["scale_error"]:
        print(f"consistent scale error: every distance is off by x{v['factor']:.3f}; "
              f"correcting it leaves {v['mean_abs_after']:.1f}%")
        if "implied_spacing" in v:
            hint = (f", which is the common {v['nearest']} spacing" if v["nearest"]
                    else ", not a common spacing, so check the gap counts for an off-by-one")
            print(f"the bolt spacing that fits is {v['implied_spacing'] * 1000:.1f} mm{hint}")
    else:
        print(f"no consistent scale error (x{v['factor']:.3f}); what is left is tapping and detection")


if __name__ == "__main__":
    main()
