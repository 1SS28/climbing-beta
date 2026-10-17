"""The deployed API: geometry only, no detector.

Vercel loads the top-level `app` from this file. The detector cannot come with
it: torch is 558 MB against a 500 MB Python bundle, and there is no GPU, so
hold detection runs in the browser from an ONNX export of the same weights.
On the labelled test split that export reaches 96.6% recall at 1280px against
96.9% for the .pt model at 1600px, which is the whole reason this split works.

What is left here is small and worth keeping in Python: the homography, the
checks that refuse a calibration, and the distances. Those have a test suite
and are about to be checked against a tape measure, so they stay in one
language rather than being reimplemented in JS where the two could drift.

Nothing is stored. The browser holds the photo and the detections and sends the
points it wants measured, so there is no upload, no cache and no state to lose
between invocations.
"""

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from measurement import Span, measure_points

PUBLIC = Path(__file__).resolve().parent / "public"

app = FastAPI(title="climbing-beta", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.post("/api/measure")
def measure(payload: Span):
    """Distances along the wall between the points the climber picked."""
    try:
        return measure_points(payload)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/health")
def health():
    return {"ok": True}


# Vercel serves public/ from its CDN, so this mount normally never runs. It
# exists so `uvicorn app:app` is the whole app locally, and so a deploy that
# routes everything here still works.
app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="public")
