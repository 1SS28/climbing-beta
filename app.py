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

The beta runs here for the same reason: the search and the figure that draws it
are tested in Python, and the browser only plays the frames it is sent.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from beta import Climber, find_start, route_chain, search
from figure import BONES, Body, frames, on_photo
from measurement import Calibration, Positive, Span, holds_from_calibration, measure_points
from scene import MAX_WALL_M, plausible

# A detector box can run a little past the frame; the hold filter clips it.
Vertex = tuple[Annotated[float, Field(allow_inf_nan=False)], Annotated[float, Field(allow_inf_nan=False)]]

PUBLIC = Path(__file__).resolve().parent / "public"

app = FastAPI(title="climbing-beta", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def caching(request, call_next):
    """Revalidate the page and scripts every time; keep the model for a year.

    Without a header the browser guesses, and a guess kept a stale app.js
    running against a new index.html. The model is 41 MB and changes only on
    retraining, which ships under a new filename, so it can be cached for good.
    On Vercel the CDN already revalidates static files, and vercel.json adds
    the long cache for the model.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/models/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif not path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.post("/api/measure")
def measure(payload: Span):
    """Distances along the wall between the points the climber picked."""
    try:
        return measure_points(payload)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, str(exc))


class BetaRequest(BaseModel):
    """A route, the holds around it, and the scale to read them at."""

    model_config = ConfigDict(extra="forbid")
    size: tuple[Positive, Positive]
    calibration: Calibration
    holds: list[list[Vertex]] = Field(min_length=3, max_length=600)
    route: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=3, max_length=100)
    height_m: Annotated[float, Field(gt=0.5, le=2.5)] = 1.68


@app.post("/api/beta")
def beta(payload: BetaRequest):
    """The move sequence for a route, and every frame of a body climbing it.

    Hands take the route, feet take any hold the calibration put on the wall.
    Each refusal says why, since "no beta" on its own teaches nothing.
    """
    if any(i >= len(payload.holds) for i in payload.route):
        raise HTTPException(400, "A route hold is not among the holds sent.")
    try:
        placed = holds_from_calibration(payload.calibration, payload.holds, payload.size)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, str(exc))

    def refuse(reason):
        return {"ok": False, "reason": reason, "basis": placed.basis, "off_wall": placed.dropped}

    index = {detected: i for i, detected in enumerate(placed.ids)}
    missing = [i for i in payload.route if i not in index]
    if missing:
        return refuse(f"route holds {missing} are not on the calibrated wall plane. "
                      "Calibrate on the panel the route is set on, or drop them.")
    ok, span = plausible(None, payload.size[1], holds=placed.holds)
    if not ok:
        return refuse(f"that scale puts the holds across {span:.1f} m of wall, which is not a "
                      f"boulder (limit {MAX_WALL_M:.0f} m). Check the spacing and the gap counts.")

    climber = Climber(height=payload.height_m)
    holds = placed.holds
    chain, why = route_chain(holds, [index[i] for i in payload.route], climber)
    if chain is None:
        return refuse(why)
    start = find_start(holds, climber, hands=chain)
    if start is None:
        return refuse("no starting stance fits a climber of this height on these holds")
    path, info = search(holds, start, chain[-1], c=climber, hands=chain)
    if path is None:
        return refuse(str(info))

    ids = placed.ids
    limbs = ("LH", "RH", "LF", "RF")
    moves = []
    for a, b in zip(path, path[1:]):
        k = next(i for i in range(4) if a[i] != b[i])
        moves.append({"limb": limbs[k], "from": ids[a[k]], "to": ids[b[k]],
                      "metres": round(float(np.linalg.norm(holds[b[k]].pos - holds[a[k]].pos)), 3)})
    body = Body(payload.height_m)
    return {"ok": True, "basis": placed.basis, "off_wall": placed.dropped,
            "skipped": [i for i in payload.route if index[i] not in set(chain)],
            "start": {limb: ids[path[0][k]] for k, limb in enumerate(limbs)},
            "moves": moves, "bones": BONES,
            "frames": on_photo(frames(path, holds, body), placed.to_pixels, body)}


@app.get("/api/health")
def health():
    return {"ok": True}


# Locally this serves the page, so `uvicorn app:app` is the whole app. On Vercel
# public/ goes to the CDN and is not in the function's filesystem at all, and
# mounting it regardless failed every API call at import.
if PUBLIC.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="public")
