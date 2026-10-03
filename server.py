"""Prototype server: upload a wall photo and measure a single route.

    .venv/bin/python server.py      then open http://127.0.0.1:8000

Route selection is manual, by design. Detection is reliable and the beta search
works, but choosing which holds form a route is not: colour grouping assumes
gyms set routes in one colour, and gyms that mark with tape break that outright.
A climber knows their route and can tap it in seconds, so the prototype asks
rather than guesses. Automatic grouping belongs on top of this later, as a
suggestion the climber corrects.

Both /api/measure and /api/beta take the same validated calibration and refuse
without one. Neither assumes a wall height.

Holds are detected once per photo and cached, because that is the slow step.
"""

import io
import logging
import uuid
from pathlib import Path
from threading import Lock
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from beta import Climber, find_start, route_chain, search
from measurement import Calibration, Measurement, holds_from_calibration, measure_route
from scene import MAX_WALL_M, plausible

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "data/uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
STATIC = ROOT / "static"

app = FastAPI()
_model = None
_detection_lock = Lock()
_cache: dict[str, dict] = {}


class BetaRequest(BaseModel):
    """A route to climb, on a wall whose scale is known."""

    model_config = ConfigDict(extra="forbid")
    id: str
    calibration: Calibration
    route: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=3, max_length=500)
    height_m: Annotated[float, Field(gt=0.5, le=2.5)] = 1.68


def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(ROOT / "runs/v1_960_best.pt"))
    return _model


def as_holds(pid: str):
    """Cached detections for a photo, or 404."""
    if pid not in _cache:
        raise HTTPException(404, "Unknown photo; upload it again.")
    return _cache[pid]


@app.post("/api/photo")
async def photo(file: UploadFile = File(...)):
    """Store a photo, detect its holds, return them as polygons."""
    raw = await file.read(20 * 1024 * 1024 + 1)
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Photo is too large; choose one under 20 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 40_000_000:
                raise HTTPException(413, "Photo is too large; use at most 40 megapixels.")
            img = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Cannot read this photo. Try a JPEG, PNG, or WebP image.")
    pid = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{pid}.jpg"
    img.save(path, quality=92)

    def detect():
        import torch
        device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
        with _detection_lock:
            return model().predict(str(path), imgsz=1600, conf=0.15, device=device, verbose=False)[0]

    try:
        r = await run_in_threadpool(detect)
    except Exception:
        logging.exception("Hold detection failed")
        path.unlink(missing_ok=True)
        raise HTTPException(503, "Hold detection is unavailable. Please try again.")
    polys = [np.asarray(p) for p in (r.masks.xy if r.masks is not None else []) if len(p) >= 3]
    _cache[pid] = {"polys": polys, "size": img.size}
    return {
        "id": pid,
        "width": img.width,
        "height": img.height,
        "holds": [{"id": i, "points": p.round(1).tolist(),
                   "centre": p.mean(axis=0).round(1).tolist()} for i, p in enumerate(polys)],
    }


@app.post("/api/beta")
async def beta(payload: BetaRequest):
    """Given chosen holds and a calibration, return the move sequence.

    Hands take the chosen route, feet take every hold the calibration placed on
    the wall, which is how most gyms set.
    """
    entry = as_holds(payload.id)
    if any(i >= len(entry["polys"]) for i in payload.route):
        raise HTTPException(400, "Selected hold does not exist; select it again.")
    try:
        holds, ids, off_wall, basis, note = holds_from_calibration(
            payload.calibration, entry["polys"], entry["size"])
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, str(exc))

    def refused(reason):
        return JSONResponse({"ok": False, "basis": basis, "note": note,
                             "off_wall": off_wall, "reason": reason}, status_code=200)

    # The browser selects by detector index; the search works over the holds
    # that survived calibration, so indices have to be translated both ways.
    index = {detected: i for i, detected in enumerate(ids)}
    missing = [i for i in payload.route if i not in index]
    if missing:
        return refused(f"holds {missing} are not on the calibrated wall plane. "
                       "Calibrate on the panel the route is set on, or deselect them.")
    route = [index[i] for i in payload.route]

    ok, span = plausible(None, entry["size"][1], holds=holds)
    if not ok:
        return refused(f"that calibration puts the holds across {span:.1f} m of wall, "
                       f"which is not a boulder (limit {MAX_WALL_M:.0f} m). "
                       "Check the hole spacing or the reference distance.")

    climber = Climber(height=payload.height_m)
    chain, why = route_chain(holds, route, climber)
    if chain is None:
        return refused(why)

    start = find_start(holds, climber, hands=chain)
    if start is None:
        return refused("no stance fits this climber on these holds")
    path, info = search(holds, start, chain[-1], c=climber, hands=chain)
    if path is None:
        return refused(str(info))

    limbs = ("LH", "RH", "LF", "RF")
    on_chain = set(chain)
    moves = []
    for a, b in zip(path, path[1:]):
        k = next(i for i in range(4) if a[i] != b[i])
        moves.append({"limb": limbs[k], "from": ids[a[k]], "to": ids[b[k]],
                      "distance_m": round(float(np.linalg.norm(holds[b[k]].pos - holds[a[k]].pos)), 2),
                      "hand": k < 2})
    return {"ok": True, "basis": basis, "note": note, "off_wall": off_wall,
            "cost": round(float(info), 2),
            "dropped": [i for i in payload.route if index[i] not in on_chain],
            "start": {limb: ids[start[k]] for k, limb in enumerate(limbs)},
            "moves": moves}


@app.post("/api/measure")
async def measure(payload: Measurement):
    """Measure selected points with an explicit reference; detection is optional."""
    entry = as_holds(payload.id)
    try:
        return measure_route(payload, entry)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/holds/{pid}")
def holds(pid: str):
    """Holds already detected for a photo, so a reload need not re-detect."""
    entry = as_holds(pid)
    W, H = entry["size"]
    return {"id": pid, "width": W, "height": H,
            "holds": [{"id": i, "points": p.round(1).tolist(),
                       "centre": p.mean(axis=0).round(1).tolist()}
                      for i, p in enumerate(entry["polys"])]}


@app.get("/api/photo/{pid}.jpg")
def photo_file(pid: str):
    p = UPLOADS / f"{pid}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
