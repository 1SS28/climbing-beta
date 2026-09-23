"""Prototype server: upload a wall photo, pick a route, get beta.

    .venv/bin/python server.py      then open http://127.0.0.1:8000

Route selection is manual, by design. Detection is reliable and the beta search
works, but choosing which holds form a route is not: colour grouping assumes
gyms set routes in one colour, and gyms that mark with tape break that outright.
A climber knows their route and can tap it in seconds, so the prototype asks
rather than guesses. Automatic grouping belongs on top of this later, as a
suggestion the climber corrects.

Holds are detected once per photo and cached, because that is the slow step.
"""

import io
import json
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from beta import Climber, describe, find_start, route_chain, search
from calibrate import calibrate
from scene import MAX_WALL_M, from_polygons, from_wall_plane, plausible
from wall import to_wall_metres

UPLOADS = Path("data/uploads")
UPLOADS.mkdir(parents=True, exist_ok=True)
STATIC = Path("static")

app = FastAPI()
_model = None
_cache: dict[str, dict] = {}


def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("runs/v1_960_best.pt")
    return _model


@app.post("/api/photo")
async def photo(file: UploadFile = File(...)):
    """Store a photo, detect its holds, return them as polygons."""
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    pid = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{pid}.jpg"
    img.save(path, quality=92)

    r = model().predict(str(path), imgsz=1600, conf=0.15, device="mps", verbose=False)[0]
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
async def beta(payload: dict):
    """Given chosen holds and a scale, return the move sequence."""
    pid = payload.get("id")
    if pid not in _cache:
        raise HTTPException(404, "unknown photo; upload it again")
    entry = _cache[pid]
    polys, (W, H) = entry["polys"], entry["size"]

    route = [int(i) for i in payload.get("route", [])]
    if len(route) < 3:
        raise HTTPException(400, "pick at least three holds")

    climber = Climber(height=float(payload.get("height_m", 1.68)))
    corners = payload.get("corners")
    if corners and len(corners) == 4:
        cal = calibrate([tuple(c) for c in corners], int(payload["cols"]), int(payload["rows"]),
                        float(payload.get("spacing_m", 0.1524)), (W, H))
        holds = from_wall_plane(polys, np.array(cal["H"]))
        scale_note = f"{cal['metres_per_pixel']*1000:.2f} mm/px from the bolt grid"
    else:
        wall_m = float(payload.get("wall_height_m", 4.5))
        holds = from_polygons(polys, H, wall_m / H)
        scale_note = f"assuming {wall_m:.1f} m across the frame"

    ok, span = plausible(None, H, holds=holds)
    if not ok:
        return JSONResponse({"ok": False, "scale": scale_note,
                             "reason": f"that scale puts the holds across {span:.1f} m of wall, "
                                       f"which is not a boulder (limit {MAX_WALL_M:.0f} m). "
                                       f"Check the hole spacing or the wall height."},
                            status_code=200)

    chain, why = route_chain(holds, route, climber)
    if chain is None:
        return JSONResponse({"ok": False, "reason": why, "scale": scale_note}, status_code=200)

    start = find_start(holds, climber, hands=chain)
    if start is None:
        return JSONResponse({"ok": False, "reason": "no stance fits this climber on these holds",
                             "scale": scale_note}, status_code=200)
    path, info = search(holds, start, chain[-1], c=climber, hands=chain)
    if path is None:
        return JSONResponse({"ok": False, "reason": str(info), "scale": scale_note}, status_code=200)

    limbs = ("LH", "RH", "LF", "RF")
    moves = []
    for a, b in zip(path, path[1:]):
        k = next(i for i in range(4) if a[i] != b[i])
        moves.append({"limb": limbs[k], "from": int(a[k]), "to": int(b[k]),
                      "distance_m": round(float(np.linalg.norm(holds[b[k]].pos - holds[a[k]].pos)), 2),
                      "hand": k < 2})
    return {"ok": True, "scale": scale_note, "cost": round(float(info), 2),
            "dropped": [h for h in route if h not in chain],
            "start": {"LH": start[0], "RH": start[1], "LF": start[2], "RF": start[3]},
            "moves": moves}


@app.get("/api/holds/{pid}")
def holds(pid: str):
    """Holds already detected for a photo, so a reload need not re-detect."""
    if pid not in _cache:
        raise HTTPException(404, "unknown photo")
    entry = _cache[pid]
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
