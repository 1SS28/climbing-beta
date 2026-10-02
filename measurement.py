"""Validated, detector-independent measurements on a single wall plane."""
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from calibrate import calibrate
from scene import scale_from_reference
from wall import to_wall_metres

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Coordinate = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Point = tuple[Coordinate, Coordinate]


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["reference"]
    points: list[Point] = Field(min_length=2, max_length=2)
    metres: Positive = Field(le=1000)


class Grid(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["grid"]
    corners: list[Point] = Field(min_length=4, max_length=4)
    cols: int = Field(strict=True, ge=1, le=1000)
    rows: int = Field(strict=True, ge=1, le=1000)
    spacing_m: Positive = Field(le=10)


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    calibration: Annotated[Reference | Grid, Field(discriminator="method")]
    points: list[Point] | None = Field(default=None, min_length=2, max_length=500)
    route: list[Annotated[int, Field(strict=True, ge=0)]] | None = Field(
        default=None, min_length=2, max_length=500)

    @model_validator(mode="after")
    def one_selection(self):
        if (self.points is None) == (self.route is None):
            raise ValueError("provide either measurement points or detected hold IDs")
        return self


def in_frame(points, size):
    p = np.asarray(points, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > np.asarray(size)).any():
        raise ValueError("all points must be inside the photo")
    return p


def measure_route(request, entry):
    size = entry["size"]
    if request.route is not None:
        if any(i >= len(entry["polys"]) for i in request.route):
            raise ValueError("selected hold does not exist; select it again")
        pixels = [np.asarray(entry["polys"][i]).mean(axis=0) for i in request.route]
    else:
        pixels = request.points
    pixels = in_frame(pixels, size)
    cal = request.calibration
    if isinstance(cal, Reference):
        ref = in_frame(cal.points, size)
        scale = scale_from_reference(ref[0], ref[1], cal.metres)
        xy = pixels * [scale, -scale]
        basis = f"Known reference: {cal.metres:g} m"
        note = "Approximate: assumes a flat wall photographed straight on; perspective is not corrected."
    else:
        corners = in_frame(cal.corners, size)
        # TL, TR, BL, BR must describe a convex, non-degenerate block.
        boundary = corners[[0, 1, 3, 2]]
        edges = np.roll(boundary, -1, axis=0) - boundary
        following = np.roll(edges, -1, axis=0)
        turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        if not (np.all(turns > 1) or np.all(turns < -1)):
            raise ValueError("pick four distinct corners around one rectangle: TL, TR, BL, BR")
        H = np.asarray(calibrate(corners, cal.cols, cal.rows, cal.spacing_m, size)["H"])
        # Reject points on/across the projective horizon instead of returning huge distances.
        inv = np.linalg.inv(H)
        denom = np.c_[np.vstack([corners, pixels]), np.ones(len(corners) + len(pixels))] @ inv[2]
        if np.min(np.abs(denom)) < 1e-9 or not (np.all(denom > 0) or np.all(denom < 0)):
            raise ValueError("calibration cannot map this route; pick a wider block around the route")
        xy = to_wall_metres(H, pixels)
        basis = f"Bolt grid: {cal.cols} × {cal.rows} gaps at {cal.spacing_m:g} m"
        note = "Perspective corrected. All reference points and route points must lie on the same flat wall panel."
    if not np.isfinite(xy).all():
        raise ValueError("calibration could not produce finite distances")
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return {
        "basis": basis, "note": note, "method": cal.method,
        "gaps": [{"from": i, "to": i + 1, "metres": round(float(d), 3)}
                 for i, d in enumerate(distances)],
        "total_m": round(float(distances.sum()), 3),
        "rise_m": round(float(np.ptp(xy[:, 1])), 3),
        "warning": "Route spans more than 6 m along the wall; check your reference." if np.ptp(xy[:, 1]) > 6 else None,
    }
