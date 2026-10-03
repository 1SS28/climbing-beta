"""Validated, detector-independent measurements on a single wall plane.

One calibration feeds both the measurement and the beta search. Keeping them on
separate paths let the search run from an assumed wall height that a
measurement would have rejected, which is how a scale putting holds across
11.4 m of wall once reached the geometry unchallenged.
"""
from typing import Annotated, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from calibrate import calibrate
from scene import MAX_WALL_M, Hold, polygon_area, scale_from_reference
from wall import to_wall_metres

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Coordinate = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Point = tuple[Coordinate, Coordinate]

# A hold's span in metres, as sqrt of its wall-plane area. A crimp is about
# 0.04 and the largest volumes about 0.9; sqrt of area keeps long rails well
# inside the upper bound, since a 1.0 x 0.08 m rail reads 0.28. Outside this
# range the mask is not a climbing hold, or the calibration did not place it on
# the wall.
MIN_HOLD_M = 0.01
MAX_HOLD_M = 1.2


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


Calibration = Annotated[Reference | Grid, Field(discriminator="method")]


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    calibration: Calibration
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


def wall_mapping(calibration, size):
    """Validate a calibration and return (to_metres, basis, note).

    to_metres maps image pixels onto the wall plane in metres with y running
    up, and raises ValueError for points the calibration cannot place. Callers
    that must not lose a point let that propagate; the hold filter catches it
    and drops the hold instead.
    """
    if isinstance(calibration, Reference):
        ref = in_frame(calibration.points, size)
        scale = scale_from_reference(ref[0], ref[1], calibration.metres)

        def to_metres(points):
            return in_frame(points, size) * [scale, -scale]

        return (to_metres,
                f"Known reference: {calibration.metres:g} m",
                "Approximate: assumes a flat wall photographed straight on; "
                "perspective is not corrected.")

    corners = in_frame(calibration.corners, size)
    # TL, TR, BL, BR must describe a convex, non-degenerate block.
    boundary = corners[[0, 1, 3, 2]]
    edges = np.roll(boundary, -1, axis=0) - boundary
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(turns > 1) or np.all(turns < -1)):
        raise ValueError("pick four distinct corners around one rectangle: TL, TR, BL, BR")

    H = np.asarray(calibrate(corners, calibration.cols, calibration.rows,
                             calibration.spacing_m, size)["H"])
    inv = np.linalg.inv(H)
    # Everything must sit on the same side of the projective horizon as the
    # block, and clear of it, or the mapping returns huge or mirrored values.
    behind = np.c_[corners, np.ones(len(corners))] @ inv[2]
    if np.min(np.abs(behind)) < 1e-9 or not (np.all(behind > 0) or np.all(behind < 0)):
        raise ValueError("pick four distinct corners around one rectangle: TL, TR, BL, BR")
    side = np.sign(behind[0])

    def to_metres(points):
        p = in_frame(points, size)
        denom = np.c_[p, np.ones(len(p))] @ inv[2]
        if np.min(np.abs(denom)) < 1e-9 or not np.all(np.sign(denom) == side):
            raise ValueError("calibration cannot map this route; "
                             "pick a wider block around the route")
        xy = to_wall_metres(H, p)
        if not np.isfinite(xy).all():
            raise ValueError("calibration could not produce finite distances")
        return xy

    return (to_metres,
            f"Bolt grid: {calibration.cols} × {calibration.rows} gaps "
            f"at {calibration.spacing_m:g} m",
            "Perspective corrected. All reference points and route points must lie "
            "on the same flat wall panel.")


class Placed(NamedTuple):
    """Holds a calibration could put on the wall, and what it could not."""

    holds: list
    ids: list
    dropped: list
    basis: str
    note: str


def holds_from_calibration(calibration, polys, size):
    """Detected polygons -> metric holds, with what the calibration rejected.

    ids maps each kept hold back to its detector index, since the browser
    selects routes by that index.

    Objects that are not on the wall are a known detector failure: brushes on
    the mat, brushes hanging from the ceiling, wall signs. A calibration gives
    the wall plane, so three things can now be said about a mask instead of
    handing all of them to the search.

    Off-plane means across the projective horizon, where the mapping is
    meaningless rather than merely wrong. Implausible means a wall-plane span
    no climbing hold has. Outside means further beyond the calibrated block
    than a boulder is tall, which is where coordinates blow up as the
    vanishing line is approached from the near side and the sign test still
    passes. Both horizon bands matter on a steeply angled wall, where the
    vanishing line falls inside the frame.

    None of these identify a brush lying on the mat in front of a vertical
    wall, which projects to a sensible size in a sensible place. That needs the
    floor line or a learned on-wall test, and is still open.
    """
    to_metres, basis, note = wall_mapping(calibration, size)
    frame = np.asarray(size, dtype=float)
    if isinstance(calibration, Grid):
        w = calibration.cols * calibration.spacing_m
        h = calibration.rows * calibration.spacing_m
        limits = (-MAX_WALL_M, w + MAX_WALL_M, -MAX_WALL_M, h + MAX_WALL_M)
    else:
        # A reference scale is affine, so every pixel in the frame maps to a
        # bounded place and nothing can run away.
        limits = None

    holds, ids, dropped = [], [], []
    for i, poly in enumerate(polys):
        # A mask vertex a fraction of a pixel outside the frame is still a hold.
        p = np.clip(np.asarray(poly, dtype=float), 0.0, frame)
        if len(p) < 3:
            dropped.append({"id": i, "reason": "mask has no area"})
            continue
        try:
            on_wall = to_metres(p)
        except ValueError:
            dropped.append({"id": i, "reason": "off the wall plane"})
            continue
        x, y = float(on_wall[:, 0].mean()), float(on_wall[:, 1].mean())
        if limits and not (limits[0] <= x <= limits[1] and limits[2] <= y <= limits[3]):
            dropped.append({"id": i, "reason": f"{x:.0f}, {y:.0f} m is off the calibrated wall"})
            continue
        span = float(np.sqrt(polygon_area(on_wall)))
        if not MIN_HOLD_M <= span <= MAX_HOLD_M:
            dropped.append({"id": i, "reason": f"spans {span:.2f} m on the wall"})
            continue
        holds.append(Hold(x=x, y=y, z=0.0, size=span))
        ids.append(i)

    return Placed(holds, ids, dropped, basis, note)


def measure_route(request, entry):
    size = entry["size"]
    if request.route is not None:
        if any(i >= len(entry["polys"]) for i in request.route):
            raise ValueError("selected hold does not exist; select it again")
        pixels = [np.asarray(entry["polys"][i]).mean(axis=0) for i in request.route]
    else:
        pixels = request.points

    to_metres, basis, note = wall_mapping(request.calibration, size)
    xy = to_metres(pixels)
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    rise = float(np.ptp(xy[:, 1]))
    return {
        "basis": basis, "note": note, "method": request.calibration.method,
        "gaps": [{"from": i, "to": i + 1, "metres": round(float(d), 3)}
                 for i, d in enumerate(distances)],
        "total_m": round(float(distances.sum()), 3),
        "rise_m": round(rise, 3),
        "warning": (f"Route spans more than {MAX_WALL_M:.0f} m along the wall; "
                    "check your reference.") if rise > MAX_WALL_M else None,
    }
