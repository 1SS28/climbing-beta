"""Holds in metric 3D, the representation everything downstream consumes.

A single photo leaves z at zero under a flat-wall assumption; depth sensing
would fill it properly. The search never sees a pixel, so it does not depend on
which capture path produced the scene.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Hold:
    """Metres, origin at the foot of the wall: +x right, +y up, +z off the wall."""

    x: float
    y: float
    z: float = 0.0
    size: float = 0.05  # across, in metres; proxy for how positive the hold is

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])


def polygon_area(poly):
    """Shoelace. Pixels squared."""
    x, y = np.asarray(poly)[:, 0], np.asarray(poly)[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def scale_from_reference(p1, p2, metres):
    """Metres per pixel from two image points a known distance apart.

    A single photo cannot recover scale, so the reference is an input. Two
    points rather than a wall height, since it accepts whatever is visible.
    Only valid for a reference lying in the wall plane.
    """
    span = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if span < 1.0:
        raise ValueError("reference points are the same pixel")
    return metres / span


def from_polygons(polys, image_height_px, metres_per_pixel):
    """Detected mask polygons -> holds in metres.

    Assumes the wall is flat and square to the sensor, so one scale applies
    across the frame. Perspective breaks this: holds high on the wall are
    further away and read smaller.
    """
    mpp = metres_per_pixel
    holds = []
    for poly in polys:
        p = np.asarray(poly, dtype=float)
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        holds.append(
            Hold(
                x=cx * mpp,
                y=(image_height_px - cy) * mpp,  # pixels run down, the wall runs up
                z=0.0,
                size=float(np.sqrt(polygon_area(p)) * mpp),
            )
        )
    return holds
