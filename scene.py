"""The platform-neutral scene: holds in metric 3D.

Everything downstream of detection consumes this and nothing else. A single
photo fills z with zero (flat-wall assumption); ARKit depth would fill it
properly. The beta search never learns which one produced it — that is the
point. It is the piece that has to survive the move to a native app, so it must
never see a pixel.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Hold:
    """One hold. Metres, origin at the foot of the wall, +x right, +y up,
    +z out from the wall plane."""

    x: float
    y: float
    z: float = 0.0
    size: float = 0.05  # across, in metres — a proxy for how positive it is

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])


def polygon_area(poly):
    """Shoelace. Pixels squared."""
    x, y = np.asarray(poly)[:, 0], np.asarray(poly)[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def from_polygons(polys, image_height_px, wall_height_m=4.5):
    """Detected mask polygons -> holds in metres.

    The scale here is the weakest link in the whole pipeline: it assumes the
    photo spans a wall of known height and that the wall is flat and parallel to
    the sensor. Perspective alone breaks that — holds at the top of a 4.5 m wall
    are further away and read smaller. This is exactly the assumption ARKit
    would replace, and the reason the search must not depend on how it was made.
    """
    mpp = wall_height_m / image_height_px  # metres per pixel
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
