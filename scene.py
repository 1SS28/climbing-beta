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


def scale_from_reference(p1, p2, metres):
    """Metres per pixel from two image points a known distance apart.

    A single photo cannot recover scale on its own — the T-nut lattice was the
    automatic route and it does not survive real images (see README) — so the
    reference comes from the user. Two points rather than a wall height because
    it costs the same and accepts whatever is actually visible: the wall's
    full height, a standard panel edge, a person of known height.

    Only valid for things lying in the wall plane; a reference held out from the
    wall measures short.
    """
    span = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if span < 1.0:
        raise ValueError("reference points are the same pixel")
    return metres / span


def from_polygons(polys, image_height_px, metres_per_pixel):
    """Detected mask polygons -> holds in metres.

    Still assumes the wall is flat and square to the sensor, so one scale holds
    everywhere in frame. Perspective breaks that — holds high on the wall are
    further away and read smaller — which is what ARKit depth would fix, and
    why the search consumes this and never the photo.
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
