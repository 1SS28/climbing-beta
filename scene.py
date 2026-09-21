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


# Bouldering walls. 6 m is already tall for a boulder, so a scale implying more
# than this is wrong, and saying so beats propagating it through the geometry.
MAX_WALL_M = 6.0

# Common commercial T-nut spacings, metres. 6 and 8 inches are both used in
# North America, 125 and 150 mm in Europe, and MoonBoard is 200 mm. Ask the gym
# rather than assume: 6 inches against 8 is a 33% scale error, and scale
# multiplies through every distance the beta search compares.
COMMON_TNUT_SPACING = {"6in": 0.1524, "8in": 0.2032, "200mm": 0.200,
                       "150mm": 0.150, "125mm": 0.125}
DEFAULT_TNUT_SPACING = 0.1524  # 6 in


def scale_from_tnuts(p1, p2, holes_apart=1, spacing_m=0.2032):
    """Metres per pixel from two T-nut centres a known number of holes apart.

    Better than a person or a wall height for one reason: the bolt holes lie in
    the wall plane, so the scale is exact where the holds are. A person standing
    in front of the wall is nearer the camera and reads slightly large.

    The grid is also the same everywhere on a wall and present on every wall, so
    a gym only has to be asked once, and holes_apart lets the two points be far
    apart, which divides the clicking error by that count.
    """
    if holes_apart < 1:
        raise ValueError("holes_apart must be at least 1")
    px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if px < 1.0:
        raise ValueError("the two points are the same pixel")
    return (spacing_m * holes_apart) / px


def plausible(metres_per_pixel, image_height_px, max_wall_m=MAX_WALL_M):
    """Sanity-check a scale against what a bouldering wall can be.

    Every scale so far has been a guess that silently decided the geometry, and
    a wrong one swung route connectivity from 9/10 to 2/10. Cheap to check.
    """
    implied = metres_per_pixel * image_height_px
    return 1.5 <= implied <= max_wall_m * 2.5, implied


def scale_from_person(person_px_height, person_m=1.70):
    """Metres per pixel, from a person of known height standing in the frame.

    The focal length cancels: at the subject's own depth the scale is just
    height over pixel height. That holds only at their depth, so it transfers
    to the wall when they stand against it, and drifts when they do not.

    Chosen over monocular metric depth, which was tested and rejected: Depth
    Anything V2 Metric Indoor put a climber 3.50 m away at 4.96-5.38 m (Small
    +54/+73%, Base +42/+48%), a bias too large to build on. Gyms are far outside
    its training distribution.

    Most gym photos contain a climber, so this is automatic in the common case,
    needs no calibration, and works at a gym nobody has visited before. The
    error is whatever the height guess is wrong by, a few percent, rather than
    the factor of two an assumed wall height can be out by.
    """
    if person_px_height <= 0:
        raise ValueError("person height in pixels must be positive")
    return person_m / float(person_px_height)


def focal_px_from_exif(focal_35mm, long_edge_px):
    """Focal length in pixels, from the 35mm-equivalent focal length in EXIF.

    Needed only to move scale between depths: metres per pixel at distance D is
    D / focal_px. iPhones record this (an iPhone 17 Pro main camera reports 24mm,
    giving 3808 px on a 5712 px edge).
    """
    return (float(focal_35mm) / 36.0) * float(long_edge_px)


def from_wall_plane(polys, H):
    """Detected mask polygons -> holds, using a wall calibration.

    Maps each mask through the homography onto the wall itself, so perspective
    is removed rather than assumed away: a hold high on the wall stops reading
    as smaller than one at eye level, and distances are measured along the wall
    instead of across the image. This is what from_polygons approximates when
    no calibration exists.
    """
    from wall import to_wall_metres

    holds = []
    for poly in polys:
        p = np.asarray(poly, dtype=float)
        on_wall = to_wall_metres(H, p)
        cx, cy = on_wall[:, 0].mean(), on_wall[:, 1].mean()
        holds.append(Hold(x=float(cx), y=float(cy), z=0.0,
                          size=float(np.sqrt(polygon_area(on_wall)))))
    return holds


def from_polygons(polys, image_height_px, metres_per_pixel):
    """Detected mask polygons -> holds in metres.

    Assumes the wall is flat and square to the sensor, so one scale applies
    across the frame. Perspective breaks this: holds high on the wall are
    further away and read smaller.
    """
    holds = []
    for poly in polys:
        p = np.asarray(poly, dtype=float)
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        holds.append(
            Hold(
                x=cx * metres_per_pixel,
                y=(image_height_px - cy) * metres_per_pixel,  # pixels run down, the wall runs up
                z=0.0,
                size=float(np.sqrt(polygon_area(p)) * metres_per_pixel),
            )
        )
    return holds
