"""Draw the beta: a body that holds the stances the search chose.

The search decides where the four contacts go and nothing else. An animation
needs shoulders, elbows, hips and knees as well, so this places them, and it
decides nothing in doing so: every frame is pinned to the search's contacts.
Where no body of the climber's height can span a stance, the limb is drawn
stretched to the hold and its reach reported, never quietly shortened onto a
hold it does not touch.

A jointed body was kept out of the search because its proportions cannot be
recovered from a photo, so each guess would decide the beta. Here the same
guesses decide only how the picture looks.

Segment lengths are fractions of height from the standard anthropometric table
(Drillis and Contini, as tabulated in Winter's Biomechanics). That table is a
body standing still, and it spans 1.20 times height from sole to palm; reaching
climbers go further by lifting the shoulder and pointing the toe. Both are
modelled as joints with limits, a shrug of up to 5% of height and a toe point
of up to 4%, rather than as bones that stretch. Without them an easy ladder
showed half its frames over-reaching, which was the figure being wrong, not
the climb. Everything is
worked out in wall metres with y up and projected onto the photo only at the
end, so the figure takes the same perspective as the holds it touches.

Elbows and knees bend partly out of the wall, as real ones do. Worked out flat,
every bend went sideways, and a fifth of the limbs folded sharper than a right
angle across the photo; a bent arm seen from behind mostly looks shorter, not
kinked. The contacts and the shoulders and hips stay on the wall plane, so
reach is unchanged and only the drawing of a bend is affected. Depth is drawn
as if seen square on, which ignores a few percent of parallax for a joint a
hand's width off the wall.
"""

from dataclasses import dataclass

import numpy as np

LIMBS = ("LH", "RH", "LF", "RF")
ROOT = {"LH": "shoulder_l", "RH": "shoulder_r", "LF": "hip_l", "RF": "hip_r"}
MIDDLE = {"LH": "elbow_l", "RH": "elbow_r", "LF": "knee_l", "RF": "knee_r"}
END = {"LF": "ankle_l", "RF": "ankle_r"}
# Fixed lengths. The shoulder girdle and the feet are left out: they move by
# design, within the shrug and toe-point limits.
RIGID = (("hip_l", "hip_r"), ("neck", "head"),
         ("shoulder_l", "elbow_l"), ("elbow_l", "LH"), ("shoulder_r", "elbow_r"), ("elbow_r", "RH"),
         ("hip_l", "knee_l"), ("knee_l", "ankle_l"), ("hip_r", "knee_r"), ("knee_r", "ankle_r"))
BONES = RIGID + (("shoulder_l", "shoulder_r"), ("hip_l", "shoulder_l"), ("hip_r", "shoulder_r"),
                 ("ankle_l", "LF"), ("ankle_r", "RF"))

COMFORT = 0.85     # how straight a limb sits when nothing forces it: a little bent
TILT = {"arm": np.radians(50), "leg": np.radians(35)}   # how far out of the wall a bend goes
BEYOND = 1.002     # reach past this is beyond the body, about 2 mm on a 1.68 m climber's arm
STEPS = 10         # frames per move


@dataclass(frozen=True)
class Body:
    """Segment lengths in metres, from height."""

    height: float = 1.68

    def segments(self, limb):
        """Shoulder to elbow and elbow to mid-palm, or hip to knee and knee to sole."""
        h = self.height
        return (0.186 * h, 0.200 * h) if limb in ("LH", "RH") else (0.245 * h, 0.285 * h)

    def slack(self, limb):
        """Extra reach from lifting the shoulder, or pointing the toe."""
        return (0.05 if limb in ("LH", "RH") else 0.04) * self.height

    @property
    def torso(self):
        return 0.288 * self.height      # shoulder line to hip line

    @property
    def shoulder(self):
        return 0.1295 * self.height     # half the shoulder width

    @property
    def hip(self):
        return 0.0955 * self.height     # half the hip width

    @property
    def neck(self):
        return 0.052 * self.height

    @property
    def head(self):
        return 0.065 * self.height      # radius


def trunk(params, body):
    """Hips, shoulders, neck and head from the hip centre and a lean angle."""
    hx, hy, lean = params
    up = np.array([np.sin(lean), np.cos(lean)])
    right = np.array([np.cos(lean), -np.sin(lean)])
    hips = np.array([hx, hy])
    chest = hips + up * body.torso
    return {"hip_l": hips - right * body.hip, "hip_r": hips + right * body.hip,
            "shoulder_l": chest - right * body.shoulder, "shoulder_r": chest + right * body.shoulder,
            "neck": chest + up * body.neck, "head": chest + up * (body.neck + body.head)}


def _strain(params, contacts, body, previous, stiffness=4.0):
    t = trunk(params, body)
    e = 0.5 * params[2] ** 2
    for limb in LIMBS:
        a, b = body.segments(limb)
        d = np.linalg.norm(contacts[limb] - t[ROOT[limb]])
        # Past full length costs through the comfort term, so the shrug and the
        # toe point are used when they are needed rather than by default.
        e += 1e6 * max(0.0, d / (a + b + body.slack(limb)) - 1) ** 2 + (d / (a + b) - COMFORT) ** 2
    if previous is not None:
        # Hold the body near where it is guided, so it shifts instead of jumping.
        e += stiffness * ((params[0] - previous[0]) ** 2 + (params[1] - previous[1]) ** 2) / body.height ** 2
        e += stiffness / 4 * (params[2] - previous[2]) ** 2
    return e


def _minimise(f, x0, step):
    """Nelder-Mead. Three parameters do not justify a scipy dependency on Vercel."""
    pts = [np.asarray(x0, dtype=float)]
    pts += [pts[0] + np.eye(len(x0))[i] * step[i] for i in range(len(x0))]
    vals = [f(p) for p in pts]
    for _ in range(600):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if vals[-1] - vals[0] < 1e-12 and max(np.abs(p - pts[0]).max() for p in pts) < 1e-7:
            break
        centre = np.mean(pts[:-1], axis=0)
        reflect = centre + (centre - pts[-1])
        fr = f(reflect)
        if fr < vals[0]:
            expand = centre + 2 * (centre - pts[-1])
            fe = f(expand)
            pts[-1], vals[-1] = (expand, fe) if fe < fr else (reflect, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = reflect, fr
        else:
            contract = centre + 0.5 * (pts[-1] - centre)
            fc = f(contract)
            if fc < vals[-1]:
                pts[-1], vals[-1] = contract, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    return pts[int(np.argmin(vals))]


def bend(root, end, a, b, choose, tilt):
    """The middle joint of a two-link limb from root to end, and its depth.

    The joint leaves the wall plane by `tilt` from the sideways bend, so what
    is returned is its position on the wall and its height off it.
    """
    v = end - root
    d = float(np.linalg.norm(v))
    if d < 1e-9:
        return root + np.array([0.0, -a]), 0.0
    u = v / d
    if d >= a + b:
        return root + u * d * a / (a + b), 0.0
    d = max(d, abs(a - b) + 1e-9)
    cos = (a * a + d * d - b * b) / (2 * a * d)
    sin = np.sqrt(max(0.0, 1 - cos * cos))
    n = np.array([-u[1], u[0]]) * np.cos(tilt)
    joint = choose(root + a * (u * cos + n * sin), root + a * (u * cos - n * sin))
    return joint, a * sin * np.sin(tilt)


def limb(root, contact, a, b, slack, arm, choose):
    """Root, middle and end joints of a limb holding `contact`, and its reach.

    Within full length the limb bends. Past it, an arm lifts its shoulder
    toward the hold and a leg points its toe, each up to `slack`. Past that the
    limb is drawn stretched so the contact stays on its hold, and reach, the
    distance over everything the limb can give, comes out above 1.
    """
    v = contact - root
    d = float(np.linalg.norm(v))
    u = v / d if d > 1e-9 else np.array([0.0, 1.0])
    full = a + b
    extra = min(max(d - full, 0.0), slack)
    if arm:
        root = root + u * extra                       # shrug
        end = contact
    else:
        end = contact - u * extra                     # toe point: the ankle stops short
    return root, bend(root, end, a, b, choose, TILT["arm" if arm else "leg"]), end, d / (full + slack)


def pose(contacts, body, previous=None, stiffness=4.0):
    """A body holding these four contacts. Returns (params, joints, reach, depth).

    depth is how far each elbow and knee sits off the wall; everything else is on it.
    """
    f = lambda p: _strain(p, contacts, body, previous, stiffness)
    step = np.array([0.1, 0.1, 0.3]) * [body.height, body.height, 1]
    if previous is None:
        hands = (contacts["LH"] + contacts["RH"]) / 2
        feet = (contacts["LF"] + contacts["RF"]) / 2
        # The hips sit about where the legs end, measured from hands to feet.
        hips = feet + (hands - feet) * 0.44
        best = min((_minimise(f, [hips[0], hips[1], lean], step) for lean in (0.0, 0.35, -0.35)), key=f)
    else:
        best = _minimise(f, previous, step * 0.3)

    joints = trunk(best, body)
    middle = (joints["hip_l"][0] + joints["hip_r"][0]) / 2
    reach, depth = {}, {}
    for name in LIMBS:
        if name in ("LH", "RH"):
            choose = lambda p, q: p if p[1] <= q[1] else q                      # elbows hang
        elif name == "LF":
            choose = lambda p, q: p if p[0] - middle <= q[0] - middle else q    # knees out
        else:
            choose = lambda p, q: p if p[0] - middle >= q[0] - middle else q
        root, mid, end, reach[name] = limb(joints[ROOT[name]], np.asarray(contacts[name], dtype=float),
                                           *body.segments(name), body.slack(name),
                                           name in ("LH", "RH"), choose)
        joints[ROOT[name]], (joints[MIDDLE[name]], depth[MIDDLE[name]]) = root, mid
        if name in END:
            joints[END[name]] = end
        joints[name] = np.array(contacts[name], dtype=float)
    return best, joints, reach, depth


def ease(t):
    return t * t * (3 - 2 * t)


def frames(path, holds, body, steps=STEPS):
    """Every frame of a beta, in wall metres.

    One limb moves per move, along the straight line between its two holds,
    eased in and out; the other three stay exactly on their holds. The last
    frame of each move is that move's stance exactly.

    The body is placed at every stance first, then carried from one to the next
    while the limb travels. Solving each frame only from the last let the hips
    sit still through a reach until full stretch forced them to snap 30 cm in a
    single frame; a climber shifts their hips during the move, not after it.
    Between stances the body is held to that glide: held loosely, it swapped
    mid-move to another comfortable position, 48 cm away in one frame on a real
    wall. Holding it harder than this buys almost no smoothness and costs
    posture, since a body on a straight line between two stances bends its
    limbs more than one free to settle. A long move also gets more frames, so
    nothing travels more than about 4 cm.
    """
    def contacts(stance):
        return {limb: np.asarray(holds[i].pos[:2], dtype=float) for limb, i in zip(LIMBS, stance)}

    stances = []
    params = None
    for stance in path:
        params = pose(contacts(stance), body, params)[0]
        stances.append(params)

    params, joints, reach, depth = pose(contacts(path[0]), body, stances[0])
    out = [{"move": 0, "moving": None, "key": True, "joints": joints, "reach": reach, "depth": depth}]
    for k, (a, b) in enumerate(zip(path, path[1:]), start=1):
        limb = next(i for i in range(4) if a[i] != b[i])
        start, end = np.asarray(holds[a[limb]].pos[:2]), np.asarray(holds[b[limb]].pos[:2])
        travel = max(np.linalg.norm(end - start), np.linalg.norm(stances[k][:2] - stances[k - 1][:2]))
        n = max(steps, int(np.ceil(1.5 * travel / 0.04)))      # 1.5: the eased curve's steepest point
        for s in range(1, n + 1):
            t = ease(s / n)
            c = contacts(a)
            c[LIMBS[limb]] = end if s == n else start + (end - start) * t
            guide = stances[k - 1] + (stances[k] - stances[k - 1]) * t
            # Firmly on the glide between stances; only reach should pull it off.
            params, joints, reach, depth = pose(c, body, guide, stiffness=40.0)
            out.append({"move": k, "moving": LIMBS[limb], "key": s == n,
                        "joints": joints, "reach": reach, "depth": depth})
    return out


def on_photo(wall_frames, to_pixels, body):
    """Project frames onto the photo, ready to send to the browser."""
    names = list(wall_frames[0]["joints"])
    out = []
    for f in wall_frames:
        pts = np.array([f["joints"][n] for n in names])
        px = to_pixels(pts)
        head = f["joints"]["head"]
        # The head's radius in pixels depends on where it is, under perspective.
        rim = to_pixels(np.array([head, head + [body.head, 0.0]]))
        out.append({"move": f["move"], "moving": f["moving"], "key": f["key"],
                    "joints": {n: [round(float(x), 2), round(float(y), 2)] for n, (x, y) in zip(names, px)},
                    "head_px": round(float(np.linalg.norm(rim[1] - rim[0])), 2),
                    "reach": {k: round(float(v), 3) for k, v in f["reach"].items()},
                    "beyond": [k for k, v in f["reach"].items() if v > BEYOND]})
    return out
