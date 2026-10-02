# Climbing Beta

Find the holds in a photo of an indoor bouldering wall, work out which ones
form a route, and propose a sequence of moves that climbs it.

The two halves of the problem are handled differently. Perception is learned,
because finding holds in a photo is what neural networks are good at. The move
sequence is searched rather than learned: a stance is a discrete assignment of
four limbs to holds, so finding a sequence is graph search under physical
constraints. That needs no training data, and every move can be explained by
the cost that chose it.

## Constraints

- Indoor gym walls only. Holds are manufactured objects against a flat wall,
  and route membership is encoded in hold colour. Outdoor rock has neither.
- Public datasets for perception. No hand labelling unless the data runs out.
- Classical search for the move sequence. No learned policy, no RL.
- Input is a single RGB photograph.
- Output is the detected holds, the subset forming one route, an ordered move
  sequence, and a cost breakdown per move.

## Pipeline

    photo -> detector -> holds -> colour grouping -> route -> search -> beta

### Hold detection

Instance segmentation rather than boxes. A hold's shape and orientation decide
where it can be grasped, and boxes on a dense wall overlap badly.

### Route grouping

Unsupervised, because route membership is not labelled in any dataset. Each
hold's mask pixels give a median colour in CIELAB, and those are clustered.
Groups are then scored on how route-like they are (see `routes.py`).

### Beta search

    Stance s = (LH, RH, LF, RF)

Each limb sits on a hold. A move changes one limb. Search the graph of
reachable stances from a start stance to any stance with a hand on the finish
hold, minimising accumulated cost.

Hands are restricted to route holds. Feet may use any hold on the wall, which
is how most gyms set. Restricting all four limbs to a sparse route makes most
routes unclimbable on paper.

## Move cost

- travel: how far the moving limb goes
- strain: limb extension as a fraction of reach limits, squared
- balance: horizontal distance from centre of mass to the feet
- quality: hold size, as a proxy for how positive it is

## Files

| file | purpose |
| --- | --- |
| `prepare.py` | ClimbInst Labelme JSON to YOLO segmentation format |
| `train.py` | fine-tune YOLO11-seg |
| `predict.py` | run a trained model, save mask overlays |
| `colour.py` | sRGB to CIELAB, and clustering holds by colour |
| `routes.py` | score how route-like a colour group is |
| `scene.py` | holds in metric 3D, the platform-neutral representation |
| `beta.py` | stance graph and the search over it |
| `v3_beta.py` | photo to beta, end to end |
| `find_routes.py` | scan the dataset for walls with colour-coded routes |
| `lattice.py` | T-nut grid scale estimation (does not work, see below) |

## Status

### V0, V1: detection
Done. YOLO11s-seg at 960px, 60 epochs, 3.8 hours on an M5 Pro. On 106 held-out
images containing 6,356 holds: mask mAP50 0.877, mAP50-95 0.548, box mAP50
0.919.

Training is I/O-bound rather than compute-bound, because Ultralytics forces 0
dataloader workers on MPS. YOLO11s at 768px ran 2.6 min/epoch against nano at
640's 2.5, despite roughly five times the compute. Resolution is nearly free on
this machine, so spend it.

Known failure: objects that are not on the wall get detected as holds, such as
brushes on the mat, a brush hanging from the ceiling, and wall signs. A
wall-plane or ground-line filter should clear most of them.

### V2: route grouping
Done. Single linkage failed outright: hold colours form a continuum from orange
through yellow to pink, so joining clusters by their nearest members chains
straight through the gaps. On a 76-hold wall with obviously distinct routes it
merged 43 of them even at a threshold of 5. Complete linkage requires every
pair in a cluster to be close, which gives the gap no foothold.

Picking the largest group is also wrong, since that is usually the neutral
greys shared by every route. `routes.py` scores groups on rise, chain
connectivity and chroma instead.

### V3, V4: beta search
Done. First working sequence is 14 moves on a colour-coded gym wall, selecting
an 8-hold green route out of 121 detected holds.

Two modelling errors had to be fixed first, both found by testing rather than
by reading the code:

Given any freedom, the search stacked all four limbs on one hold and shuffled
up the wall, because coincident contact points drive strain and balance to
zero. Fixed with a minimum torso span and a cap of two limbs per hold.

Maximum hand-to-foot span was set to 0.9x height, which deadlocked every real
route: the hands could not rise until the feet had, and the feet could not rise
until the hands had. A climber standing on a foothold and reaching overhead
spans about 1.25x their height.

A jointed body model was considered and dropped. Shoulder, hip and torso
parameters cannot be recovered from a photo, so each invented number would
quietly decide the beta. Only the distances between the four contact points are
observable, so only those are constrained.

### V5: next
Tune the cost weights against routes that a person has actually climbed. Filter
detections that are not on the wall. Infer hold orientation from mask geometry.

## Data

[ClimbInst](https://huggingface.co/datasets/ClimbVision/ClimbInst): 1,509
images claimed, 1,219 present, 89,020 instance masks, one class. Labelme
polygon JSON, one file per image, about 4.2 GB. CC BY-NC-SA 4.0.

The validation split advertised on the dataset card is empty, so `prepare.py`
carves one out of train by hashing filenames.

Masks came from SAM3 and were corrected by hand. Hold type (jug, crimp,
sloper) is not labelled, so the hold-quality term has to be inferred from mask
geometry. Route membership is not labelled either, which is what V2 recovers
from colour.

## Open questions

- Does colour clustering survive gym lighting? Partly. Complete linkage handles
  the hue continuum, but chalk and shadow are untested.
- How should reach be calibrated without knowing the wall's scale? By asking.
  The T-nut grid was the automatic candidate and it does not work. Gym walls
  are drilled on a regular lattice, so the bolt holes should be a ruler lying
  in the image. Tested two ways in `lattice.py` across three walls: dark-blob
  detection with a pair-distance histogram returns about 13,000 candidates per
  image, which is wall texture rather than holes, and an FFT power spectrum of
  a bare-wall patch shows no dominant peak (top five radii at 3.4 to 3.8x
  median, with inconsistent periods, where a lattice would give one sharp
  spike). At these resolutions a T-nut is 3 to 5 px and barely darker than the
  wall, many are hidden behind holds, and perspective smears whatever
  periodicity survives. Scale is now an input: two image points a known
  distance apart.
- Is the stance graph small enough to search exhaustively? So far yes, with
  feasibility pruning.
- Should feet be restricted to route holds? No, answered above.
- How do you evaluate proposed beta when there is no single correct answer?
  Still open, and the hardest question here. Nothing in any dataset labels the
  moves a climber made, with the partial exception of hold-usage work such as
  "The Way Up" (CVPR 2025 workshop).
- Does mask geometry predict hold type well enough to be worth using? Untested.

## Single-route measurement prototype

Run `.venv/bin/python server.py` and open http://127.0.0.1:8000.
The browser now focuses on measuring one route:

1. Upload a JPEG, PNG, or WebP photo (up to 20 MB / 40 megapixels).
2. Set an explicit scale: two points a known distance apart for an approximate,
   straight-on measurement, or four bolt-grid corners for perspective correction.
   Grid dimensions count **gaps**, not holes. Enter the actual spacing at the gym.
3. Select detected holds in route order, or use **Add points** for missed holds.
   **Move**, **Remove**, and **Earlier** let you correct the measurement sequence.
4. Measure to see each gap and the sum over the selected sequence. Zoom in for
   accurate taps and scroll to pan. Changing points or scale clears old results.

All points must lie on the same flat wall panel. Distances are between the
selected points projected onto that panel; protruding holds and routes crossing
wall corners are not reconstructed in 3D. The reported rise is the span along
that panel's up-axis, not a gravity-referenced vertical height. Real accuracy
still needs comparison with tape-measured gaps. Two-point calibration does not
remove perspective distortion.

The browser never assumes a wall height. `POST /api/measure` requires an explicit
`calibration` and either `points` (image pixel coordinates) or `route` (detected
hold IDs). For example:

```json
{
  "id": "<photo id returned by /api/photo>",
  "points": [[100, 800], [400, 400]],
  "calibration": {
    "method": "reference",
    "points": [[0, 0], [1000, 0]],
    "metres": 1
  }
}
```

For a grid, use `{"method":"grid", "corners":[[0,0],[1000,0],[0,1000],[1000,1000]],
"cols":10, "rows":10, "spacing_m":0.1}`. Corners are TL, TR, BL, BR.
Gap `from`/`to` values index the submitted sequence, not detector IDs. This
replaces the old measurement payload with optional corners/assumed wall height.
The experimental `/api/beta` and CLI search remain available with their existing
interfaces; beta search is not part of this measurement UI.

Photos are stored locally; detections remain in memory and must be uploaded
again after a server restart. Detection selects MPS, CUDA, or CPU automatically.
The trained weights must exist at `runs/v1_960_best.pt`.

Checks: `.venv/bin/python tests.py` and
`.venv/bin/python -m unittest test_measurement` (API tests require `httpx`).

## Running it

```bash
uv sync
.venv/bin/python prepare.py
.venv/bin/python train.py --model yolo11s-seg.pt --epochs 60 --imgsz 960 --batch 6 --name v1_960
.venv/bin/python v3_beta.py photo.jpg -1 "150,1750,4.5" 1.68
```

The last three arguments are the route group (-1 picks the best-scoring one), a
scale reference given as `y_top,y_bottom,metres`, and climber height in metres.

Use `.venv/bin/python` rather than `uv run`. `uv run` re-syncs the environment
and takes a lock on it, which deadlocks against any other uv process, including
a long download.

## Licence

AGPL-3.0, see LICENSE. This project imports Ultralytics, which is AGPL-3.0, so
anything built on it and distributed has to be AGPL too. Details and the
dataset terms are in THIRD_PARTY.md.
