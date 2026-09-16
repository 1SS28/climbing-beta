# Climbing Beta

## Goal

Given a photograph of an indoor bouldering wall, identify the holds, group them
into a single route, and propose a plausible sequence of moves — the beta — that
completes it.

Two halves, deliberately different in character:

- **Perception** is learned. Finding holds in a photo is exactly the problem
  neural networks are good at and hand-written rules are bad at.
- **Beta** is searched, not learned. A stance is a discrete assignment of four
  limbs to holds, so finding a sequence is graph search under physical
  constraints. No training data required, and the output is interpretable:
  every move can be explained by the cost that chose it.

## Constraints

- Indoor gym walls only. Holds are manufactured objects against a flat wall and
  route membership is encoded by colour, both of which outdoor rock lacks.
- Public datasets for perception; no hand-labelling unless the data runs out.
- Classical search for beta; no learned policy, no reinforcement learning.
- Input: a single RGB photograph.
- Output:
  - detected holds with masks
  - the subset forming one route
  - an ordered move sequence
  - a cost breakdown per move

## Pipeline

    photo -> [detector] -> holds -> [colour grouping] -> route -> [search] -> beta

### 1. Hold detection

Instance segmentation. Masks rather than boxes, because a hold's shape and
orientation determine where it can actually be grasped, and because boxes on a
dense wall overlap badly.

### 2. Route grouping

Unsupervised. Sample each hold's mask pixels, take a robust colour in a
perceptually uniform space (CIELAB), and cluster. Route membership is a
colour-cluster assignment, so this needs no labels at all.

### 3. Beta search

    Stance s = (LH, RH, LF, RF)

Each limb is a hold index or None. A move changes one limb. Search the graph of
reachable stances from a start stance to any stance with a hand on the finish
hold, minimising accumulated cost.

## Representation

Hold:

    h = (mask, centroid, area, colour, orientation)

Move cost, initially:

- reach strain — limb extension relative to a nominal wingspan
- balance — horizontal distance from centre of mass to the support base
- travel — how far the moving limb goes
- quality — hold size and orientation as a proxy for how positive it is

## Milestones

### V0
Load the dataset and render annotated holds over their images. Verifies the
data before any model exists.

### V1
Fine-tune an instance segmentation model. Report mask mAP on held-out images.

### V2
Group holds into routes by colour.

### V3
Stance graph and shortest-path beta search on ground-truth holds.

### V4
Run the whole pipeline end to end on an unseen photo.

### V5
Improve the cost model: balance, body rotation, hold orientation. Compare
proposed beta against how a person actually climbs it.

## Data

[ClimbInst](https://huggingface.co/datasets/ClimbVision/ClimbInst) —
1,509 images, 89,020 instance masks, one class (`climbing holds`, volumes
included). Labelme polygon JSON, one file per image. CC BY-NC-SA 4.0, so this
project stays non-commercial.

Its masks came from SAM3 and were then corrected by hand. Hold *type* (jug,
crimp, sloper) is not labelled, so V5's hold-quality term has to be inferred
from mask geometry rather than read off a label. Neither is route membership,
which is what V2 has to recover from colour.

About 4.2 GB in total: annotations are tiny (~14 KB each, `imageData` is null
throughout) and the images are the whole of it.

## Questions to Investigate

- Does colour clustering survive gym lighting, or is a wall-illumination
  correction needed first?
- How should reach be calibrated without knowing the climber's height or the
  wall's scale?
- Is a stance graph small enough to search exhaustively, or does it need
  pruning?
- Should feet be restricted to route holds, or is smearing on the wall allowed?
- How do you evaluate proposed beta when there is no single correct answer?
- Does mask geometry predict hold type well enough to be worth using?
