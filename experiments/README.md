# Experiments

Work that produced a finding but is not in the shipping path. Kept because the
findings are the expensive part: without them the same ideas get tried again.

Run from the project root, so the modules in the root resolve:

    .venv/bin/python -m experiments.v3_beta photo.jpg -1 "150,1750,4.5" 1.68

## Does not work

| file | finding |
| --- | --- |
| `lattice.py` | Bolt-hole detection by Hough circles works. Fitting a grid to the holes does not: `confirm_grid` checks whether a fitted grid's predicted hole positions actually contain holes, and every candidate across three walls confirms 14 to 22%, against 12% for random points. Residuals of 0.15 to 0.19 look healthy for all of them, so residual cannot see this failure. A grid at twice the true pitch passes through every detected hole and predicts twice as many positions, half on bare wall. |
| `measure_wall.py` | Automatic wall measurement built on `lattice.py`, gated to refuse below 45% confirmation. It therefore refuses on every photo tried, which is the correct answer rather than a working feature. Four tapped bolt holes are both more accurate and already in the prototype. |
| `vertical.py` | Camera tilt from the vanishing point of vertical lines. Correct on synthetic input, 0.1 to 0.2% inliers on real photos. Gym walls do not carry enough long vertical structure. |

## Works, not wired in

| file | purpose |
| --- | --- |
| `colour.py` | sRGB to CIELAB, and complete-linkage clustering of hold colours. Single linkage chains straight through the orange-yellow-pink continuum; complete linkage does not. |
| `routes.py` | Score how route-like a colour group is, on rise, chain connectivity and chroma. The largest group is usually the neutral greys every route shares. |
| `v2_routes.py`, `find_routes.py`, `v3_beta.py`, `gym_test.py` | CLI drivers over colour grouping and the beta search. |
| `tape.py` | Find route tape. Colour uniformity is the discriminator, not saturation. Needed because this gym marks routes with tape, which breaks the premise that route membership is in the hold colour. Assigning tape to holds is unvalidated. |
| `tape_beta.py` | Photo to beta through tape rather than hold colour. |
| `make_tape_labels.py`, `train_tape.py`, `eval_tape.py` | A learned tape detector. Never trained or evaluated. |
| `evaluate.py` | Detection recall sweep over resolution and confidence. Source of the 96.9% recall figure. |
| `predict.py` | Run a trained model and save mask overlays. |
| `v0_show.py`, `data.py` | The first dataset viewer. |
