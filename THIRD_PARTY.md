# Third-party terms

These terms attach to components this project depends on. They matter on
distribution, which includes publishing the source publicly, and are what
constrain the licence this project can carry.

## Ultralytics (YOLO11) - AGPL-3.0

`train.py`, `predict.py`, `v2_routes.py`, `v3_beta.py` and `find_routes.py`
import `ultralytics`. AGPL-3.0 is a strong copyleft licence: distributing a
work that builds on it, or offering it as a network service, can oblige you to
release the whole of that work under AGPL-3.0 as well.

Publishing the source, or shipping it as an app or hosted service, means one
of:

- buying an Ultralytics commercial licence, or
- retraining on a permissively licensed architecture, or
- releasing this project under AGPL-3.0.

## ClimbInst - CC BY-NC-SA 4.0

The trained weights in `runs/` come from the ClimbInst dataset, which is
non-commercial and share-alike. The weights are arguably a derivative of it.
Whether trained weights count as derivative works is legally unsettled, but the
non-commercial intent is clear enough that the weights should not be relied on
for a paid or ad-supported product.

Commercial release therefore means retraining on data that permits it, which
is the same moment the Ultralytics question has to be answered. Both
constraints fall due together.

## Practical position

Nothing here restricts building, training, or using the project privately.
Everything here restricts shipping it. Neither dependency needs resolving until
that point, and resolving either one means retraining anyway.
