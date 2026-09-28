#!/bin/bash
# Unattended V1: wait for the download, convert, smoke-test, then train for real.
# Each stage gates the next, so a failure stops the chain instead of wasting
# hours on the long run.
#
# Everything after the install runs through .venv/bin/python rather than
# `uv run`. `uv run` re-syncs the environment and takes .venv/.lock, which
# deadlocks against any other uv process. That is what stalled the first
# attempt: the multi-hour download held the lock and every other stage queued
# behind it forever.
set -o pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for download to finish and release the venv lock"
while pgrep -f snapshot_download > /dev/null; do sleep 20; done
imgs=$(find data/climbinst -name '*.jpg' | wc -l | tr -d ' ')
[ "$imgs" -ge 1200 ] || { log "FAILED: only $imgs images downloaded"; exit 1; }
log "download done: $imgs images, $(du -sh data/climbinst | cut -f1)"

log "installing ultralytics"
uv pip install -q ultralytics || { log "FAILED: install"; exit 1; }
$PY -c "import ultralytics" || { log "FAILED: ultralytics not importable"; exit 1; }

log "converting to YOLO format"
$PY prepare.py || { log "FAILED: prepare"; exit 1; }

# Cheap proof that the conversion, the MPS loop and inference all work before
# committing to a multi-hour run.
log "smoke run (nano, 640px, 15 epochs)"
$PY train.py --model yolo11n-seg.pt --epochs 15 --imgsz 640 --batch 16 --name smoke \
  || { log "FAILED: smoke"; exit 1; }
log "smoke done"

log "full run (small, 960px, 60 epochs)"
$PY train.py --model yolo11s-seg.pt --epochs 60 --imgsz 960 --batch 8 --name v1 \
  || { log "FAILED: full"; exit 1; }

log "ALL DONE"
