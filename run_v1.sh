#!/bin/bash
# Unattended V1: wait for the download, convert, smoke-test, then train for real.
# Each stage gates the next, so a failure stops the chain instead of wasting
# hours on the long run.
set -o pipefail
cd "$(dirname "$0")"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for download"
until [ "$(find data/climbinst -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')" -ge 1219 ]; do sleep 30; done
log "download done: $(du -sh data/climbinst | cut -f1)"

log "waiting for ultralytics"
until uv run python -c "import ultralytics" 2>/dev/null; do sleep 15; done

log "converting to YOLO format"
uv run python prepare.py || { log "FAILED: prepare"; exit 1; }

# Cheap proof that the conversion, the MPS loop and inference all work before
# committing to a multi-hour run.
log "smoke run (nano, 640px, 15 epochs)"
uv run python train.py --model yolo11n-seg.pt --epochs 15 --imgsz 640 --batch 16 --name smoke \
  || { log "FAILED: smoke"; exit 1; }
log "smoke done"

log "full run (small, 960px, 60 epochs)"
uv run python train.py --model yolo11s-seg.pt --epochs 60 --imgsz 960 --batch 8 --name v1 \
  || { log "FAILED: full"; exit 1; }

log "ALL DONE"
