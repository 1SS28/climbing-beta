#!/bin/bash
# Full V1 run, sized to finish overnight rather than to be maximal.
#
# Measured: nano @ 640 is ~2.5 min/epoch on this M5 Pro. YOLO11s @ 960 is ~8x
# that (3.5x params, 2.25x pixels), which is ~16 hours for 60 epochs, far too
# long. 768px for 30 epochs lands near 5 hours, and patience=15 can end it
# sooner. Resolution is kept above 640 deliberately: gym holds are small and
# dense, so pixels matter more here than model capacity.
set -o pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the smoke run to finish"
while pgrep -f "train.py --model yolo11n-seg" > /dev/null; do sleep 30; done

if [ ! -f runs/segment/runs/smoke/weights/best.pt ]; then
  log "FAILED: smoke produced no weights"
  exit 1
fi
log "smoke weights present"

log "full run (small, 768px, 30 epochs)"
$PY train.py --model yolo11s-seg.pt --epochs 30 --imgsz 768 --batch 8 --name v1 \
  || { log "FAILED: full"; exit 1; }

log "ALL DONE"
