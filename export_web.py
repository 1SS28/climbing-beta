"""Export the trained weights for the browser.

    .venv/bin/python export_web.py

Vercel cannot run the detector: torch is 558 MB against a 500 MB Python bundle
and there is no GPU, so detection moved into the page. The browser needs a
fixed-shape ONNX model, which means the input size is chosen here and locked in.

Measured on the 50-image test split at conf 0.15:

    .pt   @1600   96.9%   what the local server does
    .pt   @1280   96.2%
    .pt   @960    95.2%
    onnx  @1280   96.6%   <- shipped
    onnx  @960    96.2%

1280 costs 0.3 points against the local server and runs in 0.67s on a 24 MP
photo over WebGPU. Exporting is not retraining, so the weights are unchanged.

Browsers keep the model for a year, so new weights must ship under a new
filename, or phones that already have the old one will go on using it.
"""

import shutil
from pathlib import Path

EDGE = 1280
WEIGHTS = "runs/v1_960_best.pt"
OUT = Path("public/models/holds_1280.onnx")


def main():
    from ultralytics import YOLO

    exported = YOLO(WEIGHTS).export(format="onnx", imgsz=EDGE, dynamic=False, opset=19)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, OUT)
    print(f"{OUT}  {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
