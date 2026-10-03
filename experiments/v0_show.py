"""V0: render annotated holds over their image, to verify the data loads correctly.

    uv run python v0_show.py <annotation.json> [out.png]
"""

import sys

from PIL import ImageDraw

from experiments.data import load

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "out/v0.png"

img, polys = load(src)
draw = ImageDraw.Draw(img, "RGBA")  # RGBA mode so fills can be translucent
for p in polys:
    draw.polygon([tuple(v) for v in p], fill=(0, 255, 140, 70), outline=(0, 255, 140, 255), width=3)

img.save(dst)
print(f"{len(polys)} holds  {img.width}x{img.height}  -> {dst}")
