#!/usr/bin/env python3
"""Load one sample folder and describe what a pipeline would receive.

    python3 samples/load_sample.py samples/01-urban-dense

Requires numpy and Pillow.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Mask pixel value -> class name. Also served live at GET /classes.
CLASSES = {
    0: "background", 23: "water", 46: "ice", 69: "wood", 92: "grass",
    115: "farmland", 138: "landuse", 161: "building", 184: "railway",
    207: "road_minor", 230: "road_major", 253: "aeroway",
}

# Only present in masks fetched with ?mode=ambiguous, where the renderer could
# not attribute a boundary pixel to one class. Excluded from the one-hot stack
# and intended as ignore_index during training.
AMBIGUOUS = 255


def main(folder: Path) -> None:
    meta = json.loads((folder / "meta.json").read_text())
    labels = json.loads((folder / "labels.json").read_text())

    # Both PNGs are single channel, so these load as (H, W) uint8 -- no colour
    # conversion and no alpha channel to discard.
    image = np.array(Image.open(folder / "image.png"))
    mask = np.array(Image.open(folder / "mask.png"))

    print(f"{folder.name}  z{meta['tile']['z']}/{meta['tile']['x']}/{meta['tile']['y']}")
    print(f"  {meta['about']}\n")

    print(f"image  shape={image.shape} dtype={image.dtype} "
          f"range=[{image.min()}, {image.max()}]")
    print(f"mask   shape={mask.shape} dtype={mask.dtype} "
          f"unique={len(np.unique(mask))} values\n")

    # Every mask value must be a known class; the server snaps antialiased
    # edges before encoding, so this should never report a stray value.
    unknown = set(np.unique(mask)) - set(CLASSES) - {AMBIGUOUS}
    print("class distribution:")
    for value, count in sorted(
        zip(*np.unique(mask, return_counts=True)), key=lambda p: -p[1]
    ):
        pct = 100 * count / mask.size
        name = "ambiguous" if value == AMBIGUOUS else CLASSES.get(value, f"UNKNOWN({value})")
        print(f"  {name:<12} {value:>3}  {pct:5.2f}%")
    print(f"  unknown values: {unknown or 'none'}\n")

    # A one-hot stack is the usual next step for segmentation training.
    present = [v for v in CLASSES if (mask == v).any()]  # AMBIGUOUS is not a class
    onehot = np.stack([(mask == v) for v in present]).astype(np.uint8)
    print(f"one-hot stack: {onehot.shape}  ({len(present)} classes present)\n")

    print(f"labels: {labels['count']} total")
    by_layer: dict[str, int] = {}
    for item in labels["labels"]:
        by_layer[item["layer"]] = by_layer.get(item["layer"], 0) + 1
    for layer, n in sorted(by_layer.items(), key=lambda p: -p[1]):
        print(f"  {layer:<22} {n}")

    # The ambiguous variant is the same map with contested edges withheld.
    amb_path = folder / "mask_ambiguous.png"
    if amb_path.exists():
        amb = np.array(Image.open(amb_path))
        flagged = amb == AMBIGUOUS
        print(f"\nmask_ambiguous.png: {100 * flagged.mean():.2f}% flagged {AMBIGUOUS} "
              f"(ignore_index)")
        # Anything not flagged must match mask.png exactly.
        assert (amb[~flagged] == mask[~flagged]).all(), "ambiguous mask drifted from mask.png"
        print("  unflagged pixels identical to mask.png: yes")
        for value in sorted(present, key=lambda v: -(mask == v).sum()):
            of_class = flagged[mask == value]
            if of_class.size:
                print(f"    {CLASSES[value]:<12} {100 * of_class.mean():5.1f}% flagged")

    print("\nfirst 5 labels (x, y are pixels from top-left):")
    for item in labels["labels"][:5]:
        # Sample the mask under each label anchor -- the label frame and the
        # raster frame are identical, so this needs no transform.
        x = min(int(item["x"]), mask.shape[1] - 1)
        y = min(int(item["y"]), mask.shape[0] - 1)
        under = CLASSES.get(int(mask[y, x]), "?")
        print(f"  {item['text'][:28]:<30} ({item['x']:6.1f}, {item['y']:6.1f})  "
              f"{item['layer']:<20} over {under}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
