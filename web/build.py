"""Builds web/index.html by inlining the trained weights and a few samples.

The page has to be a single self-contained file: opened from disk it has no
server, and fetching a separate weights.json over file:// is blocked by the
browser. So the weights go straight into the HTML.

Usage:
    ../bin/train --epochs 20 --hidden 64 --export web/weights.json
    python web/build.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TEMPLATE = HERE / "template.html"
WEIGHTS = HERE / "weights.json"
OUTPUT = HERE / "index.html"

TEST_IMAGES = ROOT / "data" / "t10k-images-idx3-ubyte"
TEST_LABELS = ROOT / "data" / "t10k-labels-idx1-ubyte"

SAMPLE_COUNT = 10


def read_idx_images(path: Path, limit: int) -> list[list[int]]:
    with path.open("rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: bad magic {magic}")
        n = min(count, limit)
        blob = f.read(n * rows * cols)
    stride = rows * cols
    return [list(blob[i * stride:(i + 1) * stride]) for i in range(n)]


def read_idx_labels(path: Path, limit: int) -> list[int]:
    with path.open("rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: bad magic {magic}")
        return list(f.read(min(count, limit)))


def pick_samples() -> list[dict]:
    """One real test digit per class, so the page has something to click."""
    if not (TEST_IMAGES.exists() and TEST_LABELS.exists()):
        print("  MNIST test data not found — building without sample digits")
        return []

    images = read_idx_images(TEST_IMAGES, 400)
    labels = read_idx_labels(TEST_LABELS, 400)

    chosen: dict[int, dict] = {}
    for px, label in zip(images, labels):
        if label not in chosen:
            chosen[label] = {"label": label, "px": px}
        if len(chosen) == SAMPLE_COUNT:
            break
    return [chosen[k] for k in sorted(chosen)]


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(
            f"{WEIGHTS} not found — train first:\n"
            f"  ./bin/train --epochs 20 --hidden 64 --export web/weights.json"
        )

    model = json.loads(WEIGHTS.read_text())
    samples = pick_samples()

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace(
        "/*__WEIGHTS__*/null",
        json.dumps(model, separators=(",", ":")),
    )
    html = html.replace(
        "/*__SAMPLES__*/[]",
        json.dumps(samples, separators=(",", ":")),
    )

    OUTPUT.write_text(html, encoding="utf-8")

    params = sum(len(l["W"]) + len(l["b"]) for l in model["layers"])
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"  wrote {OUTPUT.relative_to(ROOT)}")
    print(f"  {params:,} parameters, {len(samples)} sample digits, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
