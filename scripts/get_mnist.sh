#!/usr/bin/env bash
# Downloads MNIST into data/ and unpacks it.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data && cd data

BASE="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"

for f in $FILES; do
  if [ -f "$f" ]; then echo "have $f"; continue; fi
  echo "downloading $f.gz"
  curl -fL --retry 3 -o "$f.gz" "$BASE/$f.gz"
  gunzip -f "$f.gz"
done
echo "MNIST ready in data/"
