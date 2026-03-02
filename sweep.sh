#!/usr/bin/env bash
set -euo pipefail

SEED="(d x (- (pow x 3) (* 7 (pow x 2))))"
N=10
I=5
G=100000
MAX_SIZE=100
TOP=10

STRATEGIES=(overlap random)
DISTRIBUTIONS=(uniform "proportional:100" "normal:2.0")
DISTANCES=(zhang-shasha structural)

for strategy in "${STRATEGIES[@]}"; do
  for distribution in "${DISTRIBUTIONS[@]}"; do
    for distance in "${DISTANCES[@]}"; do
      echo "=== strategy=$strategy distribution=$distribution distance=$distance ==="
      cargo run --release --bin guide-eval -- \
        -s "$SEED" \
        -n "$N" -i "$I" -g "$G" \
        --max-size "$MAX_SIZE" \
        --strategy "$strategy" \
        --distribution "$distribution" \
        --distance "$distance" \
        --top "$TOP"
    done
  done
done
