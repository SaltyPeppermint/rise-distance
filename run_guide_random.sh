#!/usr/bin/env bash
set -euo pipefail

strategies=("naive" "count-based")
full_union_opts=("" "--full-union")

for strategy in "${strategies[@]}"; do
  for full_union in "${full_union_opts[@]}"; do
    echo "=== strategy=$strategy full-union=${full_union:-no} ==="
    cargo run --release --bin guide-random -- \
      -s "(d x (- (pow x 3) (* 7 (pow x 2))))" \
      -n 10 -i 5 --max-size 20 --goals 100 --guides 10000 --eval-all \
      --guide-sample-strategy "$strategy" \
      $full_union
  done
done
