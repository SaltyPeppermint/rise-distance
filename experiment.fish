#!/usr/bin/env fish

# Build a reproducible problem set (start terms, their frontier goals, and the
# unguided peak RSS each pair really needs), then compare guide-candidate
# strategies. Every command stops the script immediately on failure. Guided
# searches choose a fresh data/guided_search/run.N output directory.

cargo build --release --bin start --bin candidates --bin verify
or exit $status

uv run scripts/generate_problems.py \
  --starts 200 \
  --min-size 10 \
  --max-size 12 \
  --language math \
  --seed 123 \
  --jobs 16 \
  --max-iters 2000 \
  --max-nodes 1000000 \
  --max-time 300 \
  --max-memory 500M \
  --min-rss 500M \
  --goals 10 \
  --path data/problems/plenty-houses
or exit $status


uv run scripts/guided_search.py \
  --policy uniform \
  --start-terms 100 \
  --goal-terms 5 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/plenty-houses
or exit $status

uv run scripts/guided_search.py \
  --policy count \
  --start-terms 100 \
  --goal-terms 5 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/plenty-houses
or exit $status