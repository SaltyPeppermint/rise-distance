#!/usr/bin/env fish

# Build a reproducible seed corpus and its goal set, then compare guide-candidate
# strategies. Every command stops the script immediately on failure. Guided
# searches choose a fresh data/guided_search/run.N output directory.

# uv run scripts/generate_starts.py \
#   --total-samples 200 \
#   --min-size 10 \
#   --max-size 12 \
#   --language math \
#   --seed 123 \
#   --workers 16 \
#   --max-iters 2000 \
#   --max-nodes 1000000 \
#   --max-time 300 \
#   --path data/start_terms/plenty-houses \
#   --max-memory 500M
# or exit $status

# uv run scripts/generate_goals.py --n 10 data/start_terms/plenty-houses/
# or exit $status


uv run scripts/guided_search.py \
  --strategy no_replacement_uniform \
  --goal-terms 5 \
  --start-terms 5 \
  --k 1 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/start_terms/plenty-houses
or exit $status

# uv run scripts/guided_search.py \
#   --strategy no_replacement_count \
#   --goal-terms 5 \
#   --start-terms 100 \
#   --k 1 \
#   --attempts 10 \
#   --seed 42 \
#   --full-union \
#   --stop-memory 250M \
#   data/start_terms/plenty-houses
# or exit $status