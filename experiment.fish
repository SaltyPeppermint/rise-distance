#!/usr/bin/env fish

# cargo build --release --bin start --bin candidates --bin verify
# or exit $status

# uv run scripts/generate_problems.py \
#   --starts 100 \
#   --min-size 40 \
#   --max-size 50 \
#   --language math \
#   --seed 123 \
#   --jobs 2 \
#   --max-iters 2000 \
#   --max-nodes 1000000 \
#   --max-time 300 \
#   --max-memory 2G \
#   --min-rss 500M \
#   --goals 10 \
#   --path data/problems/plenty-houses
# or exit $status

uv run scripts/guided_search.py \
  --policy count \
  --start-terms 100 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/plenty-houses
or exit $status

uv run scripts/guided_search.py \
  --policy count \
  --start-terms 100 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 100M \
  data/problems/plenty-houses
or exit $status