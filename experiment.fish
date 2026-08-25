#!/usr/bin/env fish

# Build a reproducible seed corpus and its goal set, then compare guide-candidate
# strategies. Every command stops the script immediately on failure. Guided
# searches choose a fresh data/guided_search/run.N output directory.

# uv run scripts/generate_seeds.py \
#   --total-samples 200 \
#   --min-size 10 \
#   --max-size 50 \
#   --language math \
#   --seed 123 \
#   --workers 16 \
#   --max-iters 2000 \
#   --max-nodes 1000000 \
#   --max-time 300 \
#   --path data/seed_terms/plenty-houses \
#   --max-memory 500M
# or exit $status

# uv run scripts/generate_goals.py --goals 10 data/seed_terms/plenty-houses/
# or exit $status

# uv run scripts/guided_search.py \
#   --strategy no_replacement_independent \
#   --goals 5 \
#   --seeds 100 \
#   --k 1 \
#   --attempts 10 \
#   --rng-seed 42 \
#   --full-union --stop-memory 100M \
#   data/seed_terms/plenty-houses
# or exit $status

uv run scripts/guided_search.py \
  --strategy no_replacement_count \
  --goals 5 \
  --seeds 20 \
  --k 1 \
  --attempts 10 \
  --rng-seed 42 \
  --full-union --stop-memory 250M \
  data/seed_terms/plenty-houses
or exit $status


uv run scripts/guided_search.py \
  --strategy no_replacement_count \
  --goals 5 \
  --seeds 20 \
  --k 1 \
  --attempts 10 \
  --rng-seed 42 \
  --full-union --stop-memory 250M \
  data/seed_terms/plenty-houses
or exit $status

# uv run scripts/guided_search.py \
#   --strategy no_replacement_independent \
#   --goals 5 \
#   --seeds 100 \
#   --k 1 \
#   --attempts 10 \
#   --rng-seed 42 \
#   --full-union --stop-memory 1000M \
#   data/seed_terms/plenty-houses
# or exit $status

# uv run scripts/guided_search.py \
#   --strategy no_replacement_independent \
#   --goals 5 \
#   --seeds 100 \
#   --k 1 \
#   --attempts 10 \
#   --rng-seed 42 \
#   --full-union --stop-memory 1000M \
#   data/seed_terms/plenty-houses
# or exit $status

# # Locally uniform candidates across guide-replay memory limits.
# uv run scripts/guided_search.py \
#   --strategy no_replacement_naive \
#   --goals 5 \
#   --seeds 100 \
#   --k 1 \
#   --attempts 10 \
#   --rng-seed 42 \
#   --full-union --stop-memory 100M \
#   data/seed_terms/plenty-houses
# or exit $status

# uv run scripts/guided_search.py \
#   --strategy no_replacement_naive \
#   --goals 5 \
#   --seeds 100 \
#   --k 1 \
#   --attempts 10 \
#   --rng-seed 42 \
#   --full-union --stop-memory 250M \
#   data/seed_terms/plenty-houses
# or exit $status

# uv run scripts/guided_search.py \
#   --strategy no_replacement_naive \
#   --goals 5 \
#   --seeds 100 \
#   --k 1 \
#   --attempts 10 \
#   --rng-seed 42 \
#   --full-union --stop-memory 500M \
#   data/seed_terms/plenty-houses
# or exit $status

# uv run scripts/guided_search.py \
#   --strategy no_replacement_naive \
#   --goals 5 \
#   --seeds 100 \
#   --k 1 \
#   --attempts 10 \
#   --rng-seed 42 \
#   --full-union --stop-memory 1000M \
#   data/seed_terms/plenty-houses
# or exit $status
