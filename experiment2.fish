#!/usr/bin/env fish

uv run scripts/guided_search.py \
  --strategy no_replacement_independent \
  --goals 5 \
  --seeds 100 \
  --k 1 \
  --attempts 10 \
  --rng-seed 42 \
  --full-union --stop-memory 500M \
  data/seed_terms/plenty-houses
or exit $status

uv run scripts/guided_search.py \
  --strategy no_replacement_count \
  --goals 5 \
  --seeds 100 \
  --k 1 \
  --attempts 10 \
  --rng-seed 42 \
  --full-union --stop-memory 250M \
  data/seed_terms/plenty-houses
or exit $status

uv run scripts/guided_search.py \
  --strategy no_replacement_count \
  --goals 5 \
  --seeds 100 \
  --k 1 \
  --attempts 10 \
  --rng-seed 42 \
  --full-union --stop-memory 500M \
  data/seed_terms/plenty-houses
or exit $status

uv run scripts/guided_search.py \
  --strategy no_replacement_count \
  --goals 5 \
  --seeds 100 \
  --k 1 \
  --attempts 10 \
  --rng-seed 42 \
  --full-union --stop-memory 1G \
  --predict-next-memory models/memory_growth.onnx \
  data/seed_terms/plenty-houses
or exit $status