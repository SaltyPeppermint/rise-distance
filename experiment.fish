#!/usr/bin/env fish

uv run scripts/generate_seeds.py \
  --total-samples 1000 \
  --min-size 10 \
  --max-size 50 \
  --language math \
  --seed 123 \
  --workers 16 \
  --max-iters 2000 \
  --max-nodes 1000000 \
  --max-time 300 \
  --path data/seed_terms/plenty-houses \
  --max-memory 500M
or exit $status

uv run scripts/generate_goals.py --goals 10 data/seed_terms/plenty-houses/
or exit $status

uv run analysis/export_memory_model.py \
  --seed-dir data/seed_terms/plenty-houses \
  --output-dir models
or exit $status

uv run scripts/guided_search.py \
  --strategy no_replacement_independent \
  --goals 5 \
  --seeds 100 \
  --k 1 \
  --attempts 10 \
  --rng-seed 42 \
  --full-union --stop-memory 250M \
  data/seed_terms/plenty-houses
or exit $status

uv run scripts/guided_search.py \
  --strategy no_replacement_independent \
  --goals 5 \
  --seeds 100 \
  --k 1 \
  --attempts 10 \
  --rng-seed 42 \
  --full-union --stop-memory 1G \
  --predict-next-memory models/memory_growth.onnx \
  data/seed_terms/plenty-houses
or exit $status
