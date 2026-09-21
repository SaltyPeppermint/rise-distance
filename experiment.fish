#!/usr/bin/env fish

function memrun
    systemd-run \
        --user \
        --wait \
        --pipe \
        --same-dir \
        --property=MemoryAccounting=yes \
        $argv
end

cargo build --release
or exit $status

memrun uv run scripts/generate_problems.py \
  --starts 1000 \
  --min-size 30 \
  --max-size 60 \
  --language math \
  --seed 123 \
  --jobs 20 \
  --max-iters 2000 \
  --max-nodes 1000000 \
  --max-time 300 \
  --max-memory 500M \
  --min-rss 500M \
  --rss-max 1G \
  --goals 2 \
  --path data/problems/expensive-bird
or exit $status

# memrun uv run scripts/guided_search.py \
#   --max-rss 450M \
#   --sampling-backoff 5 \
#   --sample-policy uniform \
#   --start-terms 100 \
#   --n-guides 10 \
#   --seed 42 \
#   --full-union \
#   data/problems/expensive-bird
# or exit $status
