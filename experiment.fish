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
  --starts 100 \
  --min-size 40 \
  --max-size 50 \
  --language math \
  --seed 123 \
  --jobs 8 \
  --max-iters 2000 \
  --max-nodes 1000000 \
  --max-time 300 \
  --max-memory 2G \
  --min-rss 500M \
  --rss-max 4G \
  --goals 10 \
  --path data/problems/squishy-potatoe
or exit $status

memrun uv run scripts/guided_search.py \
  --sampling-rss-max 1000M \
  --sampling-retries 5 \
  --policy count \
  --start-terms 100 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/squishy-potatoe
or exit $status

memrun uv run scripts/guided_search.py \
  --policy count \
  --start-terms 100 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/squishy-potatoe
or exit $status

memrun uv run scripts/guided_search.py \
  --sampling-rss-max 2000M \
  --sampling-retries 5 \
  --policy count \
  --start-terms 100 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/squishy-potatoe
or exit $status

memrun uv run scripts/guided_search.py \
  --sampling-rss-max 1000M \
  --sampling-retries 5 \
  --policy count \
  --start-terms 100 \
  --attempts 20 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/squishy-potatoe
or exit $status

memrun uv run scripts/guided_search.py \
  --sampling-rss-max 1000M \
  --sampling-retries 5 \
  --policy uniform \
  --start-terms 100 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/squishy-potatoe
or exit $status

memrun uv run scripts/guided_search.py \
  --sampling-rss-max 1000M \
  --sampling-retries 5 \
  --policy uniform \
  --start-terms 100 \
  --attempts 20 \
  --seed 42 \
  --full-union \
  --stop-memory 250M \
  data/problems/squishy-potatoe
or exit $status