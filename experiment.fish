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

# memrun uv run scripts/generate_problems.py \
#   --starts 200 \
#   --min-size 30 \
#   --max-size 49 \
#   --language math \
#   --seed 123 \
#   --jobs 8 \
#   --max-iters 2000 \
#   --max-nodes 1000000 \
#   --max-time 300 \
#   --max-memory 500M \
#   --min-rss 500M \
#   --rss-max 1G \
#   --goals 10 \
#   --path data/problems/squishy-potatoe
# or exit $status

memrun uv run scripts/guided_search.py \
  --max-rss 450M \
  --sampling-retries 5 \
  --policy uniform \
  --start-terms 100 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  data/problems/squishy-potatoe
or exit $status

memrun uv run scripts/guided_search.py \
  --max-rss 450M \
  --sampling-retries 5 \
  --policy uniform \
  --start-terms 100 \
  --attempts 10 \
  --seed 42 \
  --full-union \
  --frontier \
  data/problems/squishy-potatoe
or exit $status

# memrun uv run scripts/guided_search.py \
#   --max-rss 450M \
#   --sampling-retries 5 \
#   --policy uniform \
#   --start-terms 100 \
#   --attempts 10 \
#   --seed 42 \
#   --full-union \
#   --frontier \
#   data/problems/squishy-potatoe
# or exit $status
