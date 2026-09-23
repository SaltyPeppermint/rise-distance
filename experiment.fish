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
#   --starts 1000 \
#   --min-size 30 \
#   --max-size 60 \
#   --language math \
#   --seed 123 \
#   --jobs 20 \
#   --max-iters 2000 \
#   --max-nodes 1000000 \
#   --max-time 300 \
#   --max-memory 500M \
#   --min-rss 500M \
#   --rss-max 1G \
#   --goals 2 \
#   --path data/problems/expensive-bird
# or exit $status

set -l search_args \
  --max-rss 450M \
  --sample-policy uniform \
  --start-terms 100 \
  --n-guides 10 \
  --seed 42 \
  --full-union

memrun uv run scripts/guided_search.py $search_args \
  --sampling-backoff 5 \
  --max-depth 1 \
  --output data/guided_search/depth1-backoff-5 \
  data/problems/expensive-bird
or exit $status

memrun uv run scripts/guided_search.py $search_args \
  --sampling-backoff 5 \
  --max-depth 2 \
  --max-attempts 30 \
  --max-total-time 300 \
  --search-policy depth \
  --output data/guided_search/depth2-death-backoff-5 \
  data/problems/expensive-bird
or exit $status

memrun uv run scripts/guided_search.py $search_args \
  --sampling-backoff 5 \
  --max-depth 2 \
  --max-attempts 30 \
  --max-total-time 300 \
  --search-policy width \
  --output data/guided_search/depth2-width-backoff-5 \
  data/problems/expensive-bird
or exit $status

memrun uv run scripts/guided_search.py $search_args \
  --sampling-backoff 20 \
  --max-depth 1 \
  --output data/guided_search/depth1-backoff-20 \
  data/problems/expensive-bird
or exit $status

memrun uv run scripts/guided_search.py $search_args \
  --sampling-backoff 20 \
  --max-depth 2 \
  --max-attempts 30 \
  --max-total-time 300 \
  --search-policy depth \
  --output data/guided_search/depth2-death-backoff-20 \
  data/problems/expensive-bird
or exit $status

memrun uv run scripts/guided_search.py $search_args \
  --sampling-backoff 20 \
  --max-depth 2 \
  --max-attempts 30 \
  --max-total-time 300 \
  --search-policy width \
  --output data/guided_search/depth2-width-backoff-20 \
  data/problems/expensive-bird
or exit $status
