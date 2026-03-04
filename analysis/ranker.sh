#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLBACKEND=Agg

uv run jupyter nbconvert --to script --stdout ranker.ipynb | uv run python - 2>&1
