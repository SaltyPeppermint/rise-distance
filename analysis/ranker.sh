#!/usr/bin/env bash
set -euo pipefail

uv run jupyter nbconvert --to notebook --execute --inplace --log-level=INFO analysis/ranker.ipynb 2>&1 | tee ranker-run.log