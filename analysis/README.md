# Analysis notebooks

Guided-search experiments plus the eqsat memory model.

- `success.ipynb` compares success and whole-process peak RSS for selected
  `data/guided_search/run.*` directories. Peak-memory ratios use only
  seed/goal pairs reached by both guided and pair-matched unguided runs.
- `grid_search.ipynb` compares grid-policy success across cumulative attempt
  budgets and full-budget peak RSS.
- `memory_prediction.ipynb` explores upcoming-iteration memory prediction from
  the per-iteration traces in `data/start_terms/*/terms.json`. The deployed
  pipeline in `iteration_data.py` builds explicit pre-search decision rows and
  predicts the iteration-local peak, not the later final live allocation.

Shared statistics and plotting code lives in `helpers.py` and `plots.py`.
The memory model has its own `iteration_data.py` (trace loading),
`memory_model.py` (fitting and scoring), and `memory_plots.py`, since it reads
seed-term traces rather than guided-search runs.

## Exporting the predictive stop model

`export_memory_model.py` fits the scheduler-aware gradient-boosting peak model
on the newest compatible seed trace by default and writes:

- `models/memory_growth.onnx`, loaded from the path passed to Rust;
- `models/memory_growth.json`, including the exact feature order and the 99th
  percentile grouped out-of-fold underprediction residual used as a safety
  margin;
- `models/memory_growth_evaluation.json`, with held-out sklearn/ONNX parity,
  peak-based first-stop replay at several ceilings, per-responsible-rule tail
  results, the four scheduler-feature ablations, classifier comparison, and
  single-row inference overhead.

Run `uv run python analysis/export_memory_model.py`; use `--seed-dir` to select
a trace explicitly. The model is calibrated only on this repository's
seed-term distribution. Rust therefore leaves it off unless both
`--max-memory` and `--predict-next-memory models/memory_growth.onnx` are
supplied. Each eqsat run loads and parity-prewarms its own session before
capturing the heap baseline; the adjacent JSON manifest is the authoritative
ordered schema and exact BackoffScheduler rule set. Any different or incomplete
rule set fails before the first rewrite search. Multi-guide
`verify_reachability` runs retain only the hard limit because they do not have
a training-compatible single input term size.
