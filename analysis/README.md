# Analysis notebooks

Guided-search experiments plus the eqsat memory model.

- `success.ipynb` compares explicitly selected individual
  `data/guided_search/run.*` directories, including cost and baseline plots.
- `grid_search.ipynb` analyzes a `data/guided_search_grid/run.*` experiment.
  It intentionally contains only the reachability summary and reachability
  heatmap. Select a cumulative candidate budget in its `BUDGET` cell.
- `memory_prediction.ipynb` predicts an eqsat iteration's memory use from the
  previous iteration, using the per-iteration traces in
  `data/seed_terms/*/terms.json`. Select a seed folder in its `SEED_DIR` cell.
  A closing section sweeps the history window (`build_transitions(window=n)`)
  to test whether several past iterations beat a single one.

Shared loading and plotting code lives in `helpers.py` and `plots.py`.
The memory model has its own `iteration_data.py` (trace loading),
`memory_model.py` (fitting and scoring), and `memory_plots.py`, since it reads
seed-term traces rather than guided-search runs.

## Exporting the predictive stop model

`export_memory_model.py` fits the scalar-only gradient-boosting model on the
newest seed trace by default and writes:

- `models/memory_growth.onnx`, loaded from the path passed to Rust;
- `models/memory_growth.json`, including the exact feature order and the 99th
  percentile grouped out-of-fold underprediction residual used as a safety
  margin;
- `models/memory_growth_evaluation.json`, with held-out sklearn/ONNX parity,
  replay results at several memory ceilings, and single-row inference overhead.

Run `uv run python analysis/export_memory_model.py`; use `--seed-dir` to select
a trace explicitly. The model is calibrated only on this repository's
seed-term distribution. Rust therefore leaves it off unless both
`--max-memory` and `--predict-next-memory models/memory_growth.onnx` are
supplied. Each eqsat run loads and prewarms its own session before capturing
the heap baseline; the JSON manifest must be adjacent to the model with the
same stem. Multi-guide `verify_reachability` runs retain only the hard limit
because they do not have a training-compatible single input term size.
