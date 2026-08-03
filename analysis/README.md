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
