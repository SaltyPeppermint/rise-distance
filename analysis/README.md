# Guided-search analysis

- `success.ipynb` compares explicitly selected individual
  `data/guided_search/run.*` directories, including cost and baseline plots.
- `grid_search.ipynb` analyzes a `data/guided_search_grid/run.*` experiment.
  It intentionally contains only the reachability summary and reachability
  heatmap. Select a cumulative candidate budget in its `BUDGET` cell.

Shared loading and plotting code lives in `helpers.py` and `plots.py`.
