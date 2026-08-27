# Analysis notebooks

- `success.ipynb` compares success and whole-process peak RSS for selected
  `data/guided_search/run.*` directories. Peak-memory ratios use only
  start/goal pairs reached by both guided and pair-matched unguided runs.

Shared statistics and plotting code lives in `helpers.py` and `plots.py`.
`helpers.problem_pairs` loads the `problems.json` a run was built from
(`generate_problems.py`), including each pair's generation-time unguided
measurement.
