# Analysis notebooks

- `success.ipynb` compares success and peak RSS for selected
  `data/<experiment>/<language>/guided_search/<n>_*` runs, against
  the unguided baseline in `baseline/` each run names and the brute-force cost
  in `problems/problems.json`.

Shared statistics and plotting code lives in `helpers.py` and `plots.py`.
