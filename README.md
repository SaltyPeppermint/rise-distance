# rise-distance

## Setup

This project uses a patched version of [`egg`](https://github.com/SaltyPeppermint/egg)
as a Git submodule in `vendor/egg`.

When cloning the project, initialize the submodule at the same time:

```bash
git clone --recurse-submodules <repository-url>
cd rise-distance
```

If the project has already been cloned, initialize the submodule with:

```bash
git submodule update --init --recursive
```

Run the same update command after pulling changes that move the submodule to a
different commit.

## Running experiments

```bash
uv run scripts/experiment.py --language math   # or prop
uv run scripts/experiment.py --language prop --rerun-aborted
```

This builds the Rust bins, generates the benchmarks, computes the baseline, and does the guided-search in "benchmode"

## Data & analysis

See `analysis/README.md`
