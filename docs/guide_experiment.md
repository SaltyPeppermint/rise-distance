# Guide experiment pipeline

This is an operational guide. See
[random term generation](generation/random_terms.md) for stage 1 and
[exact novel-candidate construction](candidates/exact_novel_candidates.md) and
[exact frontier drawing policies](candidates/exact_frontier_drawing.md) for the
guide phase.

The guide experiment measures how well constructed guide candidates steer
equality saturation toward a goal. It runs in four stages, each feeding the
next through a seed folder (`data/seed_terms/<name>/`) that accumulates
`terms.json` and `args.json`:

```
generate ──▶ goal ──▶ candidates ──▶ verify
(seeds)     (goals)   (menu)          (search legs)
   └──── generate_seeds.py / generate_goals.py ────┘
                         guided_search.py orchestrates candidates + verify
```

- **generate** samples random seed terms and measures their peak memory.
- **goal** replays a big eqsat per seed and records goal terms to reach.
- **candidates** replays the guide phase and emits the guide-candidate menu.
- **verify** runs one search leg (union guides, saturate, check the goal).

`guided_search.py` runs `candidates` once per seed, then drives many parallel
`verify` legs. Logging and data wrangling live in Python; the Rust binaries are
stateless workers.

---

## 0. Build the binaries

```bash
cargo build --release \
    --bin generate \
    --bin goal --bin candidates --bin verify
```

## 1. Generate seed terms (skip if you already have a seed folder)

Writes `data/seed_terms/<auto-name>/terms.json` and `generation_args.json`.
Pick the language here—it flows through every later stage. Generated terms
have exact sizes; only those that pass the eqsat validity check are kept.

```bash
uv run scripts/generate_seeds.py \
    --total-samples 100 --min-size 10 --max-size 50 \
    --language math --seed 42 \
    --max-memory 8G \
    --max-iters 200 --max-nodes 1000000 --max-time 10 \
    --backoff-scheduler
```

Note the printed output dir (e.g. `data/seed_terms/dusky-cramp`) and use it
below. Pass `--path data/seed_terms/<name>` to choose the name yourself.

## 2. Enrich seeds with goal terms

`generate_goals.py` runs the `goal` binary once per seed and writes
`goal_terms.json` plus `goal_args.json`. Start with `--seeds` to keep it quick.

```bash
uv run scripts/generate_goals.py data/seed_terms/dusky-cramp \
    --goals 10 --seeds 3
```

## 3. Run the driver (candidate construction + parallel search legs + parquet)

`guided_search.py` calls `candidates` internally, then fans out `verify` legs
across cores. Pass at least one guide-replay `--stop-*` limit.

```bash
uv run scripts/guided_search.py data/seed_terms/dusky-cramp \
    --stop-memory 4G --seeds 3 --attempts 5 --k 10 \
    --strategy no_replacement_independent
```

Output lands in `data/guided_search/run.N/` (or pass `--output <dir>`):

- `results.parquet` / `results.json` — one row per `(seed, goal, k, attempt)`
- `config.json` — the driver args
- `candidate_run/candidates.json` — the guide-candidate menu produced

---

## Running a paired policy grid

`guided_search_grid.py` constructs one shared candidate manifest for every
size-allocation/candidate-seed cell. Each policy reuses that manifest, and
smaller attempt budgets can be analyzed as prefixes of the same run:

```bash
uv run scripts/guided_search_grid.py data/seed_terms/dusky-cramp \
    --stop-memory 4G --k 10 --attempts 250 --full-union
```

`analysis.helpers.load_grid` reads the paired parquet outputs for plotting.

## Repository experiment script

[`experiment.fish`](../experiment.fish) runs a concrete end-to-end experiment
for `data/seed_terms/plenty-houses`:

1. Generate 1,000 validated Math seed terms of sizes 10 through 50.
2. Generate ten goal terms per retained seed.
3. Run exact balanced and naive guide-candidate policies at 100M, 250M, 500M,
   and 1000M guide-replay memory limits.

Run it from the repository root:

```bash
fish experiment.fish
```

Each command is followed by `or exit $status`, so a failed stage prevents later
runs from consuming incomplete inputs. Each guided search creates a fresh
`data/guided_search/run.N` directory containing its candidate manifest,
attempt results, pair summaries, unguided baseline, and joined comparison.

## Key knobs

| Flag | Meaning |
| --- | --- |
| `--k N` | Guide-set size (guides unioned per leg). |
| `--attempts N` | Legs per `(seed, goal, k)`; each resamples a fresh subset. Counts the first try. |
| `--strategy` | Candidate policy, including independent, naive, balanced, and smallest variants. |
| `--candidate-pools` | Pools to generate in a shared manifest. Defaults to only the pool selected by `--strategy`; the grid driver supplies all pools needed by its paired strategies. |
| `--full-union` | Union guide nodes by their origin e-class (experimental; helped reachability historically). |
| `--candidate-seed N` | Rust candidate-construction seed. |
| `--size-allocation` | Root-size allocation: `greedy` or `uniform` |
| `--novel-size-goal N` | Number of novel sizes exact construction must find. |
| `--jobs N` | Concurrent `verify` legs (default `os.cpu_count()`). Lower it if the large leg egraphs exhaust RAM. |
| `--seeds N` | Only process the first N seeds. |

The `balanced` pool is diversified while frontier terms are constructed.
`independent` and `naive` draw each candidate
independently, using count-proportional and equal-local-choice weighting
respectively. The
`with_replacement_*` / `no_replacement_*` prefix is a separate Python-side
policy for selecting guide subsets from the resulting finite pool.

## Scaling notes

- Total legs per goal are bounded by `attempts`. Every leg runs a full eqsat,
  so budget wall time and watch RAM on large grids.
- Reaches are strongly k-dependent (historically ~20% at k=1, ~90% at k=100).
  A single small k on a hard seed can legitimately reach 0%; increase `--k` and
  `--attempts`, and try `--full-union`, before concluding a goal is unreachable.
