# Guide experiment pipeline

The guide experiment measures how well sampled guide terms steer equality
saturation toward a goal. It runs in four stages, each feeding the next through
a seed folder (`data/seed_terms/<name>/`) that accumulates `terms.json` and
`args.json`:

```
generate ──▶ goal ──▶ sample ──▶ verify
(seeds)     (goals)   (menu)      (search legs)
   └──────── generate_and_measure.py         └── driver.py orchestrates ──┘
```

- **generate** samples random seed terms and measures their peak memory.
- **goal** replays a big eqsat per seed and records goal terms to reach.
- **sample** replays the guide phase and emits the guide-candidate menu.
- **verify** runs one search leg (union guides, saturate, check the goal).

`driver.py` runs `sample` once, then drives many parallel `verify` legs. All
logging and data wrangling live in Python; the Rust binaries are stateless
workers.

---

## 0. Build the binaries

```bash
cargo build --release \
    --bin generate \
    --bin goal --bin sample --bin verify
```

## 1. Generate seed terms (skip if you already have a seed folder)

Writes `data/seed_terms/<auto-name>/terms.json` (+ `args.json`). Pick the
language here — it flows through every later stage.

```bash
uv run scripts/generate_seeds.py \
    --total-samples 100 --min-size 10 --max-size 50 \
    --distribution uniform --language math --seed 42 \
    --max-memory 8G \
    --max-iters 200 --max-nodes 1000000 --max-time 10 \
    --backoff-scheduler
```

Note the printed output dir (e.g. `data/seed_terms/dusky-cramp`) and use it
below. Pass `--path data/seed_terms/<name>` to choose the name yourself.

## 2. Enrich seeds with goal terms

`goal` runs a full eqsat per seed and rewrites `terms.json` **in place** with
goal terms + guide/goal-phase metadata. Start with `--take-first` to keep it
quick; enrich more later by re-running.

```bash
target/release/goal data/seed_terms/dusky-cramp --goals 10 --take-first 3
```

> `goal` overwrites `terms.json`. To keep the raw seeds, copy the folder first
> (`cp -r data/seed_terms/dusky-cramp /tmp/run1`) and point the rest at the copy.

## 3. Run the driver (sample + parallel search legs + parquet)

`driver.py` calls `sample` internally, then fans out `verify` legs across cores.
Keep `--take-first` in sync with stage 2 — the driver skips any seed that wasn't
enriched.

```bash
uv run scripts/driver.py data/seed_terms/dusky-cramp \
    --take-first 3 --attempts 5 --k 10 \
    --strategy no_replacement_count
```

Output lands in `data/guide_driver/run.N/` (or pass `--output <dir>`):

- `results.parquet` / `results.json` — one row per `(seed, goal, k, attempt)`
- `config.json` — the driver args
- `sample_run/samples.json` — the guide-candidate menu `sample` produced

---

## Reproducing the reach-rate-vs-k curve

`--k` takes a list; each `(seed, goal)` pair is tried independently at every k,
with `--attempts` resampled subsets per k (early-stops on first reach). This is
the old `guide` experiment's sweep:

```bash
uv run scripts/driver.py data/seed_terms/dusky-cramp \
    --k 1 2 5 10 50 100 --attempts 20 \
    --strategy no_replacement_count --full-union
```

The driver prints reach rate per k at the end, and
`analysis/helpers.load_driver_run` + `compute_goal_reach` read the parquet for
plotting.

## Key knobs

| Flag | Meaning |
| --- | --- |
| `--k <ints…>` | Guide-set sizes to sweep (guides unioned per leg). |
| `--attempts N` | Legs per `(seed, goal, k)`; each resamples a fresh subset. Counts the first try. |
| `--strategy` | Candidate pool: `no_replacement_count`, `with_replacement_count`, `no_replacement_naive`, `with_replacement_naive`, `smallest_novel`, `smallest_overall`. |
| `--full-union` | Union guide nodes by their origin e-class (experimental; helped reachability historically). |
| `--samples-per-strategy N` | Menu size `sample` draws per strategy (default 1000). Keep ≥ largest `k`. Values below ~30 can trip a pre-existing bigint sampler panic. |
| `--jobs N` | Concurrent `verify` legs (default `os.cpu_count()`). Lower it if the large leg egraphs exhaust RAM. |
| `--take-first N` | Only process the first N seeds. |

## Scaling notes

- Total legs per goal = `len(k) × attempts`. A `--k 1 2 5 10 50 100 --attempts
  500` run is up to 3000 legs **per goal**, each a full eqsat — budget wall time
  and watch RAM.
- Reaches are strongly k-dependent (historically ~20% at k=1, ~90% at k=100).
  A single small k on a hard seed can legitimately reach 0%; widen `--k` and
  `--attempts`, and try `--full-union`, before concluding a goal is unreachable.
