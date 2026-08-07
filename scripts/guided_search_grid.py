"""Run a paired guided-search grid over sampling policies and size allocations.

For each (size distribution, sampling seed) cell, this runner asks
``guided_search.py`` to generate one maximum-size candidate manifest. Every
strategy in that cell then reuses the exact same manifest. A cell is run once
at ``--attempts``; cumulative prefixes of its result rows give the outcomes at
smaller budgets without regenerating incompatible candidate pools.

The defaults run independent and balanced sampling over greedy, uniform, and
proportional root-size allocation with ten sampling seeds. This is intentionally
a substantial experiment. Build the release binaries before starting:

    cargo build --release --bin sample --bin verify
    uv run scripts/guided_search_grid.py data/seed_terms/inert-angel \
        --stop-nodes 10000 --full-union

Use ``--dry-run`` to print the commands and output layout without executing
them. Completed strategy directories are skipped, so rerunning the same
``--output`` resumes an interrupted grid.
"""

import dataclasses
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    path: tyro.conf.Positional[Path]
    """Seed folder containing `goal_terms.json` and `goal_args.json`."""

    output: Path | None = None
    """Grid output directory. Auto-numbered under `data/guided_search_grid/`
    when omitted."""

    distributions: str = "greedy,uniform,proportional:1"
    """Comma-separated root-size distributions accepted by `sample`."""

    sampling_seeds: str = "0,1,2"
    """Comma-separated Rust candidate-sampling seeds."""

    strategies: str = "no_replacement_independent,no_replacement_balanced"
    """Comma-separated guided-search strategies. All strategies in a cell
    reuse its first strategy's candidate manifest."""

    attempts: int = 250
    """Maximum candidate-prefix budget. Analyze smaller budgets as prefixes."""

    k: int = 1
    full_union: bool = True
    seeds: int | None = 100
    goals: int | None = 5
    rng_seed: int = 42
    jobs: int | None = None

    stop_iters: int | None = None
    stop_nodes: int | None = None
    stop_time: float | None = None
    stop_memory: str | None = None
    """Path to the ONNX next-iteration memory model forwarded to every guided
    search. Requires an effective memory ceiling."""

    sample_binary: Path = Path("target/release/sample")
    verify_binary: Path = Path("target/release/verify")
    driver: Path = Path("scripts/guided_search.py")

    dry_run: bool = False
    """Print commands without running them or creating output directories."""


def comma_values(raw: str, what: str) -> list[str]:
    """Parse a non-empty, comma-separated CLI grid dimension."""
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{what} must contain at least one value")
    if len(values) != len(set(values)):
        raise ValueError(f"{what} contains duplicate values: {raw}")
    return values


def resolve_output_dir(requested: Path | None, dry_run: bool) -> Path:
    """Choose a new grid run directory without mutating it in dry-run mode."""
    if requested is not None:
        out = requested
    else:
        base = Path("data/guided_search_grid")
        existing = [int(p.suffix[1:]) for p in base.glob("run.*") if p.suffix[1:].isdigit()]
        out = base / f"run.{max(existing, default=0) + 1}"
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
    return out


def safe_component(value: str) -> str:
    """Make a readable directory component from a CLI enum-like value."""
    return value.replace(":", "-").replace("/", "-")


def optional_flag(cmd: list[str], name: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def driver_command(
    args: Args,
    *,
    distribution: str,
    sampling_seed: int,
    strategy: str,
    candidate_pools: list[str],
    output: Path,
    samples_input: Path | None,
    unguided_input: Path | None,
) -> list[str]:
    """Construct one guided_search.py invocation."""
    cmd = [
        sys.executable,
        str(args.driver),
        str(args.path),
        "--output",
        str(output),
        "--sample-binary",
        str(args.sample_binary),
        "--verify-binary",
        str(args.verify_binary),
        "--attempts",
        str(args.attempts),
        "--strategy",
        strategy,
        "--candidate-pools",
        *candidate_pools,
        "--k",
        str(args.k),
        "--rng-seed",
        str(args.rng_seed),
        "--sampling-seed",
        str(sampling_seed),
        "--size-distribution",
        distribution,
    ]
    if args.full_union:
        cmd.append("--full-union")
    optional_flag(cmd, "--seeds", args.seeds)
    optional_flag(cmd, "--goals", args.goals)
    optional_flag(cmd, "--jobs", args.jobs)
    optional_flag(cmd, "--stop-iters", args.stop_iters)
    optional_flag(cmd, "--stop-nodes", args.stop_nodes)
    optional_flag(cmd, "--stop-time", args.stop_time)
    optional_flag(cmd, "--stop-memory", args.stop_memory)
    optional_flag(cmd, "--samples-input", samples_input)
    optional_flag(cmd, "--unguided-input", unguided_input)
    return cmd


def candidate_pool(strategy: str) -> str:
    """Map a driver strategy to the candidate key emitted by `sample`."""
    if strategy.startswith("smallest_"):
        return strategy
    for suffix in ("independent", "naive", "balanced"):
        if strategy.endswith(f"_{suffix}"):
            return f"sample_{suffix}"
    raise ValueError(f"unknown strategy {strategy!r}")


def main() -> int:
    args = tyro.cli(Args, description=__doc__)
    if all(v is None for v in (args.stop_iters, args.stop_nodes, args.stop_time, args.stop_memory)):
        print(
            "No guide-replay budget given; pass at least one of "
            "--stop-iters/--stop-nodes/--stop-time/--stop-memory.",
            file=sys.stderr,
        )
        return 2
    if args.attempts < 1 or args.k < 1:
        print("--attempts and --k must both be positive", file=sys.stderr)
        return 2

    try:
        distributions = comma_values(args.distributions, "distributions")
        strategies = comma_values(args.strategies, "strategies")
        sampling_seeds = [
            int(value) for value in comma_values(args.sampling_seeds, "sampling_seeds")
        ]
        candidate_pools = list(dict.fromkeys(candidate_pool(strategy) for strategy in strategies))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    out = resolve_output_dir(args.output, args.dry_run)
    if not args.dry_run:
        (out / "grid_config.json").write_text(
            json.dumps(
                {
                    **dataclasses.asdict(args),
                    "output": str(out),
                    "distributions_expanded": distributions,
                    "sampling_seeds_expanded": sampling_seeds,
                    "strategies_expanded": strategies,
                    "prefix_budgets": sorted(
                        {
                            args.attempts,
                            *(
                                budget
                                for budget in (1, 2, 5, 10, 20, 50, 100, 250, 500, 1000)
                                if budget <= args.attempts
                            ),
                        }
                    ),
                },
                indent=2,
                default=str,
            )
        )

    total = len(distributions) * len(sampling_seeds) * len(strategies)
    first_output = (
        out
        / f"distribution.{safe_component(distributions[0])}"
        / f"sampling_seed.{sampling_seeds[0]}"
        / safe_component(strategies[0])
    )
    unguided_manifest = first_output / "unguided_results.parquet"
    cell_number = 0
    for distribution in distributions:
        for sampling_seed in sampling_seeds:
            cell = (
                out
                / f"distribution.{safe_component(distribution)}"
                / f"sampling_seed.{sampling_seed}"
            )
            producer_out = cell / safe_component(strategies[0])
            manifest = producer_out / "sample_run" / "samples.json"

            for strategy_index, strategy in enumerate(strategies):
                cell_number += 1
                strategy_out = cell / safe_component(strategy)
                result = strategy_out / "results.parquet"
                comparison = strategy_out / "comparison.parquet"
                unguided = strategy_out / "unguided_results.parquet"
                producer_has_provenance = strategy_index > 0 or manifest.is_file()
                if (
                    result.is_file()
                    and comparison.is_file()
                    and unguided.is_file()
                    and producer_has_provenance
                    and not args.dry_run
                ):
                    print(f"[{cell_number}/{total}] already complete: {strategy_out}")
                    continue

                # The first strategy creates the manifest. On resume it can also
                # reuse an already-created manifest if verification was interrupted.
                samples_input = manifest if manifest.is_file() or strategy_index > 0 else None
                if strategy_index > 0 and not manifest.is_file() and not args.dry_run:
                    print(
                        f"Missing producer manifest {manifest}; cannot run paired strategy "
                        f"{strategy!r}. Rerun the grid to regenerate the producer cell.",
                        file=sys.stderr,
                    )
                    return 1
                unguided_input = None if strategy_out == first_output else unguided_manifest
                if unguided_input is not None and not unguided_input.is_file() and not args.dry_run:
                    print(
                        f"Missing shared unguided baseline {unguided_input}; "
                        "rerun the first grid cell.",
                        file=sys.stderr,
                    )
                    return 1

                cmd = driver_command(
                    args,
                    distribution=distribution,
                    sampling_seed=sampling_seed,
                    strategy=strategy,
                    candidate_pools=candidate_pools,
                    output=strategy_out,
                    samples_input=samples_input,
                    unguided_input=unguided_input,
                )
                print(
                    f"[{cell_number}/{total}] distribution={distribution} "
                    f"sampling_seed={sampling_seed} strategy={strategy}"
                )
                if args.dry_run:
                    # Show the path the producer will create to make the pairing
                    # explicit even though it does not exist during a dry run.
                    if strategy_index > 0 and samples_input is None:
                        cmd.extend(["--samples-input", str(manifest)])
                    print(shlex.join(cmd))
                    continue

                completed = subprocess.run(cmd, check=False)
                if completed.returncode != 0:
                    print(
                        f"Grid stopped after failed cell {strategy_out} "
                        f"(exit {completed.returncode}). Rerun with --output {out} to resume.",
                        file=sys.stderr,
                    )
                    return completed.returncode

    print(f"{'Would write' if args.dry_run else 'Wrote'} grid under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
