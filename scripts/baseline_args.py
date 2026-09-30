"""The flags the unguided baseline depends on, shared by `baseline.py` and
`guided_search.py`, and the check that a stored baseline still fits them."""

import json
import os
from pathlib import Path

import polars as pl
from pydantic import Field
from pydantic_settings import BaseSettings, CliPositionalArg, SettingsConfigDict

from common import Problem, cli_flags, parse_size


class BaselineArgs(BaseSettings):
    """The flags the unguided baseline depends on, shared with `guided_search.py`."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_implicit_flags=True)

    # I/O
    path: CliPositionalArg[Path] = Field(
        description=(
            "Problem folder with `problems.json` and `problem_args.json` "
            "(both written by `generate_problems.py`)."
        )
    )

    output: Path = Field(description="Baseline folder.")

    attempt_bin: Path = Field(
        default=Path("target/release/attempt"), description="Path to the attempt binary."
    )

    # Guide-replay budget
    #
    # At least one must be given. Replay ends when the first configured budget
    # is exhausted; omitted budgets are effectively unlimited.
    stop_iters: int | None = Field(
        default=None, gt=0, description=("Guide-replay iteration budget.")
    )

    stop_nodes: int | None = Field(
        default=None, gt=0, description=("Guide-replay egraph-node budget.")
    )

    stop_time: float | None = Field(
        default=None, gt=0, description=("Guide-replay wall-clock budget in seconds.")
    )

    max_rss: str = Field(
        default="4G",
        description=(
            "Cap each `attempt` process (and, in `guided_search.py`, each `sample` "
            "process) at this cgroup RSS limit, as a human size such as `4G`."
        ),
    )

    start_terms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only process the first N start terms in sorted order. All start terms are processed if omitted."
        ),
    )

    goal_terms: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only use the first N goals per start term in file order. All goals are used if omitted."
        ),
    )

    jobs: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        gt=0,
        description=(
            "Maximum number of concurrent `sample`/`attempt` processes. Defaults to `os.cpu_count()`."
        ),
    )

    @property
    def max_rss_bytes(self) -> int:
        return parse_size(self.max_rss)

    @property
    def language(self) -> str:
        """The language the problems were generated in."""
        return str(json.loads((self.path / "problem_args.json").read_text())["language"])

    @property
    def limits(self) -> dict[str, int | float]:
        """The configured guide-replay budgets, without the omitted ones."""
        budgets = {
            "max_iters": self.stop_iters,
            "max_nodes": self.stop_nodes,
            "max_time": self.stop_time,
        }
        return {key: value for key, value in budgets.items() if value is not None}

    def base_flags(self) -> list[str]:
        """Flags shared by every `attempt` process."""
        return cli_flags(language=self.language, **self.limits)

    def baseline_key(self) -> dict:
        """Everything a baseline row depends on besides its pair.

        Two runs with the same key can share a baseline.
        """
        return {
            "problems": str(self.path.resolve()),
            "stop_iters": self.stop_iters,
            "stop_nodes": self.stop_nodes,
            "stop_time": self.stop_time,
            "max_rss": self.max_rss_bytes,
        }


class BaselineMismatch(RuntimeError):
    """A stored baseline that was computed under other flags or misses pairs."""


def check_baseline(args: BaselineArgs, directory: Path, pairs: list[Problem]) -> None:
    """Check that the stored baseline in `directory` fits `args` and `pairs`.

    Raises `BaselineMismatch` if there is no finished baseline in `directory`,
    or it was computed under a different `baseline_key` or does not cover every pair.
    """
    if not (directory / "config.json").is_file():
        raise BaselineMismatch(f"no finished baseline in {directory}; run `baseline.py` first")
    config = json.loads((directory / "config.json").read_text())
    expected = args.baseline_key()
    stored = config["baseline_key"]
    if stored != expected:
        diff = {
            key: (stored.get(key), value)
            for key, value in expected.items()
            if stored.get(key) != value
        }
        raise BaselineMismatch(f"baseline in {directory} differs (stored, wanted): {diff}")

    wanted = pl.DataFrame(
        {"start_term": [p.start for p in pairs], "goal_term": [p.goal for p in pairs]},
        schema={"start_term": pl.String, "goal_term": pl.String},
    )
    rows = pl.read_parquet(directory / "unguided_results.parquet")
    missing = wanted.join(rows, on=["start_term", "goal_term"], how="anti")
    if len(missing):
        raise BaselineMismatch(f"baseline in {directory} misses {len(missing)} pair(s)")
