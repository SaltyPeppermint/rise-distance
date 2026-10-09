"""The arguments shared by `baseline.py` and `guided_search.py`."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from common import cli_flags, parse_size


class ReplayArgs(BaseModel):
    """The arguments `baseline.py` and `guided_search.py` share: the problems, and the
    guide-replay budget and RSS cap every `attempt` process runs under."""

    # So a misspelled key in `experiment.py` fails instead of being dropped.
    model_config = ConfigDict(extra="forbid")

    # I/O
    path: Path = Field(
        description=(
            "Problem folder with `problems.json` and `problem_args.json` "
            "(both written by `generate_problems.py`)."
        )
    )

    attempt_bin: Path = Field(
        default=Path("target/release/attempt"), description="Path to the attempt binary."
    )

    # Guide-replay budget
    #
    # At least one must be given. Replay ends when the first configured budget
    # is exhausted; omitted budgets are effectively unlimited.
    max_iters: int | None = Field(
        default=None, gt=0, description=("Guide-replay iteration budget.")
    )

    max_nodes: int | None = Field(
        default=None, gt=0, description=("Guide-replay egraph-node budget.")
    )

    max_time: float | None = Field(
        default=None, gt=0, description=("Guide-replay wall-clock budget in seconds.")
    )

    max_rss: str = Field(
        default="4G",
        description=(
            "Cap each `attempt` process (and, in `guided_search.py`, each `sample` "
            "process) at this cgroup RSS limit, as a human size such as `4G`."
        ),
    )

    n_starts: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only process the first N start terms in sorted order. All start terms are processed if omitted."
        ),
    )

    n_goals: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Only use the first N goals per start term in file order. All goals are used if omitted."
        ),
    )

    @property
    def max_rss_bytes(self) -> int:
        return parse_size(self.max_rss)

    @property
    def limits(self) -> dict[str, int | float]:
        """The configured guide-replay budgets, without the omitted ones."""
        budgets = {
            "max_iters": self.max_iters,
            "max_nodes": self.max_nodes,
            "max_time": self.max_time,
        }
        return {key: value for key, value in budgets.items() if value is not None}

    def base_flags(self, language: str) -> list[str]:
        """Flags shared by every `attempt` process."""
        return cli_flags(language=language, **self.limits)

    def baseline_key(self) -> dict:
        """Everything a baseline row depends on besides its pair.

        Two runs with the same key can share a baseline.
        """
        return {
            "problems": str(self.path.resolve()),
            "max_iters": self.max_iters,
            "max_nodes": self.max_nodes,
            "max_time": self.max_time,
            "max_rss": self.max_rss_bytes,
        }
