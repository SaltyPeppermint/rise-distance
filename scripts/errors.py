"""Custom exceptions raised by the driver scripts."""

import re

# Machine-readable eqsat progress events (`EVENT_PREFIX` in src/eqsat.rs),
# emitted under `--print-success-iters`.
EQSAT_ITER_RE = re.compile(r"^@EQSAT iter=(\d+)$", re.MULTILINE)

# Rust's exit code for a panic
PANIC_RETURNCODE = 101

# Linux's `MAX_ARG_STRLEN`: the most bytes, including the trailing NUL, that a
# single argv entry may have, else `exec` fails with `E2BIG`.
MAX_ARG_STRLEN = 32 * 4096

# Where a Rust panic message header
# `thread 'main' (609420) panicked at src/langs/math/mod.rs:97:13:`.
PANIC_HEADER_RE = re.compile(r"^thread '.*' .*panicked at ", re.MULTILINE)


class ArgTooLong(ValueError):
    """An argv entry exceeds the kernel's per-argument limit."""

    def __init__(self, what: str, flag: str, size: int) -> None:
        super().__init__(
            f"{what}: argument of {flag} is {size} bytes, "
            f"over the kernel's per-argument limit of {MAX_ARG_STRLEN - 1}"
        )


class MemoryKilled(RuntimeError):
    """A capped child was SIGKILLed by its cgroup memory limit."""

    def __init__(self, what: str, stderr: str, wall_time: float) -> None:
        super().__init__(f"{what} was killed at its RSS cap")
        self.stderr = stderr
        self.productive_iters = [int(m) for m in EQSAT_ITER_RE.findall(stderr)]
        self.wall_time = wall_time


class TimeoutKilled(RuntimeError):
    """A child that outran its hard timeout and was killed by hand."""

    def __init__(self, what: str, timeout: float, wall_time: float) -> None:
        super().__init__(f"{what} was killed at its hard timeout of {timeout:g}s")
        self.what = what
        self.wall_time = wall_time


class BinaryPanicked(RuntimeError):
    """A child exited with a Rust panic that it did not catch itself."""

    def __init__(self, what: str, stderr: str, wall_time: float) -> None:
        super().__init__(f"{what} panicked (code {PANIC_RETURNCODE}):\n{stderr}")
        self.what = what
        self.stderr = stderr
        self.wall_time = wall_time

    @property
    def panic_message(self) -> str:
        """The panic's own lines from stderr, without the progress output before
        it or the `RUST_BACKTRACE` hint after it."""
        header = PANIC_HEADER_RE.search(self.stderr)
        tail = self.stderr[header.start() :] if header else self.stderr
        return "\n".join(
            line for line in tail.strip().splitlines() if not line.startswith("note: run with")
        )

    @property
    def warning_message(self) -> str:
        """Formatted warning line(s) for logging the panic."""
        message = "\n".join(f"  {line}" for line in self.panic_message.splitlines())
        return f"WARNING: {self.what} panicked (code {PANIC_RETURNCODE}):\n{message}"


class BaselineMismatch(RuntimeError):
    """A stored baseline that was computed under other flags or misses pairs."""
