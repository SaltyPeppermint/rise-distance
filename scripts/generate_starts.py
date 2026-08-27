"""Generate validated start terms in isolated Rust processes.

The Rust ``generate plan`` command supplies the uniform size-allocation plan and
``generate one`` produces one term per invocation. This coordinator provides
deterministic per-slot seeds,
bounded process concurrency, deduplication, telemetry, and resumable JSONL
checkpoints. Workers are not externally memory-contained. Rust samples
``--max-memory`` immediately before hooks, after every individual rule search
and application, and after rebuild/finalization. The trace retains each
iteration's peak and responsible rule even when transient allocations are
freed before iteration end. Reported allocation figures are absolute process
live heap (jemalloc ``stats.allocated``), directly comparable to
``--max-memory``.
Per-slot BLAKE2 seeds intentionally replace (and differ from) the old
monolithic ChaCha stream. Term assignment is stable across worker counts;
measured wall times and allocator statistics in payloads remain observational.

Example:
    cargo build --release --bin generate
    uv run scripts/generate_starts.py --total-samples 10 --min-size 10 \
      --max-size 12 --language math --seed 42
"""

import dataclasses
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
from diceware.wordlist import WordList, get_wordlists_dir
from tqdm import tqdm

from common import exit_if_missing, parse_size, subprocess_timeout

COORDINATOR_STATE_VERSION = 3


def _load_wordlist(name: str) -> list[str]:
    path = os.path.join(get_wordlists_dir(), f"wordlist_{name}.txt")
    return list(WordList(path))


def generate_unique_dir(parent: Path, max_attempts: int = 100) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    adjectives = _load_wordlist("en_adjectives")
    nouns = _load_wordlist("en_nouns")
    for _ in range(max_attempts):
        candidate = parent / f"{secrets.choice(adjectives)}-{secrets.choice(nouns)}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError(f"Could not find an unused name under {parent}")


@dataclass
class Args:
    path: Path | None = None
    """Output directory. A fresh data/start_terms directory is used if omitted."""

    generate_binary: Path = Path("target/release/generate")
    total_samples: int = tyro.MISSING
    min_size: int = tyro.MISSING
    max_size: int = tyro.MISSING
    language: str = tyro.MISSING
    seed: int = tyro.MISSING
    retry_limit: int = 10000

    max_iters: int = 11
    max_nodes: int = 100_000
    max_time: float = 1.0
    max_memory: str | None = None
    """Absolute live-heap ceiling checked at egg's sampled rule boundaries."""
    predict_next_memory: Path | None = None

    workers: int = 4
    worker_timeout: float | None = None
    """External timeout per attempt; defaults to max_time*4+5 seconds."""
    max_job_attempts: int = 10000
    poll_interval: float = 0.1


def semantic_config(args: Args) -> dict[str, Any]:
    """Arguments that affect the requested terms and their validation."""
    return {
        "generate_binary": str(args.generate_binary),
        "total_samples": args.total_samples,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "language": args.language,
        "seed": args.seed,
        "retry_limit": args.retry_limit,
        "max_iters": args.max_iters,
        "max_nodes": args.max_nodes,
        "max_time": args.max_time,
        "max_memory": args.max_memory,
    }


def _eqsat_flags(args: Args) -> list[str]:
    flags = [
        "--max-iters",
        str(args.max_iters),
        "--max-nodes",
        str(args.max_nodes),
        "--max-time",
        str(args.max_time),
    ]
    if args.max_memory is not None:
        flags += ["--max-memory", str(parse_size(args.max_memory))]
    if args.predict_next_memory is not None:
        flags += ["--predict-next-memory", str(args.predict_next_memory)]
    return flags


def base_generate_cmd(args: Args) -> list[str]:
    return [
        str(args.generate_binary),
        "one",
        "--language",
        args.language,
        "--retry-limit",
        str(args.retry_limit),
        *_eqsat_flags(args),
    ]


def request_plan(args: Args) -> list[tuple[int, int]]:
    cmd = [
        str(args.generate_binary),
        "plan",
        "--total-samples",
        str(args.total_samples),
        "--min-size",
        str(args.min_size),
        "--max-size",
        str(args.max_size),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"generate plan failed ({proc.returncode}):\n{proc.stderr}")
    plan = [(int(size), int(count)) for size, count in json.loads(proc.stdout)]
    if sum(count for _, count in plan) != args.total_samples:
        raise RuntimeError(f"Rust returned invalid allocation plan: {plan}")
    return plan


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def atomic_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=indent)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _poll_with_rusage(running: Running) -> int | None:
    """Poll a direct child and retain Linux wait4's per-process peak RSS."""
    try:
        waited, status, usage = os.wait4(running.process.pid, os.WNOHANG)
    except ChildProcessError:
        return running.process.poll()
    if waited == 0:
        running.peak_rss = max(running.peak_rss, _rss_bytes(running.process.pid))
        return None
    running.peak_rss = max(running.peak_rss, int(usage.ru_maxrss) * 1024)
    running.process.returncode = os.waitstatus_to_exitcode(status)
    return running.process.returncode


def _stop_reason(payload: Any):
    try:
        return payload[1]["stop_reason"]
    except IndexError, KeyError, TypeError:
        return None


class Coordinator:
    def __init__(self, args: Args, plan: list[tuple[int, int]]):
        assert args.path is not None
        self.args = args
        self.path = args.path
        self.plan = plan
        self.workers_dir = self.path / "workers"
        self.workers_dir.mkdir(exist_ok=True)
        self.attempts_path = self.path / "attempts.jsonl"
        self.checkpoint_path = self.path / "checkpoint.jsonl"
        self.accepted: dict[int, dict[int, tuple[str, Any]]] = defaultdict(dict)
        self.seen: dict[int, set[str]] = defaultdict(set)
        self.pending: deque[Job] = deque()
        self.running: dict[int, Running] = {}
        self.buffered: dict[int, dict[int, tuple[Job, dict[str, Any], str, Any]]] = defaultdict(
            dict
        )
        self.latest_attempt: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.next_attempt: dict[tuple[int, int], int] = defaultdict(int)
        self.deadline = args.worker_timeout or subprocess_timeout(args.max_time)

    def load(self) -> None:
        repair_jsonl_tail(self.checkpoint_path)
        repair_jsonl_tail(self.attempts_path)
        wanted = {size: count for size, count in self.plan}
        for record in read_jsonl(self.checkpoint_path):
            size, slot = int(record["size"]), int(record["slot"])
            if size not in wanted or not 0 <= slot < wanted[size]:
                raise RuntimeError(f"Checkpoint contains out-of-plan slot: {record}")
            term = str(record["term"])
            if slot in self.accepted[size] or term in self.seen[size]:
                raise RuntimeError(f"Checkpoint contains duplicate slot or term: {record}")
            self.accepted[size][slot] = (term, record["payload"])
            self.seen[size].add(term)

        for record in read_jsonl(self.attempts_path):
            key = (int(record["size"]), int(record["slot"]), int(record["attempt"]))
            self.latest_attempt[key] = record
            pair = key[:2]
            self.next_attempt[pair] = max(self.next_attempt[pair], key[2] + 1)

        # Recover completed-but-not-yet-resolved candidates from worker shards.
        for key, record in sorted(self.latest_attempt.items()):
            size, slot, attempt = key
            if slot in self.accepted[size] or record.get("status") != "candidate":
                continue
            worker_file = self.path / record["worker_file"]
            try:
                term, payload = parse_worker_output(worker_file, size)
            except OSError, TypeError, ValueError, json.JSONDecodeError:
                continue
            self.buffered[size][slot] = (Job(size, slot, attempt), record, term, payload)

        for size, count in self.plan:
            for slot in range(count):
                if slot not in self.accepted[size] and slot not in self.buffered[size]:
                    self.pending.append(Job(size, slot, self.next_attempt[(size, slot)]))

    def command(self, job: Job, seed: int) -> list[str]:
        return [
            *base_generate_cmd(self.args),
            "--size",
            str(job.size),
            "--seed",
            str(seed),
        ]

    def launch(self, job: Job) -> None:
        if job.attempt >= self.args.max_job_attempts:
            raise RuntimeError(f"Job {job.size}/{job.slot} exhausted max_job_attempts")
        seed = derive_seed(self.args.seed, job.size, job.slot, job.attempt)
        output = self.workers_dir / (
            f"size-{job.size:03d}-slot-{job.slot:05d}-attempt-{job.attempt:05d}.json"
        )
        with output.open("w") as worker_stdout:
            proc = subprocess.Popen(
                self.command(job, seed),
                stdout=worker_stdout,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        self.running[proc.pid] = Running(job, seed, proc, output, time.monotonic())

    @staticmethod
    def terminate(running: Running) -> None:
        try:
            os.killpg(running.process.pid, signal.SIGTERM)
            running.process.wait(timeout=2)
        except ProcessLookupError, subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(running.process.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                running.process.wait(timeout=2)

    def record_failure(
        self,
        running: Running,
        status: str,
        exit_code: int | None,
        stderr: str,
        attempts_handle: Any,
        *,
        retry: bool = True,
    ) -> None:
        record = {
            "size": running.job.size,
            "slot": running.job.slot,
            "attempt": running.job.attempt,
            "seed": running.seed,
            "status": status,
            "runtime_seconds": time.monotonic() - running.started,
            "peak_rss_bytes": running.peak_rss,
            "exit_code": exit_code,
            "stop_reason": None,
            "stderr": stderr[-4000:],
        }
        append_jsonl(attempts_handle, record)
        self.latest_attempt[(running.job.size, running.job.slot, running.job.attempt)] = record
        if retry:
            self.pending.appendleft(
                Job(running.job.size, running.job.slot, running.job.attempt + 1)
            )

    def collect_finished(self, attempts_handle: Any) -> None:
        now = time.monotonic()
        for pid, running in list(self.running.items()):
            timed_out = now - running.started > self.deadline
            code = _poll_with_rusage(running)
            if code is None and not timed_out:
                continue
            if timed_out:
                self.terminate(running)
                code = running.process.returncode
            stderr_pipe = running.process.stderr
            stderr = stderr_pipe.read() if stderr_pipe else ""
            if stderr_pipe:
                stderr_pipe.close()
            del self.running[pid]
            if timed_out:
                self.record_failure(running, "timeout", code, stderr, attempts_handle)
                continue
            if code is not None and code < 0:
                self.record_failure(
                    running,
                    "signal_failure",
                    code,
                    stderr,
                    attempts_handle,
                    retry=False,
                )
                raise RuntimeError(
                    f"worker {running.job.size}/{running.job.slot}/{running.job.attempt} "
                    f"was killed by signal {-code}; aborting all workers"
                )
            if code != 0:
                self.record_failure(
                    running,
                    "process_failure",
                    code,
                    stderr,
                    attempts_handle,
                )
                continue
            try:
                term, payload = parse_worker_output(running.output_path, running.job.size)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.record_failure(
                    running,
                    "process_failure",
                    code,
                    f"{stderr}\nInvalid worker output: {error}",
                    attempts_handle,
                )
                continue
            record = {
                "size": running.job.size,
                "slot": running.job.slot,
                "attempt": running.job.attempt,
                "seed": running.seed,
                "status": "candidate",
                "runtime_seconds": now - running.started,
                "peak_rss_bytes": running.peak_rss,
                "exit_code": code,
                "stop_reason": _stop_reason(payload),
                "worker_file": str(running.output_path.relative_to(self.path)),
                "rust_draw_attempts": payload[0] if isinstance(payload, list) else None,
            }
            append_jsonl(attempts_handle, record)
            self.latest_attempt[(running.job.size, running.job.slot, running.job.attempt)] = record
            self.buffered[running.job.size][running.job.slot] = (
                running.job,
                record,
                term,
                payload,
            )

    def resolve(self, attempts_handle: Any, checkpoint_handle: Any) -> int:
        accepted = 0
        for size, count in self.plan:
            slot = 0
            while slot < count and slot in self.accepted[size]:
                slot += 1
            while slot < count and slot in self.buffered[size]:
                job, base, term, payload = self.buffered[size].pop(slot)
                if term in self.seen[size]:
                    resolved = {**base, "status": "duplicate", "term": term}
                    append_jsonl(attempts_handle, resolved)
                    self.latest_attempt[(size, slot, job.attempt)] = resolved
                    self.pending.appendleft(Job(size, slot, job.attempt + 1))
                    break
                checkpoint = {
                    "size": size,
                    "slot": slot,
                    "attempt": job.attempt,
                    "seed": base["seed"],
                    "term": term,
                    "payload": payload,
                }
                append_jsonl(checkpoint_handle, checkpoint)
                self.accepted[size][slot] = (term, payload)
                self.seen[size].add(term)
                resolved = {**base, "status": "accepted", "term": term}
                append_jsonl(attempts_handle, resolved)
                self.latest_attempt[(size, slot, job.attempt)] = resolved
                slot += 1
                accepted += 1
        return accepted

    def complete(self) -> bool:
        return all(len(self.accepted[size]) == count for size, count in self.plan)

    def run(self) -> list[list[Any]]:
        self.load()
        started = time.monotonic()
        self.attempts_path.touch(exist_ok=True)
        self.checkpoint_path.touch(exist_ok=True)
        with (
            tqdm(
                total=self.args.total_samples,
                initial=sum(map(len, self.accepted.values())),
                desc="Generating",
                unit="term",
                disable=None,
            ) as progress,
            self.attempts_path.open("a") as attempts,
            self.checkpoint_path.open("a") as checkpoints,
        ):
            try:
                progress.update(self.resolve(attempts, checkpoints))
                while not self.complete():
                    while self.pending and len(self.running) < self.args.workers:
                        job = self.pending.popleft()
                        # A recovered/running attempt for this slot takes precedence.
                        if (
                            job.slot in self.accepted[job.size]
                            or job.slot in self.buffered[job.size]
                        ):
                            continue
                        self.launch(job)
                    if not self.running:
                        progress.update(self.resolve(attempts, checkpoints))
                        if not self.pending and not self.complete():
                            raise RuntimeError("Coordinator stalled with unfinished jobs")
                        continue
                    time.sleep(self.args.poll_interval)
                    self.collect_finished(attempts)
                    progress.update(self.resolve(attempts, checkpoints))
            except BaseException:
                for running in list(self.running.values()):
                    self.terminate(running)
                    if running.process.stderr:
                        running.process.stderr.close()
                    record = {
                        "size": running.job.size,
                        "slot": running.job.slot,
                        "attempt": running.job.attempt,
                        "seed": running.seed,
                        "status": "interruption",
                        "runtime_seconds": time.monotonic() - running.started,
                        "peak_rss_bytes": running.peak_rss,
                        "exit_code": running.process.returncode,
                        "stop_reason": None,
                    }
                    append_jsonl(attempts, record)
                self.running.clear()
                raise

        output = []
        for size, count in sorted(self.plan):
            terms = {
                self.accepted[size][slot][0]: self.accepted[size][slot][1] for slot in range(count)
            }
            output.append([size, terms])
        atomic_json(self.path / "terms.json", output)
        self.print_summary(time.monotonic() - started)
        return output

    def print_summary(self, runtime: float) -> None:
        final = list(self.latest_attempt.values())
        peaks = sorted(int(row.get("peak_rss_bytes") or 0) for row in final)
        runtimes = sorted(float(row.get("runtime_seconds") or 0) for row in final)

        def percentile(values: list[float] | list[int], p: float):
            if not values:
                return 0.0
            index = round((len(values) - 1) * p)
            return float(values[index])

        retries = sum(1 for row in final if int(row["attempt"]) > 0)
        print(
            f"Completed {self.args.total_samples} terms in {runtime:.2f}s; "
            f"attempts={len(final)}, retries={retries}; "
            f"runtime p50/p95={percentile(runtimes, 0.5):.2f}/{percentile(runtimes, 0.95):.2f}s; "
            f"peak RSS p50/p95/max={percentile(peaks, 0.5) / 2**20:.1f}/"
            f"{percentile(peaks, 0.95) / 2**20:.1f}/{percentile(peaks, 1) / 2**20:.1f} MiB",
            file=sys.stderr,
        )


def parse_worker_output(path: Path, expected_size: int) -> tuple[str, Any]:
    del expected_size
    output = json.loads(path.read_text())
    if not isinstance(output, dict) or set(output) != {"term", "payload"}:
        raise ValueError("worker did not return one {term, payload} object")
    if not isinstance(output["term"], str) or not isinstance(output["payload"], list):
        raise TypeError("worker returned invalid term or payload")
    return output["term"], output["payload"]


def validate_args(args: Args) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.total_samples < 0 or args.min_size > args.max_size:
        raise ValueError("invalid sample count or size range")


def prepare_generation_args(args: Args) -> None:
    assert args.path is not None
    path = args.path / "generation_args.json"
    wanted = semantic_config(args)
    if path.exists():
        existing = json.loads(path.read_text())
        compatibility = dict(existing.get("compatibility", {}))
        legacy_backoff = compatibility.pop("backoff_scheduler", True)
        if legacy_backoff is not True or compatibility != wanted:
            raise RuntimeError(
                f"{path} is incompatible with this run; choose a new --path or restore "
                "the original generation arguments"
            )
    else:
        atomic_json(path, generation_record(args), indent=2)


def main() -> int:
    args = tyro.cli(Args, description=__doc__)
    try:
        validate_args(args)
        if args.path is None:
            args.path = generate_unique_dir(Path("data/start_terms"))
            print(f"Auto-generated output dir: {args.path}", file=sys.stderr)
        else:
            args.path.mkdir(parents=True, exist_ok=True)
        exit_if_missing(args.generate_binary)
        plan = request_plan(args)
        prepare_generation_args(args)
        print(
            f"Generating {args.total_samples} terms with {args.workers} workers -> "
            f"{args.path / 'terms.json'}",
            file=sys.stderr,
        )
        Coordinator(args, plan).run()
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed checkpoints are safe to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"generated start terms: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
