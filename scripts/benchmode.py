#!/usr/bin/env python3

"""Run a command under isolated Linux benchmode (no SMT/turbo, locked frequency,
housekeeping slices confined to CPU 0, workload in delegated `bench.slice`)."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

CPU_DIR = Path("/sys/devices/system/cpu")
CGROUP_DIR = Path("/sys/fs/cgroup")
PSTATE_STATUS_PATHS = (CPU_DIR / "amd_pstate" / "status", CPU_DIR / "intel_pstate" / "status")
INTEL_NO_TURBO = CPU_DIR / "intel_pstate" / "no_turbo"
CPUFREQ_BOOST = CPU_DIR / "cpufreq" / "boost"
SMT_CONTROL = CPU_DIR / "smt" / "control"
SLICES = ("system.slice", "user.slice", "init.scope")
HOUSEKEEPING_CPU = 0
FREQ_MHZ: int | None = None  # None = auto-detect from cpufreq base_frequency or ACPI nominal_freq


def benchmode_cpus() -> list[int]:
    """Parse `BENCHMODE_CPUS` (comma-separated CPU IDs) inside benchmode."""
    cpus = [int(part) for part in os.environ.get("BENCHMODE_CPUS", "").split(",") if part]
    if not cpus:
        raise RuntimeError("expected to run inside benchmode (`BENCHMODE_CPUS` is not set)")
    return cpus


def benchmode_cgroup() -> Path:
    """Return the delegated benchmode cgroup root."""
    raw = os.environ.get("BENCHMODE_CGROUP", "")
    path = Path(raw)
    if not raw or not path.is_dir():
        raise RuntimeError("expected to run inside benchmode (`BENCHMODE_CGROUP` is not set)")
    return path


def _which(name: str, fallback: str) -> str:
    return shutil.which(name) or fallback


def _read_sysfs(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    if (text := _read_sysfs(path)) is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _write_sysfs(path: Path, value: str) -> None:
    try:
        path.write_text(f"{value}\n")
    except OSError as exc:
        print(f"benchmode: could not write '{value}' to {path}: {exc}", file=sys.stderr)


def _cpu_dirs() -> list[tuple[int, Path]]:
    return sorted((int(p.name[3:]), p) for p in CPU_DIR.glob("cpu[0-9]*") if p.name[3:].isdigit())


def _cpufreq_files(*, include_boost: bool = False) -> Iterator[Path]:
    for _, cpu_dir in _cpu_dirs():
        cpufreq = cpu_dir / "cpufreq"
        if not cpufreq.is_dir():
            continue
        # Restore order matters: governor before EPP, max before min.
        yield cpufreq / "scaling_governor"
        if (epp := cpufreq / "energy_performance_preference").exists():
            yield epp
        if include_boost and (boost := cpufreq / "boost").exists():
            yield boost
        yield cpufreq / "scaling_max_freq"
        yield cpufreq / "scaling_min_freq"


def _pstate_status_path() -> Path | None:
    return next((p for p in PSTATE_STATUS_PATHS if p.exists()), None)


def _detect_freq_khz() -> int:
    if FREQ_MHZ is not None:
        return FREQ_MHZ * 1000
    sources = (
        (CPU_DIR / "cpu0" / "cpufreq" / "base_frequency", 1),
        (CPU_DIR / "cpu0" / "acpi_cppc" / "nominal_freq", 1000),
    )
    for path, scale in sources:
        if (val := _read_int(path)) is not None and val > 0:
            return val * scale
    paths = ", ".join(str(p) for p, _ in sources)
    sys.exit(f"benchmode: could not auto-detect frequency from {paths}; set FREQ_MHZ in {__file__}")


def _select_bench_cpus(bench_cpus: Sequence[int], *, sequential: bool) -> list[int]:
    """Select one core (preferring highest max freq) if `sequential`, else all `bench_cpus`."""
    if not sequential:
        return sorted(bench_cpus)
    best = max(
        bench_cpus,
        key=lambda cpu_id: (
            _read_int(CPU_DIR / f"cpu{cpu_id}" / "cpufreq" / "cpuinfo_max_freq") or 0
        ),
    )
    return [best]


def _online_cpus() -> list[int]:
    return [cpu_id for cpu_id, cpu_dir in _cpu_dirs() if _read_sysfs(cpu_dir / "online") != "0"]


def _systemctl(*args: str) -> str:
    systemctl = _which("systemctl", "/run/current-system/sw/bin/systemctl")
    return subprocess.run(
        [systemctl, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@contextmanager
def isolated_cpus(*, sequential: bool = False) -> Generator[tuple[int, list[int]]]:
    freq_khz = _detect_freq_khz()
    freq_mhz = round(freq_khz / 1000)

    # Apply/restore order: driver mode first (resets cpufreq policies), then SMT/boost.
    global_writes: list[tuple[Path, str]] = []
    if (pstate := _pstate_status_path()) is not None and _read_sysfs(pstate) in (
        "active",
        "guided",
    ):
        global_writes.append((pstate, "passive"))
    if _read_sysfs(SMT_CONTROL) == "on":
        global_writes.append((SMT_CONTROL, "off"))
    if INTEL_NO_TURBO.exists():
        global_writes.append((INTEL_NO_TURBO, "1"))
    elif CPUFREQ_BOOST.exists():
        global_writes.append((CPUFREQ_BOOST, "0"))
    has_global_boost = INTEL_NO_TURBO.exists() or CPUFREQ_BOOST.exists()

    # Save everything before changing anything, while all CPUs are still online.
    to_save = [p for p, _ in global_writes] + list(
        _cpufreq_files(include_boost=not has_global_boost)
    )
    saved_sysfs = [(p, val) for p in to_save if (val := _read_sysfs(p)) is not None]
    saved_slices = [(s, _systemctl("show", "-P", "AllowedCPUs", s)) for s in SLICES]
    all_cpus = _read_sysfs(CPU_DIR / "possible")

    try:
        # In active/guided mode firmware picks the clock and may ignore boost/scaling_max_freq.
        for path, val in global_writes:
            _write_sysfs(path, val)

        online_cpus = _online_cpus()
        for cpu_id in online_cpus:
            p = CPU_DIR / f"cpu{cpu_id}" / "cpufreq"
            if not p.is_dir():
                continue
            if not has_global_boost and (cpu_boost := p / "boost").exists():
                _write_sysfs(cpu_boost, "0")
            _write_sysfs(p / "scaling_governor", "performance")
            if freq_khz < int((p / "scaling_min_freq").read_text().strip()):
                _write_sysfs(p / "scaling_min_freq", str(freq_khz))
            _write_sysfs(p / "scaling_max_freq", str(freq_khz))
            actual_max = int((p / "scaling_max_freq").read_text().strip())
            _write_sysfs(p / "scaling_min_freq", str(min(freq_khz, actual_max)))

        bench_cpus = [c for c in online_cpus if c != HOUSEKEEPING_CPU]
        if not bench_cpus:
            sys.exit(f"benchmode: no benchmark CPUs left after excluding CPU {HOUSEKEEPING_CPU}")

        selected = _select_bench_cpus(bench_cpus, sequential=sequential)
        for s in SLICES:
            _systemctl("set-property", "--runtime", s, f"AllowedCPUs={HOUSEKEEPING_CPU}")

        yield freq_mhz, selected
    finally:
        prev_handlers = {
            sig: signal.signal(sig, signal.SIG_IGN)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            # Restore sysfs first so SMT siblings are back online before widening slice cpusets.
            for path, val in saved_sysfs:
                _write_sysfs(path, val)
            for s, cpus in saved_slices:
                try:
                    if not cpus and all_cpus:
                        # Workaround for upstream systemd bug (systemd/systemd#43846):
                        # clearing `AllowedCPUs=` on a running unit does not unset its live
                        # `cpuset.cpus` (and writing "" directly fails with ENOSPC on populated
                        # cgroups), so widen to all CPUs before clearing the runtime property.
                        # https://github.com/systemd/systemd/issues/43846
                        _systemctl("set-property", "--runtime", s, f"AllowedCPUs={all_cpus}")
                    _systemctl("set-property", "--runtime", s, f"AllowedCPUs={cpus}")
                except subprocess.CalledProcessError as exc:
                    print(
                        f"benchmode: could not restore AllowedCPUs for {s}: {exc}",
                        file=sys.stderr,
                    )
            print("benchmode off", file=sys.stderr)
        finally:
            for sig, prev in prev_handlers.items():
                signal.signal(sig, prev)


def _prepare_delegated_cgroup() -> Path:
    """Move this process into a `driver/` leaf and enable memory/cpuset for worker cgroups.

    Cgroup v2's no-internal-processes rule requires the root to be empty before enabling
    subtree controllers.
    """
    lines = Path("/proc/self/cgroup").read_text().strip().splitlines()
    rel = next(line.split("::", 1)[1] for line in lines if "::" in line).lstrip("/")
    root = CGROUP_DIR / rel
    controllers = set((root / "cgroup.controllers").read_text().split())
    needed = [c for c in ("memory", "cpuset") if c in controllers]
    driver = root / "driver"
    driver.mkdir(exist_ok=True)
    (driver / "cgroup.procs").write_text(f"{os.getpid()}\n")
    if needed:
        (root / "cgroup.subtree_control").write_text(" ".join(f"+{c}" for c in needed) + "\n")
    return root


def _exec_with_env(env_file: Path, cpu_str: str, cmd: Sequence[str]) -> NoReturn:
    user_env: dict[str, str] = json.loads(env_file.read_text())
    user_env["BENCHMODE"] = "1"
    user_env["BENCHMODE_CPUS"] = cpu_str
    user_env["BENCHMODE_CGROUP"] = str(_prepare_delegated_cgroup())
    try:
        os.execvpe(cmd[0], list(cmd), user_env)
    except OSError as exc:
        sys.exit(f"benchmode: failed to exec {cmd[0]}: {exc}")


def _run_wrapped(env_file: Path, *, sequential: bool, cmd: Sequence[str]) -> int:
    if os.geteuid() != 0:
        sys.exit("error: internal benchmode wrapper requires root")
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        sys.exit("error: SUDO_UID/SUDO_GID not set")

    def _handle_stop(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGHUP, _handle_stop)

    with isolated_cpus(sequential=sequential) as (freq_mhz, selected_cpus):
        cpu_str = ",".join(str(c) for c in selected_cpus)
        print(
            f"benchmode on: {freq_mhz} MHz, housekeeping CPU {HOUSEKEEPING_CPU}, "
            f"bench CPUs {cpu_str} ({len(selected_cpus)} core(s))",
            file=sys.stderr,
        )

        io_flag = "--pty" if (sys.stdin.isatty() and sys.stdout.isatty()) else "--pipe"
        systemd_run = _which("systemd-run", "/run/current-system/sw/bin/systemd-run")
        run_cmd = [
            systemd_run,
            "--quiet",
            io_flag,
            "--wait",
            "--collect",
            "--same-dir",
            "--slice=bench.slice",
            f"--uid={uid}",
            f"--gid={gid}",
            "-p",
            f"AllowedCPUs={cpu_str}",
            "-p",
            "MemoryAccounting=yes",
            "-p",
            "Delegate=yes",
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            "--_exec",
            str(env_file),
            cpu_str,
            *cmd,
        ]
        # Prevent systemd-run --pty from emitting OSC 0 terminal title sequences.
        systemd_env = {**os.environ, "SYSTEMD_ADJUST_TERMINAL_TITLE": "0"}
        try:
            return subprocess.run(run_cmd, env=systemd_env, check=False).returncode
        except KeyboardInterrupt:
            return 130


def elevate(cmd: Sequence[str], *, sequential: bool = False) -> int:
    """Re-run `cmd` under benchmode isolation via a single `sudo` wrapper."""
    if os.geteuid() == 0:
        sys.exit(
            "error: do not invoke benchmode with sudo directly; "
            "it calls sudo itself so child processes stay user-owned."
        )

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="bench-env-", suffix=".json", delete=False
    ) as tmp:
        json.dump(dict(os.environ), tmp)
        env_file = Path(tmp.name)

    try:
        wrap_cmd = [
            _which("sudo", "/run/wrappers/bin/sudo"),
            sys.executable,
            str(Path(__file__).resolve()),
            "--_wrap",
            str(env_file),
            "sequential" if sequential else "parallel",
            *cmd,
        ]
        proc = subprocess.Popen(wrap_cmd)
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                # Let the root wrapper finish its finally cleanup before returning.
                continue
    finally:
        env_file.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--_exec"] and len(args) >= 4:
        _exec_with_env(Path(args[1]), args[2], args[3:])
    if args[:1] == ["--_wrap"] and len(args) >= 4:
        sys.exit(_run_wrapped(Path(args[1]), sequential=args[2] == "sequential", cmd=args[3:]))
    sys.exit("error: benchmode.py is invoked internally via elevate()")


if __name__ == "__main__":
    main()
