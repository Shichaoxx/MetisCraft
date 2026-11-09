"""Run bounded, non-retrying CPU DDP diagnostics and retain every trial's evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = ("legacy", "bounded", "restore_only", "release_ddp", "cleanup", "barrier")
REQUIRED_STABLE = {"bounded", "cleanup"}
DESCRIPTIONS = {
    "legacy": "Old fixture, no OMP/MKL or rank-local thread limits",
    "bounded": "Current fixture: child OMP/MKL=2 and rank-local torch threads=1",
    "restore_only": "Legacy plus restore loop.accumulate, retaining DDP/model locals",
    "release_ddp": "Legacy plus release only DDP and collect GC; observe/model remain",
    "cleanup": "Legacy plus restore monkeypatch, release DDP/model/observer, collect GC",
    "barrier": "Legacy plus successful-path barrier before destroying the group",
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def child_limits(cpus):
    os.sched_setaffinity(0, cpus)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def stop_own_group(proc):
    # start_new_session ensures this group belongs only to this diagnostic trial.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError(f"diagnostic process {proc.pid} did not terminate")


def read_phases(case_dir):
    result = {}
    for rank in range(2):
        filename = case_dir / f"rank-{rank}-phases.jsonl"
        rows = []
        if filename.exists():
            for line in filename.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"phase": "partial-record", "raw": line})
        result[str(rank)] = rows
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=here.parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.timeout <= 90:
        parser.error("repeats must be positive and timeout must be between 1 and 90 seconds")
    if not hasattr(os, "sched_getaffinity"):
        parser.error("this native Gloo diagnostic requires Linux CPU affinity support")
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if not (workspace / "tests/test_runtime_contracts.py").is_file():
        parser.error("workspace must contain the original CPU tail regression fixture")
    if output.exists():
        parser.error("output already exists; use a new directory to preserve prior evidence")
    output.mkdir(parents=True)
    cpus = sorted(os.sched_getaffinity(0))[:4]
    report = {
        "workspace": str(workspace),
        "python": sys.executable,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "cpu_affinity": cpus,
        "variants": {name: DESCRIPTIONS[name] for name in args.variants},
        "required_stable": sorted(REQUIRED_STABLE.intersection(args.variants)),
        "trials": [],
        "status": "running",
        "interpretation": (
            "No retries. Legacy/isolation failures remain evidence, not a stability pass. "
            "Any bounded/cleanup failure fails this command. Zero failures in finitely many "
            "trials do not prove the absence of an intermittent native shutdown race. "
            "Weak references describe Python wrappers, not C++ destruction or thread joins."
        ),
    }
    sources = [
        here / "cpu_ddp_probe.py",
        here / "run_cpu_ddp_probe.py",
        here / "terminate_trace.cpp",
        workspace / "tests/test_runtime_contracts.py",
        workspace / "src/metiscraft/train/loop.py",
    ]
    report["source_sha256"] = {
        str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True, check=False
    )
    report["commit"] = git.stdout.strip() if git.returncode == 0 else None
    git_diff = subprocess.run(
        ["git", "diff", "--", ".github/diagnostics", "tests/test_runtime_contracts.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "source.diff").write_text(git_diff.stdout, encoding="utf-8")
    save_json(output / "matrix.json", report)
    compiler = shutil.which("g++")
    if compiler is None:
        report.update(status="setup_failed", error="g++ is required for the native terminate trace")
        save_json(output / "matrix.json", report)
        return 2
    trace = output / "terminate_trace.so"
    compile_cmd = [compiler, "-std=c++17", "-shared", "-fPIC", "-g", "-O0"]
    compile_cmd += [str(here / "terminate_trace.cpp"), "-o", str(trace)]
    with (output / "compile.log").open("w", encoding="utf-8") as stream:
        compiled = subprocess.run(
            compile_cmd, stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=60
        )
    if compiled.returncode:
        report.update(status="setup_failed", error="native trace compilation failed")
        save_json(output / "matrix.json", report)
        return 2

    for repeat in range(1, args.repeats + 1):
        for variant in args.variants:
            case = f"{repeat:02d}-{variant}"
            case_dir = output / case
            case_dir.mkdir()
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONFAULTHANDLER="1",
                TORCH_SHOW_CPP_STACKTRACES="1",
                PYTHONPATH=os.pathsep.join([str(workspace / "src"), str(workspace)]),
            )
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                env.pop(name, None)
            if variant == "bounded":
                env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            command = [
                sys.executable,
                str(here / "cpu_ddp_probe.py"),
                "--workspace",
                str(workspace),
                "--output",
                str(case_dir),
                "--variant",
                variant,
                "--trace",
                str(trace),
            ]
            started = time.monotonic()
            timed_out = False
            save_json(case_dir / "command.json", command)
            with (case_dir / "process.log").open("w", encoding="utf-8") as stream:
                proc = subprocess.Popen(
                    command,
                    cwd=workspace,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    preexec_fn=lambda: child_limits(cpus),
                )
                try:
                    code = proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    code = stop_own_group(proc)
                except BaseException:
                    stop_own_group(proc)
                    raise
            log = (case_dir / "process.log").read_text(errors="replace")
            phases = read_phases(case_dir)
            result = {
                "case": case,
                "variant": variant,
                "repeat": repeat,
                "exit_code": code,
                "parent_signal": signal.Signals(-code).name if code < 0 else None,
                "worker_signals": re.findall(r"process (\d+) terminated with signal (SIG\w+)", log),
                "native_terminate": "METISCRAFT_NATIVE_TERMINATE_BACKTRACE" in log,
                "timeout": timed_out,
                "seconds": round(time.monotonic() - started, 2),
                "passed": code == 0 and not timed_out,
                "required_stable": variant in REQUIRED_STABLE,
                "phases": phases,
                "last_phases": {
                    rank: rows[-1]["phase"] if rows else None for rank, rows in phases.items()
                },
            }
            save_json(case_dir / "result.json", result)
            report["trials"].append(result)
            save_json(output / "matrix.json", report)
            print(
                json.dumps({key: value for key, value in result.items() if key != "phases"}),
                flush=True,
            )
    failed = [row["case"] for row in report["trials"] if not row["passed"]]
    gating = [
        row["case"] for row in report["trials"] if row["required_stable"] and not row["passed"]
    ]
    report.update(
        status="failed" if gating else "complete",
        all_trials_passed=not failed,
        failed_cases=failed,
        stability_failures=gating,
    )
    save_json(output / "matrix.json", report)
    print(f"Evidence: {output}; {len(report['trials'])} trials, {len(failed)} failures", flush=True)
    return int(bool(gating))


if __name__ == "__main__":
    raise SystemExit(main())
