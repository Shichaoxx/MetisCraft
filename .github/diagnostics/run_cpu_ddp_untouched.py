"""Compare the original CPU DDP pytest fixture without worker-stage instrumentation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

VARIANTS = ("legacy", "current")
FIXTURE = "test_cpu_ddp_tail_uses_real_global_question_count"
THREAD_LIMIT = b"    torch.set_num_threads(1)\n"
VISIBLE_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONPATH",
    "PYTHONFAULTHANDLER",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "TORCH_SHOW_CPP_STACKTRACES",
    "METISCRAFT_TERMINATE_TRACE",
)
SITE_CUSTOMIZE = '''"""Quiet startup hooks; only fatal errors produce trace output."""
import ctypes
import faulthandler
import os

faulthandler.enable(all_threads=True)
_metiscraft_trace = ctypes.CDLL(os.environ["METISCRAFT_TERMINATE_TRACE"])
_metiscraft_trace.metiscraft_install_terminate_trace()
os.environ["METISCRAFT_TERMINATE_TRACE_PID"] = str(os.getpid())
'''


def sha256(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def save_json(filename, value):
    filename.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def child_limits(cpus):
    os.sched_setaffinity(0, cpus)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def stop_own_group(proc):
    sent = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            sent.append(sig.name)
        except ProcessLookupError:
            pass
        try:
            return proc.wait(timeout=5), sent
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError(f"owned diagnostic group {proc.pid} did not stop")


def run_process(command, *, cwd, env, log, cpus, timeout):
    started = time.monotonic()
    timed_out, sent = False, []
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            preexec_fn=lambda: child_limits(cpus),
        )
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            code, sent = stop_own_group(proc)
        except BaseException:
            stop_own_group(proc)
            raise
    return {
        "pid": proc.pid,
        "exit_code": code,
        "parent_signal": signal.Signals(-code).name if code < 0 else None,
        "timeout": timed_out,
        "cleanup_signals": sent,
        "seconds": round(time.monotonic() - started, 3),
    }


def git_read(workspace, *arguments):
    proc = subprocess.run(
        ["git", *arguments], cwd=workspace, capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def assertion_fingerprint(source):
    assertions = [
        ast.dump(node, include_attributes=False)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assert)
    ]
    return {
        "count": len(assertions),
        "sha256": hashlib.sha256("\n".join(assertions).encode()).hexdigest(),
    }


def trial_environment(variant, inject, trace, workspace):
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "METISCRAFT_TERMINATE_TRACE_PID"):
        env.pop(name, None)
    if variant == "current":
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    previous_path = env.get("PYTHONPATH")
    env.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONFAULTHANDLER="1",
        TORCH_SHOW_CPP_STACKTRACES="1",
        METISCRAFT_TERMINATE_TRACE=str(trace),
        PYTHONPATH=os.pathsep.join([str(inject), str(workspace / "src")])
        + (os.pathsep + previous_path if previous_path else ""),
    )
    return env


def junit_counts(filename):
    if not filename.is_file():
        return None
    try:
        root = ET.parse(filename).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        return {
            key: sum(int(suite.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    except (ET.ParseError, ValueError):
        return None


def rank_outputs(case):
    results = {}
    for filename in sorted((case / "pytest-tmp").rglob("rank-*.json")):
        try:
            results[str(filename.relative_to(case))] = json.loads(filename.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            results[str(filename.relative_to(case))] = {"read_error": str(exc)}
    return results


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=here.parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.timeout <= 90:
        parser.error("repeats must be positive; timeout must be between 1 and 90 seconds")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must not repeat")
    if not hasattr(os, "sched_getaffinity"):
        parser.error("the CPU Gloo diagnostic requires Linux CPU affinity support")
    workspace, output = args.workspace.resolve(), args.output.resolve()
    fixture = workspace / "tests/test_runtime_contracts.py"
    support = [workspace / "tests" / name for name in ("__init__.py", "conftest.py")]
    if not all(filename.is_file() for filename in [fixture, *support]):
        parser.error("workspace must contain the original fixture and tests package helpers")
    current = fixture.read_bytes()
    if current.count(THREAD_LIMIT) != 1:
        parser.error("expected exactly one original rank-local torch.set_num_threads(1) line")
    legacy = current.replace(THREAD_LIMIT, b"", 1)
    assertions = assertion_fingerprint(current)
    if assertion_fingerprint(legacy) != assertions:
        parser.error("legacy transformation unexpectedly changes original assertions")
    if output.exists():
        parser.error("output already exists; use a new directory to preserve all evidence")
    output.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[:4]
    dependencies = sorted(
        {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}.items()
    )
    module = importlib.util.find_spec("metiscraft")
    report = {
        "status": "running",
        "workspace": str(workspace),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "commit": git_read(workspace, "rev-parse", "HEAD"),
        "working_tree": git_read(workspace, "status", "--porcelain"),
        "parent_metiscraft_origin": module.origin if module else None,
        "trial_source_root": str(workspace / "src"),
        "train_loop_sha256": sha256(workspace / "src/metiscraft/train/loop.py"),
        "dependencies": dependencies,
        "parent_environment": {name: os.environ.get(name) for name in VISIBLE_ENV},
        "cpu_affinity": cpus,
        "repeats": args.repeats,
        "variants": args.variants,
        "timeout_seconds": args.timeout,
        "fixture_sha256": {"source": sha256(fixture)},
        "assertions": assertions,
        "legacy_change": "Remove only the unique torch.set_num_threads(1) source line",
        "trials": [],
        "interpretation": (
            "No test retries and no worker-stage instrumentation. Legacy failures are retained "
            "as evidence; any current failure makes this command fail. Passing finite trials "
            "does not prove the absence of a native shutdown race. Trace startup is quiet and "
            "does not record per-process installation events; spawn starts Python normally, "
            "so the same sitecustomize applies to pytest and spawned workers."
        ),
    }
    summary = output / "matrix.json"
    save_json(summary, report)
    try:
        for variant, source in (("legacy", legacy), ("current", current)):
            copied = output / "workspaces" / variant / "tests"
            copied.mkdir(parents=True)
            (copied / fixture.name).write_bytes(source)
            for filename in support:
                shutil.copyfile(filename, copied / filename.name)
            report["fixture_sha256"][variant] = sha256(copied / fixture.name)
        report["support_sha256"] = {filename.name: sha256(filename) for filename in support}
        inject = output / "trace-startup"
        inject.mkdir()
        startup = inject / "sitecustomize.py"
        startup.write_text(SITE_CUSTOMIZE, encoding="utf-8")
        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError("g++ is required for the native terminate trace")
        trace = output / "terminate_trace.so"
        command = [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-g",
            "-O0",
            str(here / "terminate_trace.cpp"),
            "-o",
            str(trace),
        ]
        report["compile"] = {
            "command": command,
            **run_process(
                command,
                cwd=workspace,
                env=dict(os.environ),
                log=output / "compile.log",
                cpus=cpus,
                timeout=60,
            ),
        }
        if report["compile"]["exit_code"] != 0 or report["compile"]["timeout"]:
            raise RuntimeError("native terminate trace compilation failed")
        report["trace"] = {
            "source_sha256": sha256(here / "terminate_trace.cpp"),
            "library_sha256": sha256(trace),
            "sitecustomize_sha256": sha256(startup),
            "normal_startup_output": False,
            "normal_startup_file_io": False,
            "crash_output": "faulthandler and METISCRAFT_NATIVE_TERMINATE_BACKTRACE on stderr",
            "scope": "pytest parent and Python spawn children; no Torch import in sitecustomize",
        }
        save_json(summary, report)
        for repeat in range(1, args.repeats + 1):
            for variant in args.variants:
                case = output / f"{repeat:02d}-{variant}"
                case.mkdir()
                cwd = output / "workspaces" / variant
                env = trial_environment(variant, inject, trace, workspace)
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"tests/{fixture.name}::{FIXTURE}",
                    "--basetemp",
                    str(case / "pytest-tmp"),
                    "--junitxml",
                    str(case / "junit.xml"),
                    "-o",
                    "cache_dir=" + str(case / "pytest-cache"),
                ]
                save_json(
                    case / "command.json",
                    {
                        "command": command,
                        "cwd": str(cwd),
                        "environment": {name: env.get(name) for name in VISIBLE_ENV},
                    },
                )
                result = run_process(
                    command,
                    cwd=cwd,
                    env=env,
                    log=case / "process.log",
                    cpus=cpus,
                    timeout=args.timeout,
                )
                log = (case / "process.log").read_text(errors="replace")
                counts = junit_counts(case / "junit.xml")
                result.update(
                    case=case.name,
                    variant=variant,
                    repeat=repeat,
                    worker_signals=re.findall(
                        r"process (\d+) terminated with signal (SIG\w+)", log
                    ),
                    native_terminate="METISCRAFT_NATIVE_TERMINATE_BACKTRACE" in log,
                    junit=counts,
                    rank_outputs=rank_outputs(case),
                )
                result["passed"] = (
                    result["exit_code"] == 0
                    and not result["timeout"]
                    and counts == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
                    and "Error in sitecustomize" not in log
                )
                save_json(case / "result.json", result)
                report["trials"].append(result)
                save_json(summary, report)
                print(
                    json.dumps({k: v for k, v in result.items() if k != "rank_outputs"}), flush=True
                )
        failed = [trial["case"] for trial in report["trials"] if not trial["passed"]]
        gating = [
            trial["case"]
            for trial in report["trials"]
            if trial["variant"] == "current" and not trial["passed"]
        ]
        report.update(
            status="failed" if gating else "complete",
            failed_cases=failed,
            current_failures=gating,
            all_trials_passed=not failed,
        )
        save_json(summary, report)
        print(
            f"Evidence: {output}; {len(report['trials'])} trials, {len(failed)} failures",
            flush=True,
        )
        return int(bool(gating))
    except BaseException as exc:
        report.update(
            status="setup_failed" if not report["trials"] else "interrupted", error=str(exc)
        )
        save_json(summary, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
