"""Diagnostic subprocess for the CPU DDP tail fixture; never used by package runtime."""

from __future__ import annotations

import argparse
import atexit
import ctypes
import faulthandler
import gc
import json
import math
import os
import platform
import sys
import time
import weakref
from datetime import timedelta
from pathlib import Path

VARIANTS = ("legacy", "bounded", "restore_only", "release_ddp", "cleanup", "barrier")
_TRACE_LIBRARY = None


def install_trace(filename):
    global _TRACE_LIBRARY
    faulthandler.enable(all_threads=True)
    if filename:
        _TRACE_LIBRARY = ctypes.CDLL(str(Path(filename).resolve()))
        _TRACE_LIBRARY.metiscraft_install_terminate_trace()


def native_threads():
    """Thread names are observations, not proof that a C++ destructor ran."""
    result = []
    directory = Path("/proc/self/task")
    if not directory.is_dir():
        return result
    for item in sorted(directory.iterdir(), key=lambda entry: int(entry.name)):
        try:
            result.append({"tid": int(item.name), "name": (item / "comm").read_text().strip()})
        except (FileNotFoundError, ProcessLookupError):
            pass  # A native thread can exit between listing and reading its name.
    return result


def reference_states(references):
    # Do not retain the result of ref(): that would change the lifetime under investigation.
    return {name: None if ref is None else ref() is not None for name, ref in references.items()}


def mark(output, rank, phase, references=None, **extra):
    threads = native_threads()
    row = {
        "phase": phase,
        "rank": rank,
        "pid": os.getpid(),
        "time_ns": time.monotonic_ns(),
        "native_threads": threads,
        "native_thread_count": len(threads),
        "python_wrappers_alive": reference_states(references or {}),
        **extra,
    }
    with (Path(output) / f"rank-{rank}-phases.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()


def watch(references, name, value, output, rank):
    def released(_reference):
        # This closure captures strings/rank only, never the object being watched.
        mark(output, rank, "python-wrapper-released", object=name)

    try:
        references[name] = weakref.ref(value, released)
    except TypeError:
        references[name] = None


def worker(rank, output, workspace, variant, trace):
    install_trace(trace)
    references = {}
    atexit.register(mark, output, rank, "python-atexit", references)
    mark(output, rank, "worker-enter", references)
    sys.path[:0] = [str(Path(workspace) / "src"), workspace]

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from metiscraft.config import ObjectiveConfig, TrainConfig
    from metiscraft.train import loop
    from tests.test_runtime_contracts import Scalar, collate

    mark(output, rank, "imports-complete", references, torch=torch.__version__)
    if variant == "bounded":
        torch.set_num_threads(1)
    mark(
        output,
        rank,
        "init-enter",
        references,
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
    )
    dist.init_process_group(
        "gloo",
        init_method=(Path(output) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    mark(output, rank, "init-exit", references)
    original = loop.accumulate
    model = net = observe = None
    completed = False
    try:
        model = Scalar()
        net = DistributedDataParallel(model)
        watch(references, "model", model, output, rank)
        watch(references, "ddp", net, output, rank)
        watch(references, "reducer", net.reducer, output, rank)
        watch(references, "process_group", net.process_group, output, rank)
        mark(
            output,
            rank,
            "ddp-created",
            references,
            use_python_reducer=getattr(net, "_use_python_reducer", None),
        )
        seen, grads = [], []

        def observe(net, micro, objective, **kwargs):
            seen.extend(i for part in micro for i in part["ids"])
            result = original(net, micro, objective, **kwargs)
            grads.append(float(model.weight.grad))
            return result

        loop.accumulate = observe
        summary = loop.train_loop(
            model,
            [{"id": i} for i in range(5)],
            collate,
            TrainConfig(batch_size=2, grad_accum=3, lr=0, bf16=False, max_grad_norm=1000),
            ObjectiveConfig(),
            device=torch.device("cpu"),
            net=net,
        )
        mark(output, rank, "train-return", references)
        (Path(output) / f"rank-{rank}.json").write_text(
            json.dumps({"summary": summary, "seen": seen, "grads": grads}), encoding="utf-8"
        )
        completed = True
        mark(output, rank, "result-written", references)
    finally:
        if variant in ("restore_only", "cleanup"):
            loop.accumulate = original
            mark(output, rank, "monkeypatch-restored", references)
        if variant in ("release_ddp", "cleanup"):
            net = None
            if variant == "cleanup":
                observe = None
                model = None
            gc.collect()
            mark(output, rank, "explicit-release-and-gc", references)
        if variant == "barrier" and completed:
            mark(output, rank, "barrier-enter", references)
            dist.barrier()
            mark(output, rank, "barrier-exit", references)
        mark(output, rank, "destroy-enter", references)
        dist.destroy_process_group()
        mark(output, rank, "destroy-exit", references)
    mark(output, rank, "worker-return", references)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--trace", default="")
    args = parser.parse_args()
    output, workspace = args.output.resolve(), args.workspace.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "environment.json").exists():
        parser.error("diagnostic output already contains a run; use a new directory")
    install_trace(args.trace)
    sys.path[:0] = [str(workspace / "src"), str(workspace)]

    import pytest
    import torch
    import torch.multiprocessing as mp

    facts = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "torch_git": torch.version.git_version,
        "variant": args.variant,
        "affinity": sorted(os.sched_getaffinity(0)),
        "parallel": torch.__config__.parallel_info(),
        "env": {
            name: os.getenv(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        "weakref_scope": "Python wrappers only; not proof of C++ backend destruction or joins",
    }
    (output / "environment.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    mp.spawn(
        worker,
        args=(str(output), str(workspace), args.variant, args.trace),
        nprocs=2,
    )
    rows = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(2)]
    seen = [i for row in rows for i in row["seen"]]
    # Preserve the original fixture's sample-count and analytical-gradient assertions.
    assert len(seen) == 6
    reference = sum((1 / (1 + math.exp(-(i + 1)))) * (i + 1) for i in seen) / len(seen)
    for row in rows:
        assert row["summary"]["steps"] == 1
        assert row["summary"]["sample_presentations"] == 6
        assert row["grads"] == pytest.approx([reference], rel=1e-6)
    (output / "assertions.json").write_text(
        json.dumps({"passed": True, "global_presentations": len(seen), "reference": reference}),
        encoding="utf-8",
    )
    print(f"PASS {args.variant} {output}", flush=True)


if __name__ == "__main__":
    main()
