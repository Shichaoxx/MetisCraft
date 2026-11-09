"""Fresh-process coverage of Gloo cleanup, including DDP's lazy dependencies."""

import json
from pathlib import Path

import pytest


def _gloo_thread_names():
    directory = Path("/proc/self/task")
    if not directory.is_dir():
        return []
    names = []
    for item in directory.iterdir():
        try:
            name = (item / "comm").read_text().strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if name in ("gloo_tcp_loop", "pt_gloo_runloop"):
            names.append(name)
    return names


def _lifecycle_worker(rank, rendezvous, output):
    import weakref
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from metiscraft.train.distributed import init_process_group

    torch.set_num_threads(1)

    def exercise():
        init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=40),
        )
        group_ref = weakref.ref(dist.group.WORLD)
        net = None
        try:
            model = torch.nn.Linear(1, 1, bias=False)
            net = DistributedDataParallel(model)
            loss = net(torch.tensor([[float(rank + 1)]])).sum()
            loss.backward()
            assert model.weight.grad.item() == pytest.approx(1.5)
        finally:
            net = None
            dist.destroy_process_group()
        return group_ref

    group_ref = exercise()
    # Evaluate after the training frame has returned: a retained function default
    # must not keep the group or its native threads alive until Python finalizes.
    assert not dist.is_initialized()
    assert group_ref() is None, "a lazy dependency retained the destroyed process group"
    threads = _gloo_thread_names()
    assert not threads, f"Gloo threads survived process-group cleanup: {threads}"
    Path(output, f"lifecycle-rank-{rank}.json").write_text(
        json.dumps({"group_released": True, "gloo_threads": threads})
    )


def test_cpu_ddp_releases_group_and_threads_before_interpreter_shutdown(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU gloo is unavailable")
    mp.spawn(
        _lifecycle_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
    )
    for rank in range(2):
        evidence = json.loads((tmp_path / f"lifecycle-rank-{rank}.json").read_text())
        assert evidence == {"group_released": True, "gloo_threads": []}
