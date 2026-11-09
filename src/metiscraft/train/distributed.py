"""Process-group setup shared by training and CPU distributed regressions."""

from __future__ import annotations

import importlib
from typing import Any

import torch.distributed as dist


def init_process_group(backend: str, **kwargs: Any) -> None:
    # PyTorch's public autograd collectives bind group.WORLD in function defaults.
    # DDP/Dynamo may first import them after a group exists, retaining that group
    # and its Gloo threads until interpreter shutdown. Load them before creation.
    if backend == "gloo":
        importlib.import_module("torch.distributed.nn.functional")
    dist.init_process_group(backend, **kwargs)
