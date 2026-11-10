"""Process-group setup shared by training and CPU distributed regressions."""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager
from typing import Any

import torch.distributed as dist

LOGGER = logging.getLogger(__name__)


@contextmanager
def process_group_scope():
    """Release a group created in this scope, including when setup or training fails.

    A caller-owned group remains the caller's responsibility. Teardown adds no
    collective: a failed rank must not enter another barrier during cleanup.
    """
    initialized_before = dist.is_initialized()
    try:
        yield
    except BaseException:
        if not initialized_before and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                LOGGER.exception("Process-group cleanup failed while handling the original error")
        raise
    else:
        if not initialized_before and dist.is_initialized():
            dist.destroy_process_group()


def init_process_group(backend: str, **kwargs: Any) -> None:
    # PyTorch's public autograd collectives bind group.WORLD in function defaults.
    # DDP/Dynamo may first import them after a group exists, retaining that group
    # and its Gloo threads until interpreter shutdown. Load them before creation.
    if backend == "gloo":
        importlib.import_module("torch.distributed.nn.functional")
    dist.init_process_group(backend, **kwargs)
