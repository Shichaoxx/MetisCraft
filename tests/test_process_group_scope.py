"""Ownership and exception contracts for recipe process-group teardown."""

import pytest

from metiscraft.train.distributed import process_group_scope


@pytest.fixture
def group(monkeypatch):
    from metiscraft.train import distributed

    state = {"initialized": False, "destroyed": 0, "error": None}

    def destroy():
        state["destroyed"] += 1
        if state["error"]:
            raise state["error"]
        state["initialized"] = False

    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(distributed.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(
        distributed.dist, "barrier", lambda: pytest.fail("cleanup must not add a collective")
    )
    return state


@pytest.mark.parametrize("error", [None, ValueError("training failed"), KeyboardInterrupt()])
def test_owned_group_is_released_on_return_error_and_interrupt(group, error):
    def exercise():
        with process_group_scope():
            group["initialized"] = True
            if error is not None:
                raise error

    if error is None:
        exercise()
    else:
        with pytest.raises(type(error)) as caught:
            exercise()
        assert caught.value is error
    assert group["destroyed"] == 1
    assert not group["initialized"]


@pytest.mark.parametrize("fail", [False, True])
def test_caller_owned_group_is_preserved(group, fail):
    group["initialized"] = True

    def exercise():
        with process_group_scope():
            if fail:
                raise ValueError("caller will handle this")

    if fail:
        with pytest.raises(ValueError):
            exercise()
    else:
        exercise()
    assert group["initialized"]
    assert group["destroyed"] == 0


def test_failure_before_group_creation_does_not_attempt_cleanup(group):
    with pytest.raises(ValueError, match="setup failed"):
        with process_group_scope():
            raise ValueError("setup failed")
    assert group["destroyed"] == 0


def test_cleanup_failure_does_not_hide_original_error(group, caplog):
    original = ValueError("model initialization failed")
    group["error"] = RuntimeError("teardown failed")
    with pytest.raises(ValueError) as caught:
        with process_group_scope():
            group["initialized"] = True
            raise original
    assert caught.value is original
    assert group["destroyed"] == 1
    assert "teardown failed" in caplog.text


def test_cleanup_failure_after_success_is_reported(group):
    group["error"] = RuntimeError("teardown failed")
    with pytest.raises(RuntimeError, match="teardown failed"):
        with process_group_scope():
            group["initialized"] = True
    assert group["destroyed"] == 1
