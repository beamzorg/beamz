"""An old component must never receive undersized compact scratch."""

from types import SimpleNamespace

import pytest

from beamz.simulation import backend
from beamz.simulation._cuda_abi import (
    CUDA_ABI_VERSION,
    CUDA_MSTEP_FULL_INTERMEDIATES_TARGET,
    CUDA_MSTEP_TARGET,
    CUDA_STREAMED_TARGETS,
)
from beamz.simulation.cuda.capabilities import mstep_allocation_contract


def extension(*optional):
    return SimpleNamespace(
        __abi_version__=CUDA_ABI_VERSION,
        registrations=lambda: {
            name: object()
            for name in {*CUDA_STREAMED_TARGETS, CUDA_MSTEP_TARGET, *optional}
        },
    )


@pytest.mark.parametrize("full", [False, True])
def test_allocation_contract_requires_explicit_optional_registration(monkeypatch, full):
    module = extension(*([CUDA_MSTEP_FULL_INTERMEDIATES_TARGET] if full else []))
    monkeypatch.setattr(backend, "_load_extension", lambda: module)
    target, cells = mstep_allocation_contract(128 * 10000)
    assert target == (
        CUDA_MSTEP_FULL_INTERMEDIATES_TARGET if full else CUDA_MSTEP_TARGET
    )
    assert cells == (128 if full else 128 * 10000)
    assert CUDA_MSTEP_TARGET in backend._validated_registrations(module)


def test_missing_component_keeps_conservative_scratch(monkeypatch):
    def missing():
        raise ImportError("optional extension absent")

    monkeypatch.setattr(backend, "_load_extension", missing)
    assert mstep_allocation_contract(4096) == (CUDA_MSTEP_TARGET, 4096)


def test_wrong_abi_cannot_advertise_a_smaller_workspace(monkeypatch):
    module = extension(CUDA_MSTEP_FULL_INTERMEDIATES_TARGET)
    module.__abi_version__ = -1
    monkeypatch.setattr(backend, "_load_extension", lambda: module)
    assert mstep_allocation_contract(4096) == (CUDA_MSTEP_TARGET, 4096)
