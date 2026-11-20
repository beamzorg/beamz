"""Conservative allocation contracts advertised by the loaded CUDA component."""

from __future__ import annotations

from beamz.simulation._cuda_abi import (
    CUDA_MSTEP_FULL_INTERMEDIATES_TARGET,
    CUDA_MSTEP_TARGET,
)


def mstep_allocation_contract(scratch_cells: int) -> tuple[str, int]:
    """Select an additive target before shrinking its unused compact buffers.

    A nominal backend/ABI/version does not establish this capability. The
    registered implementation must explicitly promise full-bank intermediates.
    Importing metadata does not initialize a GPU or register FFI targets.
    """
    from beamz.simulation import backend

    try:
        targets = backend._validated_registrations(backend._load_extension())
    except (ImportError, OSError, backend.CudaBackendUnavailable):
        targets = {}
    if CUDA_MSTEP_FULL_INTERMEDIATES_TARGET in targets:
        return CUDA_MSTEP_FULL_INTERMEDIATES_TARGET, 128
    return CUDA_MSTEP_TARGET, scratch_cells
