from __future__ import annotations

from typing import Any

import numpy as np

from ._contracts import _FIELDS


class _Unsupported(ValueError):
    pass


_METRICS = ("isotropic_uniform", "axis_uniform", "rectilinear")
_STORAGE_AXES = (0, 1, 2)


def _validate(program: Any = "", ctx: Any = "", *, native_effects: bool = False) -> None:
    if (not native_effects and program.coefficients.e_inverse_offdiagonal.size or
        program.dispersion is not None and (program.dispersion.regions or program.dispersion.interfaces)):
        raise _Unsupported("tensor/dispersive physics uses the CUDA interoperability binding")
    if any(getattr(program.grid, name.capitalize()).dtype != np.dtype("float32") for name in _FIELDS):
        raise _Unsupported("non-FP32 fields use the CUDA interoperability binding")
    if program.sharding.layout.enabled: raise _Unsupported("Shared traced compiler requires an unsharded simulation")
    if any(
        (not native_effects and m.recorder_index >= 0)
        or m.accumulate_power or m.accumulate_frequency
        or (not native_effects and (not m.dft_enabled or m.freq_count <= 0 or m.dft_point_count <= 0))
        for m in program.monitors
    ):
        raise _Unsupported("Shared traced compiler admits field/flux DFT monitors; time recorders require another lowering")
    if isinstance(ctx, str): return

    if program.config.cuda_storage_axes != _STORAGE_AXES:
        raise _Unsupported("Native compiler currently requires canonical physical storage")
    if not native_effects and any(rest for _,rest in ctx.source_batches.values()):
        raise _Unsupported("Native compiler requires batched slab/volume sources")
