"""Native streamed-CUDA corrections for tensor permeability and tensor-pole media.

Each tensor-pole region takes two ``beamz_cuda_tensor_media`` launches per
half-step: the pole advance, then the inverse-background correction applied in
place to the field box. They reproduce
:meth:`beamz.simulation.anisotropic.TensorPolarizationRegion.advance` and
``apply_tensor_operator`` exactly, up to float32 rounding order. The inverse
launch also maps full static tensors (non-symmetric eps or mu, conductivity
tensors) through their antisymmetric cross terms. For CCPR regions
(:mod:`beamz.simulation.ccpr`) the pole launch also applies the explicit half
``k r`` of the centred ``B`` term, ``k = (dt / (2 d)) B`` and ``r`` the
previous field change, and a third launch adds the implicit half
``k (E^{n+1} - E^n)`` to the pole state.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CUDA_ABI_VERSION

_OP_POLE = 0
_OP_INVERSE = 1
_OP_EDOT = 2


def _empty():
    return jnp.zeros((0,), dtype=jnp.float32)


def supports(box, dtype) -> bool:
    """Whether a box runs natively: 3D, float32, and without periodic seams."""
    return (
        len(box.components) == 3 and not box.seams and jnp.dtype(dtype) == jnp.float32
    )


def _attributes(box, *, op, coupled, poles=0, sign=0.0):
    cells = tuple(s.stop - s.start for s in box.cell_slices)
    origin = tuple(s.start for s in box.cell_slices)
    return {
        "abi_version": np.int32(CUDA_ABI_VERSION),
        "op": np.int32(op),
        "family": np.int32(0 if box.family == "E" else 1),
        "coupled": np.int32(coupled),
        "poles": np.int32(poles),
        "cells_z": np.int32(cells[0]),
        "cells_y": np.int32(cells[1]),
        "cells_x": np.int32(cells[2]),
        "origin_z": np.int32(origin[0]),
        "origin_y": np.int32(origin[1]),
        "origin_x": np.int32(origin[2]),
        "sign": np.float32(sign),
    }


def _widths(box):
    """Cell widths in storage order (z, y, x); empty for uniform axes."""
    if box.widths is None:
        return (_empty(),) * 3
    return tuple(
        _empty() if w is None else jnp.asarray(w, jnp.float32)
        for w in reversed(box.widths)
    )


def _f32(value, shape):
    return jnp.broadcast_to(jnp.asarray(value, jnp.float32), shape)


def advance_poles(region, old_fields, flat):
    """Native ``region.advance``: (site delta, new pole state).

    For CCPR regions ``flat`` ends with the previous field change ``r``.
    """
    from . import runtime as rt

    box = region.box
    poles = int(np.shape(region.c1)[0])
    coupled = region.offdiagonal is not None
    # The flat state interleaves (current, previous) per component.
    current, previous = tuple(flat[0:6:2]), tuple(flat[1:6:2])
    edot = getattr(region, "edot_diagonal", ())
    if edot:
        explicit = (
            *(_f32(v, s) for v, s in zip(flat[6:9], box.shapes, strict=True)),
            *(_f32(d, (poles, *s)) for d, s in zip(edot, box.shapes, strict=True)),
            _empty()
            if region.edot_offdiagonal is None
            else jnp.asarray(region.edot_offdiagonal, jnp.float32),
        )
    else:
        explicit = (_empty(),) * 7
    arguments = (
        *(_f32(v, s) for v, s in zip(old_fields, box.shapes, strict=True)),
        *_widths(box),
        jnp.asarray(region.c1, jnp.float32).reshape(-1),
        jnp.asarray(region.c2, jnp.float32).reshape(-1),
        *(
            _f32(d, (poles, *s))
            for d, s in zip(region.diagonal, box.shapes, strict=True)
        ),
        jnp.asarray(region.offdiagonal, jnp.float32) if coupled else _empty(),
        *current,
        *previous,
        *explicit,
    )
    results = (
        *current,
        *previous,
        *(jnp.empty(s, jnp.float32) for s in box.shapes),
    )
    aliases = {12 + k: k for k in range(6)}
    outputs = rt._ffi_call(abi.CUDA_TENSOR_MEDIA_TARGET, results, aliases)(
        *arguments, **_attributes(box, op=_OP_POLE, coupled=coupled, poles=poles)
    )
    new_flat = tuple(
        x for pair in zip(outputs[0:3], outputs[3:6], strict=True) for x in pair
    )
    return tuple(outputs[6:9]), new_flat


def apply_inverse(state, box, inverse, source, *, base=None, sign):
    """``field = (base or field) + sign * Inv(source)`` on the box, in place.

    ``inverse`` may carry antisymmetric cross terms (``inverse.skew``).
    """
    from . import runtime as rt

    names = tuple(c.lower() for c in box.components)
    fields = tuple(getattr(state, name) for name in names)
    coupled = inverse.offdiagonal is not None
    if base is None:
        base = (_empty(),) * 3
    else:
        base = tuple(_f32(v, s) for v, s in zip(base, box.shapes, strict=True))
    arguments = (
        *fields,
        *(_f32(v, s) for v, s in zip(source, box.shapes, strict=True)),
        *base,
        *_widths(box),
        *(_f32(d, s) for d, s in zip(inverse.diagonal, box.shapes, strict=True)),
        jnp.asarray(inverse.offdiagonal, jnp.float32) if coupled else _empty(),
        _empty() if inverse.skew is None else jnp.asarray(inverse.skew, jnp.float32),
    )
    outputs = rt._ffi_call(
        abi.CUDA_TENSOR_MEDIA_TARGET, fields, {k: k for k in range(3)}
    )(*arguments, **_attributes(box, op=_OP_INVERSE, coupled=coupled, sign=sign))
    return state._replace(**dict(zip(names, outputs, strict=True)))


def add_edot(region, change, flat):
    """Native CCPR update ``p += (dt / d) B change`` per pole; returns the flat state."""
    from . import runtime as rt

    box = region.box
    poles = int(np.shape(region.c1)[0])
    coupled = region.edot_offdiagonal is not None
    current = tuple(flat[0::2])
    arguments = (
        *(_f32(v, s) for v, s in zip(change, box.shapes, strict=True)),
        *_widths(box),
        *(
            _f32(d, (poles, *s))
            for d, s in zip(region.edot_diagonal, box.shapes, strict=True)
        ),
        jnp.asarray(region.edot_offdiagonal, jnp.float32) if coupled else _empty(),
        *current,
    )
    outputs = rt._ffi_call(
        abi.CUDA_TENSOR_MEDIA_TARGET, current, {10 + k: k for k in range(3)}
    )(*arguments, **_attributes(box, op=_OP_EDOT, coupled=coupled, poles=poles))
    return tuple(x for pair in zip(outputs, flat[1::2], strict=True) for x in pair)
