"""Lower fused 3D Yee phases to the optional BeamZ CUDA typed FFI targets."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CUDA_ABI_VERSION

_PHASE_H = 0
_PHASE_E = 1
_COMPONENT_CODE = {name: index for index, name in enumerate(("Hx", "Hy", "Hz"))}
_COMPONENT_CODE.update({name: index for index, name in enumerate(("Ex", "Ey", "Ez"))})
_EMPTY = np.empty((0,), dtype=np.float32)
_EMPTY_SOURCE_GROUP = (
    np.zeros((0, 1, 1, 1), dtype=np.float32),
    np.zeros((0, 1), dtype=np.float32),
    np.zeros((0, 3), dtype=np.int32),
)
_METRIC_KIND_CODE = {
    "isotropic_uniform": 0,
    "axis_uniform": 1,
    "rectilinear": 2,
}

_FIELD_COUNT = abi.FIELD_COUNT
_CPML_TERM_COUNT = abi.CPML_TERM_COUNT
_SOURCE_GROUP_COUNT = abi.SOURCE_GROUP_COUNT
_CPML_H_PSI_INPUT = abi.CPML_H_PSI_INPUT_OFFSET
_CPML_E_PSI_INPUT = abi.CPML_E_PSI_INPUT_OFFSET
_TEMPORAL_FIELD_WORKSPACE_INPUT = abi.TEMPORAL_FIELD_WORKSPACE_INPUT
_TEMPORAL_PSI_WORKSPACE_INPUT = abi.TEMPORAL_PSI_WORKSPACE_INPUT


@dataclass(frozen=True, slots=True)
class NativeSchedulePlan:
    """Validated compilation-time choices consumed by the native program ABI.

    The native launcher revalidates these capability bits against decoded buffer
    shapes before capture.  Keeping the decision here prevents Python, FFI
    decoding, program scheduling, and leaf kernels from independently guessing
    which specialization is sound.
    """

    layout: int
    flags: int
    monitor_count: int = 0
    coincident_source_group_mask: int = 0
    disjoint_source_group_mask: int = 0
    logical_shape: tuple[int, int, int] = (0, 0, 0)
    temporal_steps: int = 1

    @property
    def uses_temporal_fields(self) -> bool:
        return bool(self.flags & abi.NATIVE_SCHEDULE_TEMPORAL)


def _cpml_payload(terms):
    """Metadata and profile arrays in the shared forward/transpose ABI order."""
    return (
        _term_metadata(terms),
        *(value for term in terms for value in (term.a, term.b, term.inv_kappa)),
    )


def _phase_materials(coeffs, phase):
    """Native ABI order: x/y/z decay, then x/y/z source; arrays stay dynamic."""
    return tuple(
        getattr(coeffs, f"{phase}_{kind}_{axis}")
        for kind in ("decay", "source")
        for axis in "xyz"
    )


def _metallic_edge_mask(edges: frozenset[str]) -> int:
    order = ("front", "back", "bottom", "top", "left", "right")
    return sum(1 << index for index, name in enumerate(order) if name in edges)


def _boundary_code(edges: frozenset[str], terms=()) -> int:
    """Pack PEC faces and a CPML thickness safe for every supplied phase.

    A non-zero CPML thickness selects a CUDA specialization which deliberately
    skips the per-term packed-slab descriptor.  It is therefore valid only when
    *all* supplied H and E recurrences have the same symmetric slab.  Phase-only
    FFI calls pass one phase; native program calls pass both phases.
    """
    thickness = 0
    if terms:
        first = int(terms[0].slab.low)
        if first > 0 and all(
            int(term.slab.low) == first and int(term.slab.high) == first
            for term in terms
        ):
            thickness = first
    return _metallic_edge_mask(edges) | (thickness << 8)


def _term_metadata(terms) -> jnp.ndarray:
    return jnp.asarray(
        [
            (
                _COMPONENT_CODE[term.component],
                int(term.axis),
                int(term.slab.low),
                int(term.slab.high),
                1 if float(term.sign) > 0.0 else -1,
            )
            for term in terms
        ],
        dtype=jnp.int32,
    ).reshape((len(terms), 5))


def _shape(value):
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _fields(state):
    return (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)


def _ffi_call(target, result_values, aliases):
    return jax.ffi.ffi_call(
        target,
        tuple(_shape(value) for value in result_values),
        input_output_aliases=aliases,
        vmap_method="sequential",
    )


def _metric_kind_code(ctx) -> np.int32:
    try:
        return np.int32(_METRIC_KIND_CODE[ctx.config.metric_kind])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported CUDA derivative metric kind: {ctx.config.metric_kind!r}"
        ) from exc


def _program_attributes(ctx, nsteps: int, plan: NativeSchedulePlan):
    return {
        "abi_version": np.int32(CUDA_ABI_VERSION),
        "cuda_flags": np.int32(ctx.config.cuda_flags),
        "graph_cache_capacity": np.int32(ctx.config.cuda_graph_cache_capacity),
        "nsteps": np.int32(nsteps),
        "dt": np.float32(ctx.dt),
        "resolution": np.float32(ctx.resolution),
        "boundary_code": np.int32(
            _boundary_code(
                ctx.boundary.cpml.metallic_edges,
                (*ctx.boundary.cpml.h_terms, *ctx.boundary.cpml.e_terms),
            )
        ),
        "metric_kind": _metric_kind_code(ctx),
        "program_layout": np.int32(plan.layout),
        "cpml_enabled": np.int32(ctx.boundary.cpml.enabled),
        "monitor_count": np.int32(plan.monitor_count),
        "coincident_source_group_mask": np.int32(plan.coincident_source_group_mask),
        "disjoint_source_group_mask": np.int32(plan.disjoint_source_group_mask),
        "schedule_flags": np.int32(plan.flags),
        "logical_z": np.int32(plan.logical_shape[0]),
        "logical_y": np.int32(plan.logical_shape[1]),
        "logical_x": np.int32(plan.logical_shape[2]),
        "temporal_steps": np.int32(plan.temporal_steps),
    }
