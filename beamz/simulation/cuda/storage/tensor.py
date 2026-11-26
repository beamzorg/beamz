"""Full-tensor payloads for the existing streamed phase and program targets."""

import jax.numpy as jnp

from beamz.const import EPS_0
from beamz.simulation import _cuda_abi as abi
from beamz.simulation.tensor_regions import TensorRegion


def coefficients(ctx, coeffs):
    """Form the true inverse-diagonal update once per compiled invocation."""
    scale = jnp.float32(ctx.dt) / jnp.float32(EPS_0)
    if coeffs.e_source_x.ndim == 2 and coeffs.e_source_x.dtype == jnp.int32:
        return coeffs._replace(
            **{
                f"e_decay_{c}": scale * getattr(coeffs, f"e_inverse_diagonal_{c}")
                for c in "xyz"
            }
        )
    return coeffs._replace(
        **{
            f"e_{kind}_{c}": jnp.float32(1)
            if kind == "decay"
            else scale * getattr(coeffs, f"e_inverse_diagonal_{c}")
            for c in "xyz"
            for kind in ("decay", "source")
        }
    )


def payload(ctx, coeffs, *, scratch_factory=jnp.empty):
    if ctx.differentiable:
        raise ValueError("Full-tensor CUDA autodiff requires backend='jax'.")
    cross = coeffs.e_inverse_offdiagonal
    if isinstance(cross, TensorRegion):
        outer = tuple(
            (max(0, lo - 1), min(n, hi + 1))
            for (lo, hi), n in zip(cross.bounds, cross.node_shape, strict=True)
        )
        values = jnp.pad(
            cross.values,
            tuple(
                (lo - start, stop - hi)
                for (lo, hi), (start, stop) in zip(cross.bounds, outer, strict=True)
            )
            + ((0, 0),),
        )
        origin = tuple(lo for lo, _ in outer)
    else:
        values, origin = cross, (0, 0, 0)
    widths = tuple(
        jnp.float32(ctx.resolution)
        if ctx.config.metric_kind == "isotropic_uniform"
        else jnp.reciprocal(getattr(ctx.metrics, "e_to_h_" + axis))
        for axis in "zyx"
    )
    scratch = tuple(
        scratch_factory(
            tuple(n - int(a == 2 - c) for a, n in enumerate(values.shape[:3])),
            dtype=jnp.float32,
        )
        for c in range(3)
    )
    return (values, jnp.asarray(origin, jnp.int32), *widths, *scratch)


def append_io(arguments, results, aliases, tensor):
    aliases = dict(aliases)
    for c in range(3):
        aliases[len(arguments) + abi.TENSOR_INPUT_COUNT - 3 + c] = len(results) + c
    return (*arguments, *tensor), (*results, *tensor[-3:]), aliases


def run_steps(
    state,
    ctx,
    coeffs,
    groups,
    monitors,
    nsteps,
    *,
    observation_origin,
    observation_step_offset,
):
    """Reuse the in-place CUDA graph's source/DFT ordering with a tensor payload."""
    from beamz.simulation.cuda import runtime as rt

    tensor = payload(ctx, coeffs)
    coeffs = coefficients(ctx, coeffs)
    args, results, aliases = rt._graph_io(state, ctx, coeffs)
    args = (*args, *rt._source_group_arguments(groups))
    args, results, aliases, dft_start, count = rt._monitor_io(
        state,
        monitors,
        args,
        results,
        aliases,
        observation_origin=observation_origin,
        observation_step_offset=observation_step_offset,
    )
    args, results, aliases = append_io(args, results, aliases, tensor)
    plan = rt.NativeSchedulePlan(
        layout=abi.PROGRAM_LAYOUT_MONITOR_IN_PLACE,
        flags=rt._native_schedule_flags(ctx, coeffs, sources=True, monitors=count),
        monitor_count=count,
        coincident_source_group_mask=rt._coincident_source_group_mask(groups),
        disjoint_source_group_mask=rt._disjoint_source_group_mask(groups),
    )
    outputs = rt._ffi_call(abi.CUDA_PROGRAM_TARGET, results, aliases)(
        *args, **rt._program_attributes(ctx, nsteps, plan)
    )
    result = (
        rt._replace_graph_outputs(state, outputs)
        if ctx.boundary.cpml.enabled
        else rt._replace_fields(state, outputs)
    )
    return rt._replace_dft(result, outputs, dft_start) if count else result
