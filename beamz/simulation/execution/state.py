"""Initial and continuation state allocation for compiled simulations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation import observe as monitor_runtime
from beamz.simulation import sharding as sharding_runtime
from beamz.simulation.backend import CUDA_BF16_PSI
from beamz.simulation.model import CompiledProgram, SimulationState
from beamz.simulation.preparation_trace import trace_preparation


@trace_preparation("initial_state")
def initial_program_state(
    program: CompiledProgram,
    *,
    t: float,
    current_step: int,
    continuation: SimulationState | None = None,
    monitor_steps: int | None = None,
) -> SimulationState:
    """Allocate or restore every runtime buffer required by a compiled plan."""
    from beamz.simulation import execute as execution_runtime

    cpml, layout = program.boundary.cpml, program.sharding.layout
    psi_dtype = None
    if (
        program.config.backend in {"cuda_streamed", "cuda_mstep"}
        and cpml.enabled
        and program.config.cuda_flags & CUDA_BF16_PSI
    ):
        psi_dtype = jnp.bfloat16

    def field(name):
        if continuation is None and layout.enabled:
            value = getattr(program.grid, name)
            if isinstance(value, np.ndarray) and all(
                stride == 0 for stride in value.strides
            ):
                shape = layout.padded_shapes[name]
                target = sharding_runtime._array_sharding(
                    program,
                    jax.ShapeDtypeStruct(shape, value.dtype),
                    program.sharding.mesh,
                )
                return jax.jit(
                    lambda: jnp.zeros(shape, dtype=value.dtype), out_shardings=target
                )()
        # Fresh runs use the compiled lattice; continuations supply evolved canonical
        # arrays without reconstructing a mutable field container.
        if continuation is None:
            return execution_runtime._copy_initial_field(getattr(program.grid, name))
        value = getattr(continuation, name.lower())
        dtype = field_dtype(name)
        return value if value.dtype == dtype else value.astype(dtype)

    def field_dtype(name):
        # Query metadata without copying two additional full fields just to
        # choose the CPML dtype during fresh-state construction.
        dtype = getattr(program.grid, name).dtype
        if continuation is not None:
            dtype = jnp.result_type(dtype, getattr(continuation, name.lower()).dtype)
        return dtype

    def zeros(shape, dtype):
        shape = tuple(int(value) for value in shape)
        return (
            np.broadcast_to(np.zeros((), dtype=np.dtype(dtype)), shape)
            if layout.enabled
            else jnp.zeros(shape, dtype=dtype)
        )

    # Continue compatible packed CPML memories; a changed boundary plan starts clean.
    def restore_psi(old, terms, dtype):
        dtype = dtype if psi_dtype is None else psi_dtype
        shapes = tuple(term.slab.shape for term in terms)
        if len(old) == len(shapes) and all(
            tuple(value.shape)
            in (shape, sharding_runtime.logical_cpml_shape(layout, term))
            for value, shape, term in zip(old, shapes, terms, strict=True)
        ):
            return tuple(
                sharding_runtime._pad_high_to_shape(
                    # Host-created setup slabs stay on the host, but evolved
                    # device slabs must not round-trip through NumPy at every
                    # continuation boundary.
                    (np.asarray if isinstance(value, np.ndarray) else jnp.asarray)(
                        value, dtype=dtype
                    ),
                    shape,
                    pad_value=0.0,
                )
                for value, shape in zip(old, shapes, strict=True)
            )
        return tuple(zeros(shape, dtype) for shape in shapes)

    old_h = () if continuation is None else continuation.cpml_psi_h_terms
    old_e = () if continuation is None else continuation.cpml_psi_e_terms
    recorder_shapes = tuple(
        shape
        for spec in program.monitors
        if spec.recorder_index >= 0
        for shape in spec.field_shapes
    )
    # Check metadata before allocating: compatible histories can be much larger
    # than the fields, and must not need a second bank just to restore a state.
    if (
        program.monitors
        and continuation is not None
        and continuation.dft_vec_re.shape
        == (monitor_runtime.monitor_dft_value_size(program.monitors),)
        and continuation.dft_vec_im.shape
        == (monitor_runtime.monitor_dft_value_size(program.monitors),)
        and continuation.dft_weight_sum.shape
        == (monitor_runtime.monitor_dft_weight_size(program.monitors),)
        and tuple(value.shape[1:] for value in continuation.recorded_fields)
        == recorder_shapes
    ):
        monitor_values = {
            name: getattr(continuation, name) for name in monitor_runtime.MONITOR_FIELDS
        }
    else:
        monitor_values = monitor_runtime.empty_monitor_values(
            program,
            num_steps=max(
                1,
                int(
                    program.config.num_steps if monitor_steps is None else monitor_steps
                ),
            ),
        )
    # A continuation can promote the fields (for example, Periodic -> Bloch).
    # Promote recorder storage too, retaining all previous samples and metadata.
    recorded_fields = list(monitor_values["recorded_fields"])
    for spec in program.monitors:
        if spec.recorder_index < 0:
            continue
        for component, index in zip(
            spec.canonical_components, spec.field_buffer_indices, strict=True
        ):
            value = recorded_fields[index]
            dtype = jnp.result_type(value.dtype, field_dtype(component))
            if value.dtype != dtype:
                recorded_fields[index] = value.astype(dtype)
    monitor_values["recorded_fields"] = tuple(recorded_fields)
    from beamz.simulation.dispersion import initial_polarization

    return SimulationState(
        polarization=initial_polarization(program.dispersion, continuation)
        if program.dispersion is not None
        else (),
        ex=field("Ex"),
        ey=field("Ey"),
        ez=field("Ez"),
        hx=field("Hx"),
        hy=field("Hy"),
        hz=field("Hz"),
        cpml_psi_h_terms=restore_psi(old_h, cpml.h_terms, field_dtype("Hx")),
        cpml_psi_e_terms=restore_psi(old_e, cpml.e_terms, field_dtype("Ez")),
        **monitor_values,
        t=jnp.asarray(t, dtype=jnp.float32),
        current_step=jnp.asarray(current_step, dtype=jnp.int32),
    )
