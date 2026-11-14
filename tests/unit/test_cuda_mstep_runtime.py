"""CPU contracts for two-step workspace ownership and native result routing."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.cuda.native import abi
from beamz.simulation.compile.backend import CudaBackendUnavailable
from beamz.simulation.cuda.native import mstep
from beamz.simulation.cuda import runtime
from beamz.simulation.cuda.storage import mstep_plan_inputs
from tests.unit.test_cuda_runtime_contract import _program_and_state


@pytest.mark.parametrize("cpml,monitor", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("steps,bound_buffers", [(2, False), (4, True), (5, False)])
def test_mstep_native_workspace_aliases_clocks_and_output_banks(
    monkeypatch, cpml, monitor, steps, bound_buffers
):
    program, state, context = _program_and_state(
        cpml=cpml, monitor=monitor, shape=(17, 41, 57)
    )
    packed = runtime.pack_dft_monitors(program.monitors) if monitor else None
    groups = (None,) * 9
    plan = mstep.build_mstep_plan(*mstep_plan_inputs(program, groups, packed))
    assert plan.core_cells > 0
    buffers = tuple(jnp.asarray(a) for a in mstep._plan_arrays(plan))
    fields = runtime._fields(state)
    state_count = 18 if cpml else 6
    monitor_start = state_count + 12
    captured = []

    def ffi(target, metadata, **options):
        assert target == abi.CUDA_MSTEP_TARGET
        assert options["vmap_method"] == "sequential"
        aliases = options["input_output_aliases"]
        assert sorted(aliases.values()) == list(range(len(metadata)))
        assert len(metadata) == monitor_start + 6
        for index, field in enumerate(fields):
            assert metadata[state_count + index].shape == field.shape
            assert metadata[state_count + 6 + index].shape == (plan.scratch_cells,)

        def call(*arguments, **attributes):
            captured.append(attributes)
            assert attributes["nsteps"] == steps
            assert attributes["temporal_steps"] == 2
            assert attributes["program_layout"] == abi.PROGRAM_LAYOUT_YEE_IN_PLACE
            assert attributes["monitor_count"] == int(monitor)
            assert bool(attributes["schedule_flags"] & abi.NATIVE_SCHEDULE_CPML) == cpml
            assert (
                bool(attributes["schedule_flags"] & abi.NATIVE_SCHEDULE_MONITORS)
                == monitor
            )
            assert float(arguments[-8]) == 0.5
            assert arguments[-7] is state.current_step
            assert int(arguments[-6]) == 19
            for actual, expected in zip(arguments[-5:], buffers, strict=True):
                np.testing.assert_array_equal(actual, expected)
            outputs = [None] * len(metadata)
            for input_index, output_index in aliases.items():
                value = arguments[input_index]
                if state_count + 6 <= output_index < monitor_start:
                    np.testing.assert_array_equal(value, np.zeros(plan.scratch_cells))
                outputs[output_index] = jnp.full_like(value, output_index + 1)
            return tuple(outputs)

        return call

    monkeypatch.setattr(jax.ffi, "ffi_call", ffi)
    result = mstep.run_mstep(
        state,
        context,
        program.coefficients,
        groups,
        packed,
        plan,
        steps,
        observation_origin=0.5,
        observation_step_offset=19,
        plan_buffers=buffers if bound_buffers else None,
    )
    assert len(captured) == 1
    bank = state_count if (steps // 2) % 2 else 0
    for index, value in enumerate(runtime._fields(result)):
        np.testing.assert_array_equal(value, np.full(value.shape, bank + index + 1))
    if cpml:
        for index, value in enumerate(
            (*result.cpml_psi_h_terms, *result.cpml_psi_e_terms)
        ):
            np.testing.assert_array_equal(value, np.full(value.shape, index + 7))
    if monitor:
        for index, value in enumerate(
            (result.dft_vec_re, result.dft_vec_im, result.dft_weight_sum)
        ):
            np.testing.assert_array_equal(
                value, np.full(value.shape, monitor_start + index + 1)
            )
    else:
        assert result.dft_vec_re is state.dft_vec_re
    assert result.t is state.t
    assert result.current_step is state.current_step


@pytest.mark.parametrize("steps,core_cells", [(1, 1), (4, 0)])
def test_mstep_fallback_preserves_clock_and_forces_capacity_schedule(
    monkeypatch, steps, core_cells
):
    program, state, context = _program_and_state(cpml=False)
    groups = (None,) * 9
    plan = mstep.build_mstep_plan(*mstep_plan_inputs(program, groups, None))
    plan = replace(plan, core_cells=core_cells)
    calls = []

    def native(actual_state, ctx, coeffs, sources, monitors, nsteps, **clock):
        assert actual_state is state
        assert ctx.low_memory
        assert coeffs is program.coefficients
        assert sources is groups
        assert monitors is None
        calls.append((nsteps, clock))
        return state

    monkeypatch.setattr(runtime, "run_native_steps", native)
    assert (
        mstep.run_mstep(
            state,
            context,
            program.coefficients,
            groups,
            None,
            plan,
            steps,
            observation_origin=0.5,
            observation_step_offset=19,
        )
        is state
    )
    assert calls == [
        (steps, {"observation_origin": 0.5, "observation_step_offset": 19})
    ]


def test_mstep_rejects_non_float32_fields_before_lowering():
    program, state, context = _program_and_state(cpml=False, shape=(17, 41, 57))
    groups = (None,) * 9
    plan = mstep.build_mstep_plan(*mstep_plan_inputs(program, groups, None))
    state = state._replace(hx=state.hx.astype(jnp.bfloat16))
    with pytest.raises(CudaBackendUnavailable, match="float32 fields"):
        mstep.run_mstep(
            state,
            context,
            program.coefficients,
            groups,
            None,
            plan,
            2,
            observation_origin=0.0,
            observation_step_offset=0,
        )


def test_bound_plan_buffers_survive_lowering_and_compilation():
    from tests.unit.test_cuda_mstep_plan import _plan

    plan, _ = _plan((3, 4, 5))

    def scan(state, coefficients, dispersion, buffers):
        assert dispersion is None
        return state * coefficients + sum(jnp.sum(value) for value in buffers)

    bound = mstep.bind_plan_buffers(jax.jit(scan), plan)
    expected = 6 + sum(np.sum(value) for value in mstep._plan_arrays(plan))
    assert int(bound(2, 3)) == expected
    lowered = bound.lower(2, 3)
    assert lowered.compiler_ir() is not None
    compiled = lowered.compile()
    assert compiled.buffers is bound.buffers
    assert int(compiled(2, 3)) == expected
    assert compiled.cost_analysis() is not None
    assert bound.clear_cache() is None
