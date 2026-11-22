"""The named CUDA boundary preserves ABI layout, aliases and state ownership."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.cuda.engine._capabilities import _Unsupported, _validate
from beamz.simulation.cuda.engine._contracts import (
    _DFT,
    _FIELDS,
    _IN,
    _OUT,
    _ZEROABLE,
    _Packet,
    _StateContract,
)


def test_packet_preserves_buffer_identity_and_freezes_attributes():
    values = tuple(np.asarray(i, np.int32) for i in range(134))
    attributes = {"nsteps": 17}
    packet = _Packet(values, attributes)
    attributes["nsteps"] = 19
    assert packet.attributes["nsteps"] == 17
    with pytest.raises(TypeError):
        packet.attributes["nsteps"] = 23
    with pytest.raises(FrozenInstanceError):
        packet.arguments = ()
    state = packet._state_inputs()
    for name in (*_FIELDS, *_DFT):
        assert state[name] is values[_IN[name]]
    assert all(
        v is values[abi.CPML_H_PSI_INPUT_OFFSET + i]
        for i, v in enumerate(state["cpml_psi_h_terms"])
    )
    assert all(
        v is values[abi.CPML_E_PSI_INPUT_OFFSET + i]
        for i, v in enumerate(state["cpml_psi_e_terms"])
    )
    assert _IN["workspace_hx"] == abi.TEMPORAL_FIELD_WORKSPACE_INPUT
    assert _IN["workspace_psi_0"] == abi.TEMPORAL_PSI_WORKSPACE_INPUT
    assert [_OUT[n] for n in _DFT] == [30, 31, 32]
    assert not any(
        _IN[n] in _ZEROABLE
        for n in (
            "h_metric_z",
            "e_metric_x",
            "monitor_frequencies",
            "time",
            "elapsed_steps",
        )
    )
    with pytest.raises(ValueError, match="buffer contract"):
        _Packet(values[:-1])


@pytest.mark.parametrize("ownership", ["private", "preserved", "donated"])
def test_state_horizon_and_ownership_are_independent(ownership):
    contract = _StateContract(
        total_steps=29, initial_step=17, steps=12, ownership=ownership
    )
    assert contract._stop == 29 and contract.ownership == ownership
    with pytest.raises(ValueError, match="outside"):
        _StateContract(29, 30, 1, ownership)
    with pytest.raises(ValueError, match="num_steps"):
        _StateContract(29, 17, 13, ownership)


@pytest.mark.parametrize("progress", [False, True])
def test_public_continuation_reports_preservation_and_absolute_horizon(progress):
    import beamz as bz

    sim = bz.Simulation(
        domain=(0.4e-6, 0.3e-6), resolution=0.1e-6, time=np.arange(9) * 1e-17
    )
    first = sim.advance(
        num_steps=3, backend="jax", progress=progress, performance=False
    )
    saved = np.asarray(first.state.ex).copy()
    second = sim.advance(
        state=first.state,
        num_steps=2,
        backend="jax",
        progress=progress,
        performance=False,
    )
    assert first.results.cuda_plan["state_ownership"] == "private"
    assert second.results.cuda_plan["state_ownership"] == "preserved"
    assert (
        second.results.cuda_plan["initial_step"] == 3
        and second.results.cuda_plan["final_step"] == 5
    )
    np.testing.assert_array_equal(first.state.ex, saved)
    third = sim.advance(
        state=second.state,
        num_steps=2,
        backend="jax",
        donate_state=True,
        performance=False,
    )
    assert (
        third.results.cuda_plan["state_ownership"] == "donated"
        and int(third.state.current_step) == 7
    )


@pytest.mark.parametrize(
    "feature", ["tensor", "dispersion", "dtype", "sharding", "recorder"]
)
def test_native_capability_failure_has_one_owner(feature):
    p = SimpleNamespace(
        coefficients=SimpleNamespace(e_inverse_offdiagonal=np.empty(0)),
        dispersion=None,
        grid=SimpleNamespace(
            **{
                n: np.zeros((), np.float32)
                for n in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            }
        ),
        sharding=SimpleNamespace(layout=SimpleNamespace(enabled=False)),
        monitors=(),
    )
    _validate(p)
    if feature == "tensor":
        p.coefficients.e_inverse_offdiagonal = np.ones(1)
    elif feature == "dispersion":
        p.dispersion = SimpleNamespace(regions=(1,), interfaces=())
    elif feature == "dtype":
        p.grid.Ex = np.zeros((), np.float64)
    elif feature == "sharding":
        p.sharding.layout.enabled = True
    else:
        p.monitors = (SimpleNamespace(recorder_index=0),)
    with pytest.raises(_Unsupported):
        _validate(p)


def test_native_no_cpml_layout_preserves_existing_yee_abi():
    from beamz.simulation.cuda.engine._contracts import _CPML, _YEE

    assert (_CPML.input_count, _CPML.output_count) == (134, 36)
    assert (_YEE.input_count, _YEE.output_count) == (84, 24)
    values = tuple(np.asarray(i, np.int32) for i in range(84))
    packet = _Packet(values)
    assert packet._layout is _YEE
    assert _YEE.source == 36 and _YEE.monitor == 63
    assert [_YEE.outputs[n] for n in _DFT] == [18, 19, 20]
    assert packet._state_inputs()["cpml_psi_h_terms"] == ()
    assert packet._state_inputs()["cpml_psi_e_terms"] == ()
    for name in (*_FIELDS, *_DFT):
        assert packet._buffer(name) is values[_YEE.inputs[name]]
