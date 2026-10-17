"""Exercise calibration policy on CPU with a deterministic device boundary."""

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from beamz.simulation import _cuda_abi as abi
from beamz.simulation import backend, execute, sharding
from beamz.simulation.cuda import tuning
from beamz.simulation.model import RunConfig


@pytest.fixture
def calibration(monkeypatch, tmp_path):
    @dataclass
    class Program:
        config: RunConfig
        coefficients: object

    program = Program(
        RunConfig(
            resolution=1,
            dt=1,
            num_steps=1025,
            plane_2d="xy",
            is_3d=True,
            cuda_flags=abi.CUDA_DEFAULT_FLAGS,
        ),
        np.array([2], dtype=np.float32),
    )
    state = (np.array([1, 2, 3], dtype=np.float32),)
    control = SimpleNamespace(
        program=program,
        state=state,
        builds=[],
        calls=[],
        failures={},
        durations={},
        clock=0.0,
        path=tmp_path / "profile.json",
        signature=SimpleNamespace(t0=2.0),
    )
    monkeypatch.setattr(
        tuning.jax, "devices", lambda: [SimpleNamespace(device_kind="RTX 3090")]
    )
    monkeypatch.setattr(
        tuning, "_gpu_status", lambda: dict(uuid="test", driver="test", power_limit=350)
    )
    monkeypatch.setattr(tuning, "_safe_headroom", lambda: True)
    monkeypatch.setattr(
        backend,
        "cuda_backend_status",
        lambda **kw: SimpleNamespace(targets=(abi.CUDA_MSTEP_TARGET,)),
    )
    monkeypatch.setattr(tuning, "_cache_path", lambda *args: (control.path, "key"))
    monkeypatch.setattr(tuning.time, "perf_counter", lambda: control.clock)
    monkeypatch.setattr(sharding, "prepare_state", lambda p, s, **kw: s)
    monkeypatch.setattr(sharding, "place_tree", lambda p, c: c)

    def initial(p, **kwargs):
        assert p is control.program
        assert kwargs == dict(t=2.0, current_step=0, monitor_steps=1025)
        return state

    monkeypatch.setattr(execute, "initial_program_state", initial)

    def build(candidate):
        axes = candidate.config.cuda_storage_axes
        tile = "32x8" if candidate.config.cuda_flags & abi.CUDA_SHELL32X8 else "64x4"
        name = "".join(map(str, axes)) + "/" + tile
        if candidate.config.cuda_schedule == "mstep":
            name = "".join(map(str, axes)) + "/mstep"
        control.builds.append((name, candidate.config.num_steps))
        if control.failures.get(name) == "compile":
            raise ValueError("unsupported layout")
        if control.failures.get(name) == "unavailable":
            raise backend.CudaBackendUnavailable("workspace exceeds headroom")

        def run(s, coefficients):
            assert s is state
            assert coefficients is program.coefficients
            np.testing.assert_array_equal(s[0], [1, 2, 3])
            control.calls.append(name)
            control.clock += control.durations.get(
                name, 0.5 if name == "120/64x4" else 1.0
            )
            output = s[0] + coefficients[0]
            if control.failures.get(name) == "different":
                output[-1] += 1
            if control.failures.get(name) == "nonfinite":
                output[-1] = np.nan
            return (output,)

        return SimpleNamespace(
            lower=lambda s, c: SimpleNamespace(compile=lambda: run),
            cuda_schedule_decision=dict(
                selected="streamed"
                if control.failures.get(name) == "fallback"
                else candidate.config.cuda_schedule,
                reason="no eligible columns",
            ),
        )

    monkeypatch.setattr(execute, "build_scan", build)
    return control


def select(control, mode="calibrate"):
    return tuning._select_program(control.program, control.signature, (mode, "", ()))


def test_calibration_validates_complete_state_and_balances_timing(calibration):
    c = calibration
    c.failures = {
        "012/32x8": "compile",
        "120/32x8": "different",
        "201/32x8": "nonfinite",
    }
    selected = select(c)
    report = tuning.tuning_report(selected)
    assert selected.config.num_steps == 1025
    assert selected.config.cuda_storage_axes == (1, 2, 0)
    assert c.program.config.cuda_storage_axes == (0, 1, 2)
    assert {steps for _, steps in c.builds} == {execute.CUDA_GRAPH_MAX_STEPS}
    assert report["validated"] and not report["cache_hit"]
    assert report["requested_steps"] == 1025
    assert report["calibration_steps"] == execute.CUDA_GRAPH_MAX_STEPS
    assert report["rejected"] == {
        "012/32x8": "unsupported layout",
        "120/32x8": "Complete state differs from baseline",
        "201/32x8": "Nonfinite calibration state",
    }
    assert report["samples_s"] == {
        "012/64x4": [1.0] * 6,
        "120/64x4": [0.5] * 6,
        "201/64x4": [1.0] * 6,
    }
    orders = report["timing_orders"]
    assert orders[3:] == [order[::-1] for order in orders[:3][::-1]]
    assert json.loads(c.path.read_text()) == report
    np.testing.assert_array_equal(c.state[0], [1, 2, 3])
    calls = len(c.calls)
    cached = select(c)
    assert tuning.tuning_report(cached)["cache_hit"]
    assert cached.config.cuda_storage_axes == (1, 2, 0)
    assert len(c.calls) == calls
    select(c, "refresh")
    assert len(c.calls) > calls


@pytest.mark.parametrize("contents", ["broken json", "{}", '{"key":"wrong"}'])
def test_invalid_cache_is_recalibrated(calibration, contents):
    calibration.path.write_text(contents)
    report = tuning.tuning_report(select(calibration))
    assert report["validated"] and not report["cache_hit"]
    assert report["choice"]["axes"] == [1, 2, 0]


@pytest.mark.parametrize("failure", ["compile", "nonfinite"])
def test_invalid_baseline_cannot_produce_cache(calibration, failure):
    calibration.failures["012/64x4"] = failure
    if failure == "compile":
        with pytest.raises(ValueError, match="unsupported layout"):
            select(calibration)
    else:
        selected = select(calibration)
        assert selected is calibration.program
        assert tuning.tuning_report(selected)["validated"] is False
    assert not calibration.path.exists()


@pytest.mark.parametrize("safe_calls", [0, 2, 7])
def test_headroom_loss_aborts_without_persisting(calibration, monkeypatch, safe_calls):
    answers = iter([True] * safe_calls + [False])
    monkeypatch.setattr(tuning, "_safe_headroom", lambda: next(answers))
    selected = select(calibration)
    assert selected is calibration.program
    assert "headroom" in tuning.tuning_report(selected)["reason"]
    assert not calibration.path.exists()


def test_unavailable_telemetry_skips_calibration(calibration, monkeypatch):
    monkeypatch.setattr(tuning, "_gpu_status", lambda: None)
    selected = select(calibration)
    assert selected is calibration.program
    assert "telemetry unavailable" in tuning.tuning_report(selected)["reason"]
    assert not calibration.builds


@pytest.mark.parametrize("winner", ["120/mstep", "120/64x4"])
def test_auto_calibration_selects_and_caches_the_measured_schedule(calibration, winner):
    c = calibration
    c.program = replace(
        c.program, config=replace(c.program.config, cuda_schedule="auto")
    )
    c.durations[winner] = 0.25
    selected = select(c)
    report = tuning.tuning_report(selected)
    assert len(report["samples_s"]) == 9
    assert selected.config.cuda_schedule == (
        "mstep" if winner.endswith("mstep") else "streamed"
    )
    assert selected.config.cuda_storage_axes == (1, 2, 0)
    assert select(c).config == selected.config


@pytest.mark.parametrize(
    "failure", ["unavailable", "different", "nonfinite", "fallback"]
)
def test_invalid_multistep_candidate_cannot_win(calibration, failure):
    c = calibration
    c.program = replace(
        c.program, config=replace(c.program.config, cuda_schedule="auto")
    )
    c.failures = {f"{axes}/mstep": failure for axes in ("012", "120", "201")}
    selected = select(c)
    report = tuning.tuning_report(selected)
    assert selected.config.cuda_schedule == "streamed"
    assert len(report["rejected"]) == 3
    assert len(report["samples_s"]) == 6


@pytest.mark.parametrize("limitation", ["capacity", "missing_target"])
def test_multistep_search_respects_capacity_and_extension_support(
    calibration, monkeypatch, limitation
):
    c = calibration
    c.program = replace(
        c.program,
        config=replace(
            c.program.config,
            cuda_schedule="auto",
            cuda_memory_policy="capacity" if limitation == "capacity" else "auto",
        ),
    )
    if limitation == "missing_target":
        monkeypatch.setattr(
            backend, "cuda_backend_status", lambda **kw: SimpleNamespace(targets=())
        )
    selected = select(c)
    assert selected.config.cuda_schedule == "streamed"
    assert len(tuning.tuning_report(selected)["samples_s"]) == 6
