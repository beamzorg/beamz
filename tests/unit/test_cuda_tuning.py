from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.cuda import tuning
from beamz.simulation.model import RunConfig


def test_selection_rejects_noise_and_one_sided_wins():
    samples = {
        "012/64x4": [1.0] * 6,
        "noise": [0.99] * 6,
        "drift": [0.5, 0.5, 0.5, 1.1, 1.1, 1.1],
        "winner": [0.8] * 6,
    }
    assert tuning.choose_winner(samples) == "winner"
    del samples["winner"]
    assert tuning.choose_winner(samples) == "012/64x4"


def test_manual_overrides_are_in_policy(monkeypatch):
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", "012")
    monkeypatch.setenv("BEAMZ_CUDA_AUTOTUNE", "off")
    policy = tuning.tuning_policy_from_env()
    assert policy[0] == "off"
    assert ("BEAMZ_CUDA_STORAGE_AXES", "012") in policy[2]
    monkeypatch.setenv("BEAMZ_CUDA_AUTOTUNE", "invalid")
    with pytest.raises(ValueError, match="AUTOTUNE"):
        tuning.tuning_policy_from_env()


def test_persistent_identity_changes_for_workload_build_and_hardware(
    tmp_path, monkeypatch
):
    @dataclass
    class Signature:
        num_steps: int = 256
        materials: str = "binary"
        cuda_tuning_policy: tuple = ()

    device = SimpleNamespace(
        device_kind="RTX 3090", id=0, client=SimpleNamespace(platform_version="driver1")
    )
    monkeypatch.setattr(tuning, "_implementation_identity", lambda: "build1")
    policy = ("auto", str(tmp_path), ())
    base = tuning._cache_path(Signature(), device, policy)[1]
    assert (
        tuning._cache_path(Signature(cuda_tuning_policy=("refresh",)), device, policy)[
            1
        ]
        == base
    )
    assert tuning._cache_path(Signature(num_steps=128), device, policy)[1] != base
    assert tuning._cache_path(Signature(materials="smooth"), device, policy)[1] != base
    assert (
        tuning._cache_path(Signature(), device, policy, {"power_limit": 400})[1] != base
    )
    monkeypatch.setattr(tuning, "_implementation_identity", lambda: "build2")
    assert tuning._cache_path(Signature(), device, policy)[1] != base


def test_apply_changes_only_layout_and_shell():
    @dataclass
    class Program:
        config: RunConfig
        coefficients: object

    coefficients = object()
    program = Program(
        RunConfig(
            resolution=1,
            dt=1,
            num_steps=256,
            plane_2d="xy",
            is_3d=True,
            cuda_flags=abi.CUDA_DEFAULT_FLAGS,
        ),
        coefficients,
    )
    selected = tuning._apply(program, dict(axes=[1, 2, 0], shell_tile="32x8"))
    assert selected.coefficients is coefficients
    assert selected.config.cuda_storage_axes == (1, 2, 0)
    assert selected.config.cuda_flags == abi.CUDA_DEFAULT_FLAGS | abi.CUDA_SHELL32X8
    assert program.config.cuda_storage_axes == (0, 1, 2)
    with pytest.raises(ValueError):
        tuning._apply(program, dict(axes=[2, 1, 0], shell_tile="32x8"))


def test_nonfinite_state_cannot_be_certified():
    with pytest.raises(FloatingPointError):
        tuning._state_hashes((np.array([np.nan], dtype=np.float32),))


def test_eligibility_handles_packed_lossless_material_tables():
    config = RunConfig(
        resolution=1,
        dt=1,
        num_steps=256,
        plane_2d="xy",
        is_3d=True,
        backend="cuda_streamed",
        cuda_flags=abi.CUDA_DEFAULT_FLAGS,
    )
    term = SimpleNamespace(slab=SimpleNamespace(low=12, high=12))
    program = SimpleNamespace(
        config=config,
        grid=SimpleNamespace(conductivity=np.float32(0)),
        coefficients=SimpleNamespace(
            e_inverse_offdiagonal=np.array([]),
            e_decay_x=np.array([0.1, 0.2]),
            h_decay_x=np.float32(1),
            h_decay_y=np.float32(1),
            h_decay_z=np.float32(1),
        ),
        boundary=SimpleNamespace(
            logical_component_shapes={
                name: (128, 256, 512) for name in ("Ex", "Ey", "Ez")
            },
            cpml=SimpleNamespace(
                enabled=True, h_terms=(term,) * 6, e_terms=(term,) * 6
            ),
        ),
        sources=(),
        monitors=(),
    )
    assert tuning._eligible(program)
    program.config = replace(config, sharding=replace(config.sharding, enabled=True))
    assert not tuning._eligible(program)
    program.config = config
    program.grid.conductivity = np.float32(1)
    assert not tuning._eligible(program)
    program.grid.conductivity = np.float32(0)
    term.slab.high = 11
    assert not tuning._eligible(program)
    term.slab.high = 12
    program.boundary.logical_component_shapes = {
        name: (1024, 1024, 24) for name in ("Ex", "Ey", "Ez")
    }
    assert not tuning._eligible(program)


def test_manual_and_disabled_policies_never_calibrate(monkeypatch):
    monkeypatch.setattr(
        tuning, "_eligible", lambda _: pytest.fail("Eligibility evaluated")
    )
    program = object()
    assert tuning.select_program(program, None, ("off", "", ())) is program
    assert (
        tuning.select_program(
            program, None, ("auto", "", (("BEAMZ_CUDA_STORAGE_AXES", "012"),))
        )
        is program
    )


def test_gpu_headroom_guard(monkeypatch):
    for free, temperature, expected in [
        (4096, 70, True),
        (2048, 70, False),
        (8000, 85, False),
    ]:
        monkeypatch.setattr(
            tuning,
            "_gpu_status",
            lambda free=free, temperature=temperature: dict(
                free_mib=free, temperature_c=temperature
            ),
        )
        assert tuning._safe_headroom() is expected


def test_cache_write_is_atomic_and_readable(tmp_path):
    path = tmp_path / "nested/record.json"
    tuning._write_record(path, {"choice": "012"})
    assert '"012"' in path.read_text()
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((1024, 256, 63), (1, 2, 0)),
        ((1024, 256, 64), (1, 2, 0)),
        ((1024, 256, 65), (1, 2, 0)),
        ((256, 1024, 64), (2, 0, 1)),
        ((128, 256, 512), (0, 1, 2)),
        ((257, 193, 341), (0, 1, 2)),
        ((256, 256, 256), (0, 1, 2)),
        # Moving the longest dimension to x is not universally faster.
        ((1024, 64, 256), (0, 1, 2)),
    ],
)
def test_predictor_regression_shapes(shape, axes):
    report = tuning.predict_layout(shape)
    assert report["choice"] == dict(axes=axes, shell_tile="64x4")
    assert report["logical_shape"] == shape
    assert set(report["candidate_work"]) == {"012", "120", "201"}


@pytest.mark.parametrize("shape", [(24, 256, 512), (128, 256), (128, 256, 512.5)])
def test_predictor_rejects_invalid_geometry(shape):
    with pytest.raises(ValueError):
        tuning.predict_layout(shape)


@pytest.mark.parametrize("device_kind", ["NVIDIA GeForce RTX 3090", "RTX 3090 Ti"])
def test_auto_uses_geometry_without_trials_telemetry_or_persistent_cache(
    monkeypatch, tmp_path, device_kind
):
    @dataclass
    class Program:
        config: RunConfig
        boundary: object

    program = Program(
        RunConfig(
            resolution=1,
            dt=1,
            num_steps=1025,
            plane_2d="xy",
            is_3d=True,
            cuda_flags=abi.CUDA_DEFAULT_FLAGS,
        ),
        SimpleNamespace(
            logical_component_shapes=dict(
                Ex=(1025, 257, 64), Ey=(1025, 256, 65), Ez=(1024, 257, 65)
            )
        ),
    )
    monkeypatch.setattr(tuning, "_eligible", lambda _: True)
    monkeypatch.setattr(
        tuning.jax, "devices", lambda: [SimpleNamespace(device_kind=device_kind)]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Automatic prediction attempted calibration or profile access")

    for name in ("_select_program", "_gpu_status", "_cache_path", "_write_record"):
        monkeypatch.setattr(tuning, name, forbidden)
    selected = tuning.select_program(program, None, ("auto", str(tmp_path), ()))
    if device_kind.endswith("RTX 3090"):
        assert selected.config.cuda_storage_axes == (1, 2, 0)
        assert selected.config.num_steps == 1025
        assert selected.boundary is program.boundary
        report = tuning.tuning_report(selected)
        assert report["selection_method"] == "geometry"
        assert report["calibration_s"] == report["calibration_steps"] == 0
        assert not report["validated"]
        report["choice"]["axes"] = []
        assert tuning.tuning_report(selected)["choice"]["axes"] == [1, 2, 0]
    else:
        assert selected is program
    assert list(tmp_path.iterdir()) == []
