"""Physical-workload and validation contracts for the nanoparticle runner."""

import math

import numpy as np
import pytest

from scripts.benchmark_cuda_mie import (
    LIGHT_SPEED,
    build_simulation,
    case_spec,
    compare_states,
    failure_status,
    parser,
)


def arguments(*extra):
    return parser().parse_args(["--output", "/tmp/unused-mie.json", *extra])


def test_fixed_physical_scene_refines_space_and_time_together():
    args = arguments()
    coarse, fine = (case_spec(args, dx) for dx in (5, 1))
    assert coarse["shape_zyx"] == [136] * 3
    assert fine["shape_zyx"] == [680] * 3
    assert fine["material_cells"] == 125 * coarse["material_cells"]
    assert fine["pml_cells"] == coarse["pml_cells"] == 12
    assert coarse["realized_pml_nm"] == 60
    assert fine["realized_pml_nm"] == 12
    assert fine["timesteps"] == pytest.approx(5 * coarse["timesteps"], abs=4)
    for spec in (coarse, fine):
        assert spec["actual_duration_fs"] >= args.duration_fs
        assert spec["actual_duration_fs"] < args.duration_fs + spec["dt_s"] * 1e15
        assert spec["dt_s"] == pytest.approx(
            0.95 * spec["resolution_nm"] * 1e-9 / (LIGHT_SPEED * math.sqrt(3))
        )


def test_fixed_cell_pml_is_explicit_and_changes_physical_thickness():
    args = arguments("--pml-cells", "12")
    coarse, fine = (case_spec(args, dx) for dx in (5, 1))
    assert coarse["realized_pml_nm"] == 60
    assert fine["realized_pml_nm"] == 12
    assert coarse["pml_policy"] == fine["pml_policy"] == "fixed_cells"


@pytest.mark.parametrize("cells", [0, 16, 80])
def test_benchmark_rejects_other_cpml_cell_counts(cells):
    with pytest.raises(SystemExit):
        arguments("--pml-cells", str(cells))


@pytest.mark.parametrize("resolution", [5, 4, 2.5, 2, 1])
def test_every_resolution_uses_twelve_cpml_cells(resolution):
    spec = case_spec(arguments(), resolution)
    assert spec["pml_cells"] == 12
    assert spec["realized_pml_nm"] == 12 * resolution


@pytest.mark.parametrize(
    "extra,resolution,message",
    [
        (["--courant", "1.1"], 5, "courant"),
        (["--radius-nm", "240"], 5, "more space"),
        (["--particle-index", "0"], 5, "positive"),
        (["--steps", "1"], 5, "two timesteps"),
        ([], 3, "integer multiple"),
        (["--wavelength-nm", "nan"], 5, "finite"),
    ],
)
def test_invalid_or_overlapping_physical_geometry_rejected(extra, resolution, message):
    with pytest.raises(ValueError, match=message):
        case_spec(arguments(*extra), resolution)


def test_fixed_steps_is_labelled_as_different_physical_work():
    args = arguments("--steps", "32")
    coarse, fine = (case_spec(args, dx) for dx in (5, 1))
    assert coarse["timesteps"] == fine["timesteps"] == 32
    assert coarse["duration_policy"] == "fixed_steps"
    assert coarse["actual_duration_fs"] == pytest.approx(5 * fine["actual_duration_fs"])


def test_sphere_is_localized_and_monitors_enclose_it():
    args = arguments("--cube-nm", "800", "--resolutions-nm", "10", "--steps", "32")
    spec = case_spec(args, 10)
    sim = build_simulation(args, spec)
    eps = np.asarray(sim._material_grid(progress=False).permittivity)
    assert eps.shape == (80,) * 3
    assert set(np.unique(eps)) == {1, 4}
    assert eps[40, 40, 40] == 4
    assert eps[0, 40, 40] == 1
    volume_nm3 = np.count_nonzero(eps == 4) * 10**3
    assert volume_nm3 == pytest.approx(4 / 3 * np.pi * 100**3, rel=0.02)
    assert len(sim.sources) == 1
    assert len(sim.monitors) == 6
    assert all(len(m.fields) == 6 for m in sim.monitors)


def test_parity_gate_checks_small_auxiliary_arrays_and_integer_clocks():
    state = {
        "large_field": np.ones(10, np.float32),
        "tiny_psi": np.array([1e-15], np.float32),
        "clock": np.array(32, np.int32),
    }
    copy = {k: v.copy() for k, v in state.items()}
    assert compare_states(state, copy)["passed"]
    copy["tiny_psi"] *= 2
    with pytest.raises(ValueError, match="parity failed"):
        compare_states(state, copy)
    copy = {k: v.copy() for k, v in state.items()}
    copy["clock"] += 1
    with pytest.raises(ValueError, match="integer/clock"):
        compare_states(state, copy)


def test_oom_is_distinguished_from_other_errors():
    assert failure_status("RESOURCE_EXHAUSTED: allocating device buffers") == "oom"
    assert failure_status("CUDA_ERROR_OUT_OF_MEMORY") == "oom"
    assert failure_status("unsupported source") == "error"
