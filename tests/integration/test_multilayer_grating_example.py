"""Physical normalization and independent-layer contracts for issue #240."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.optimization.multilayer_grating import (
    Config,
    GratingProblem,
    gaussian_overlap_power,
    input_mode_diagnostics,
)


def test_gaussian_overlap_power_and_direction():
    impedance = 260.0
    area = 1e-14
    target = jnp.ones(100) * np.sqrt(2 * impedance / (100 * area))
    fields = jnp.zeros((6, 1, 100), dtype=jnp.complex64)
    fields = fields.at[1].set(target).at[3].set(-target / impedance)
    np.testing.assert_allclose(
        gaussian_overlap_power(fields, target, area, impedance), 1, rtol=2e-6
    )
    np.testing.assert_allclose(
        gaussian_overlap_power(3j * fields, target, area, impedance), 9, rtol=2e-6
    )
    downward = fields.at[3].multiply(-1)
    np.testing.assert_allclose(
        gaussian_overlap_power(downward, target, area, impedance), 0, atol=1e-12
    )


def test_layer_filter_independence_and_symmetry():
    problem = object.__new__(GratingProblem)
    problem.config = Config()
    problem.n = 12
    density = jnp.asarray(
        np.random.default_rng(42).uniform(0.2, 0.8, (2, 12, 12)), dtype=jnp.float32
    )
    physical = problem.physical_density(density, 4.0)
    changed = problem.physical_density(density.at[0].add(0.05), 4.0)
    np.testing.assert_array_equal(physical[1], changed[1])
    assert not np.allclose(physical[0], changed[0])
    np.testing.assert_allclose(physical, physical[:, ::-1], atol=1e-6)
    assert float(physical.min()) >= 0 and float(physical.max()) <= 1
    gradient = jax.grad(lambda d: problem.physical_density(d, 4.0)[1].sum())(density)
    np.testing.assert_array_equal(gradient[0], 0)
    assert float(jnp.linalg.norm(gradient[1])) > 0


@pytest.mark.parametrize(
    "kwargs", [{"dx": 0}, {"chunk_steps": 0}, {"betas": ()}, {"filter_radii": (0.1,)}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_material_mapping_matches_native_empty_and_filled_layers():
    # A small 3D plan checks the actual staggered native raster supports; no FDTD
    # execution is needed to detect a missing layer or accidental fixed-port edit.
    problem = GratingProblem(Config(dx=0.1, aperture=1.2, run_time_fs=20))
    # Monitor coordinates are already in Beamz's translated frame. The public
    # plane is centered on zero, and the target must peak over the aperture.
    np.testing.assert_allclose(np.mean(problem.fiber_x_um), 0, atol=1e-6)
    np.testing.assert_allclose(np.mean(problem.fiber_y_um), 0, atol=1e-6)
    target = np.asarray(problem.gaussian).reshape(len(problem.fiber_y_um), -1)
    np.testing.assert_allclose(target, target[::-1], rtol=1e-6)
    iy, ix = np.unravel_index(target.argmax(), target.shape)
    assert abs(problem.fiber_y_um[iy]) <= problem.config.dx
    assert (
        abs(problem.fiber_x_um[ix] - (problem.x0 + problem.x1) / 2) <= problem.config.dx
    )
    empty = jnp.zeros((2, problem.n, problem.n))
    base = problem.coefficients(empty)
    for i, axis in enumerate("xyz"):
        name = f"e_permittivity_{axis}"
        np.testing.assert_array_equal(getattr(base, name), getattr(problem.base, name))
        for layer in range(2):
            filled = problem.coefficients(empty.at[layer].set(1))
            difference = np.asarray(getattr(filled, name) - getattr(base, name))
            np.testing.assert_allclose(
                difference, problem.layer_delta[layer][i], atol=1e-6
            )
            assert np.any(difference > 0)
            # Native interface smoothing can redistribute tensor components at
            # the guide junction, so individual deltas need not be positive.
            assert np.all(np.asarray(getattr(filled, name)) > 0)
            source_index = round(
                (problem.source_x + problem.domain[0] / 2) / problem.config.dx
            )
            np.testing.assert_array_equal(difference[:, :, : source_index + 1], 0)


def test_input_mode_diagnostics_detect_polarization_and_parity():
    spec = SimpleNamespace(
        freq_count=1,
        sample_region=SimpleNamespace(
            axis_interval=lambda axis: SimpleNamespace(start=0, stop=4)
        ),
    )
    fields = np.zeros((6, 1, 8), dtype=complex)
    fields[1] = 1
    checks = input_mode_diagnostics(fields, spec)
    assert checks["te_fraction"] == 1
    assert checks["ey_symmetry_error"] == 0
    fields[1] = np.tile([-1, -1, 1, 1], 2)
    assert input_mode_diagnostics(fields, spec)["ey_symmetry_error"] == 2
    fields[1] = 0
    fields[2] = 1
    assert input_mode_diagnostics(fields, spec)["te_fraction"] == 0
