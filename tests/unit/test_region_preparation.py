"""Region preparation must preserve global interpolation and bounded reads."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from beamz._region_array import RegionArray, place_region_array, tiles
from beamz.devices.sources.mode_launch import (
    _centered_transverse_symmetry_axes,
    _material_mirror_symmetric,
)
from beamz.lattice import component_shapes, sample_voxel_grid_at_component_3d
from beamz.simulation.kernels import precompute_e_update_coefficients
from beamz.simulation.region_materials import electric_coefficients, sample_material


@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
def test_sample_matches_global_at_boundaries_and_regions(component):
    grid = np.random.default_rng(8).uniform(1, 12, (9, 7, 11)).astype(np.float32)
    shape = component_shapes(grid.shape)[component]
    recipe = sample_material(grid, component, shape)
    reference = np.asarray(sample_voxel_grid_at_component_3d(grid, component))
    np.testing.assert_array_equal(np.asarray(recipe), reference)
    for region in tiles(shape, np.float32, budget=64):
        np.testing.assert_array_equal(recipe[region], reference[region])
    np.testing.assert_array_equal(recipe[1, :, 2:6], reference[1, :, 2:6])
    np.testing.assert_array_equal(
        recipe[(np.array([0, 3]), np.array([2, 5]), 1)],
        reference[(np.array([0, 3]), np.array([2, 5]), 1)],
    )


@pytest.mark.parametrize("lossy", [False, True])
def test_coefficient_recipe_matches_eager(lossy):
    rng = np.random.default_rng(9)
    eps = rng.uniform(1, 12, (9, 7, 11)).astype(np.float32)
    sigma = (
        rng.uniform(0, 100, eps.shape).astype(np.float32)
        if lossy
        else np.array(0, np.float32)
    )
    shape = component_shapes(eps.shape)["Ex"]
    epsilon = sample_material(eps, "Ex", shape)
    conductivity = sample_material(sigma, "Ex", shape)
    actual = electric_coefficients(shape, conductivity, epsilon, 1e-16)
    expected = precompute_e_update_coefficients(
        shape, np.asarray(conductivity), np.asarray(epsilon), 1e-16, (slice(None),) * 3
    )
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.broadcast_to(np.asarray(a), shape), b, rtol=2e-7)


def test_placement_never_materializes_recipe(monkeypatch):
    monkeypatch.setattr("beamz._region_array.TILE_BYTES", 256)
    shape = (17, 12, 9)
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    reads = []

    def read(region):
        value = source[region]
        reads.append(value.size)
        return value

    recipe = RegionArray(shape, np.float32, read).padded((18, 12, 10), fill=1)

    def no_array(*args, **kwargs):
        raise AssertionError("Full materialization")

    monkeypatch.setattr(RegionArray, "__array__", no_array)
    target = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    actual = np.asarray(place_region_array(recipe, target))
    np.testing.assert_array_equal(actual[:17, :, :9], source)
    assert np.all(actual[17] == 1)
    assert np.all(actual[:, :, 9] == 1)
    assert reads and max(reads) * 4 <= 256


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_symmetry_matches_full_volume(axis):
    rng = np.random.default_rng(10)
    a = rng.uniform(1, 12, (8, 10, 12)).astype(np.float32)
    a = (a + np.flip(a, axis=axis)) / 2
    assert _material_mirror_symmetric(a, axis)
    a[0, 0, 0] += 1e-3
    assert not _material_mirror_symmetric(a, axis)


def test_source_symmetry_depends_on_local_material():
    source = SimpleNamespace(axis="x", center=(1.5, 6.0, 4.0))
    material = np.ones((8, 12, 10), dtype=np.float32)
    fields = SimpleNamespace(permittivity=material)
    assert _centered_transverse_symmetry_axes(source, fields, None, 1.0) == (0, 1)
    material[0, 1, 8] = 2.0
    assert _centered_transverse_symmetry_axes(source, fields, None, 1.0) == (0, 1)
    material[0, 1, 1] = 2.0
    assert _centered_transverse_symmetry_axes(source, fields, None, 1.0) == ()


def test_public_partitioned_pipeline_without_recipe_materialization():
    import os
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import jax
import numpy as np
from unittest.mock import patch
from beamz._region_array import RegionArray, SeparableMask
from beamz.simulation.execute import initial_program_state
from tests.unit.test_cuda_sharded_features import mode_simulation, native_cpu_backend, assert_state_close
sim = mode_simulation()
reference = sim.advance(num_steps=12, backend="jax", performance=False).state
with native_cpu_backend():
    for backend in ("jax", "cuda_streamed"):
        for axis in ("x", "z"):
            cfg = dict(axis=axis, num_devices=2, backend="cpu")
            with patch.object(RegionArray, "__array__", side_effect=AssertionError("global materialization")):
                program = sim.compile(num_steps=6, backend=backend, sharding=cfg)
                other = sim.compile(num_steps=5, backend=backend, sharding=cfg)
                assert other.coefficients is program.coefficients
                assert other.grid is program.grid
                assert other.config.num_steps == 5
                assert isinstance(program.grid.eps_x, RegionArray)
                assert isinstance(program.boundary.metallic.ex_mask, SeparableMask)
                initial = initial_program_state(program, t=0, current_step=0)
                assert not initial.ex.sharding.is_fully_replicated
                assert all(isinstance(v, jax.Array) for v in program.coefficients)
                first = sim.advance(num_steps=6, backend=backend, sharding=cfg, performance=False)
                final = sim.advance(state=first.state, num_steps=6, backend=backend, sharding=cfg, performance=False)
            assert_state_close(reference, final.state)
print("public mode source/monitor preparation and continuation passed")
""",
        ],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=2",
        ),
        check=True,
        timeout=180,
    )


def test_boundary_extrusion_copy_only_when_required():
    from beamz.devices._boundary_compile import _extrusion_changes_material

    material = np.ones((9, 11, 13), np.float32)
    edges = [("front", 0, "low", 2), ("right", 2, "high", 2)]
    assert not _extrusion_changes_material(material, edges)
    material[0, 3, 4] = 2
    assert _extrusion_changes_material(material, edges)


def test_region_basic_negative_and_strided_slices():
    value = np.arange(9 * 7 * 11).reshape(9, 7, 11)
    recipe = RegionArray(value.shape, value.dtype, value.__getitem__)
    for key in [
        (slice(None, None, -1), slice(None, None, 2), slice(None)),
        (slice(7, 1, -2), 3, slice(None, None, -2)),
        (slice(1, 0), slice(None), slice(None)),
    ]:
        np.testing.assert_array_equal(recipe[key], value[key])


@pytest.mark.parametrize("scalar", [False, True])
def test_region_physical_conductivity_keeps_magnetic_update_lossless(scalar):
    from types import SimpleNamespace

    from beamz.lattice import build_material_coefficients
    from beamz.simulation.region_materials import build_region_materials

    shape = (3, 4, 5)
    sigma = np.asarray(100, np.float32) if scalar else np.full(shape, 100, np.float32)
    fields = SimpleNamespace(
        permittivity=np.full(shape, 2, np.float32),
        conductivity=sigma,
        permeability=np.ones(shape, np.float32),
        has_pml=True,
        has_cpml=True,
        **{c: np.zeros(s, np.float32) for c, s in component_shapes(shape).items()},
    )
    eager = build_material_coefficients(fields)
    regional = build_region_materials(fields)
    for axis in "xyz":
        np.testing.assert_array_equal(getattr(regional, "sigma_m_h" + axis), 0)
        np.testing.assert_array_equal(getattr(eager, "sigma_m_h" + axis), 0)
        np.testing.assert_allclose(
            np.asarray(getattr(regional, "sig_" + axis)),
            np.asarray(getattr(eager, "sig_" + axis)),
        )
