"""Short continuation tails must not duplicate immutable dense material banks."""

import jax
import numpy as np
import pytest

from beamz import PEC
from beamz._region_array import RegionArray
from beamz.design import MaterialGrid
from tests.performance.h100_workloads import H100Workload


def _simulation():
    return H100Workload(
        name="dense_preparation_reuse",
        shape_zyx=(8, 10, 12),
        timesteps=64,
        resolution=100e-9,
        pml_cells=2,
        heterogeneous=True,
        cpml=True,
        source=True,
        monitor=True,
    ).build()


def test_dense_tail_reuses_materials_and_matches_unsegmented_execution():
    simulation = _simulation()
    simulation.clear_compiled_cache()
    reference = simulation.advance(num_steps=41, backend="jax", performance=False)
    first = simulation.compile(num_steps=32, backend="jax")
    tail = simulation.compile(num_steps=9, backend="jax")
    assert not isinstance(first.grid.eps_x, RegionArray)
    assert tail.grid is first.grid
    assert tail.coefficients is first.coefficients
    assert tail.boundary is first.boundary
    assert tail.metrics is first.metrics
    assert tail.config.num_steps == 9
    prefix = simulation.advance(num_steps=32, backend="jax", performance=False)
    actual = simulation.advance(
        state=prefix.state, num_steps=9, backend="jax", performance=False
    )
    for expected, observed in zip(
        jax.tree.leaves(reference.state), jax.tree.leaves(actual.state), strict=True
    ):
        np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=1e-10)


def test_reuse_rebuilds_source_and_monitor_plans():
    simulation = _simulation()
    simulation.clear_compiled_cache()
    original = simulation.compile(num_steps=32, backend="jax")
    empty = simulation.updated_copy(sources=(), monitors=()).compile(
        num_steps=9, backend="jax"
    )
    assert empty.coefficients is original.coefficients
    assert empty.sources == ()
    assert empty.monitors == ()
    assert original.sources and original.monitors


@pytest.mark.parametrize("changed", ["dt", "material", "boundary"])
def test_changed_physics_does_not_reuse_material_preparation(changed):
    simulation = _simulation()
    simulation.clear_compiled_cache()
    original = simulation.compile(num_steps=32, backend="jax")
    if changed == "dt":
        updated = simulation.updated_copy(time=simulation.time * 0.9)
    elif changed == "boundary":
        updated = simulation.updated_copy(boundaries=(PEC(edges="all"),))
    else:
        material = simulation.material_grid
        updated = simulation.updated_copy(
            material_grid=MaterialGrid(
                permittivity=np.asarray(material.permittivity) * 1.1,
                conductivity=material.conductivity,
                permeability=material.permeability,
                resolution=material.resolution,
                shape=material.shape,
            )
        )
    other = updated.compile(num_steps=9, backend="jax")
    assert other.coefficients is not original.coefficients
    assert other.grid is not original.grid


def test_direct_yee_permittivity_does_not_build_discarded_voxel_interpolants(
    monkeypatch,
):
    import beamz.lattice as lattice
    from beamz.lattice import component_shapes

    shape = (8, 10, 12)
    supports = component_shapes(shape)
    direct = {"eps_" + a: np.full(supports["E" + a], 2.25, np.float32) for a in "xyz"}
    grid = MaterialGrid(
        np.ones(shape),
        0.0,
        1.0,
        1e-7,
        shape,
        yee_materials=direct,
        smoothing="farjadpour_diagonal",
    )

    def discarded(*args, **kwargs):
        pytest.fail("Already-sampled Yee permittivity must not be reinterpolated")

    monkeypatch.setattr(
        lattice, "sample_voxel_grid_at_e_component_3d_centered", discarded
    )
    import beamz as bz

    program = bz.Simulation(material_grid=grid, time=np.arange(3) * 1e-17).compile(
        backend="jax"
    )
    for a in "xyz":
        np.testing.assert_array_equal(
            getattr(program.grid, "eps_" + a), direct["eps_" + a]
        )


@pytest.mark.parametrize("value,dtype", [(0.0, np.float32), (2 + 3j, np.complex64)])
def test_initial_broadcast_fields_allocate_on_device_without_host_materialization(
    monkeypatch, value, dtype
):
    import jax.numpy as jnp

    from beamz.simulation.model import _copy_initial_field

    lattice = np.broadcast_to(np.asarray(value, dtype=dtype), (5, 7, 9))
    original = jnp.array

    def array(data, *args, **kwargs):
        assert data is not lattice, "Do not materialize a full host field to transfer"
        return original(data, *args, **kwargs)

    monkeypatch.setattr(jnp, "array", array)
    first, second = _copy_initial_field(lattice), _copy_initial_field(lattice)
    np.testing.assert_array_equal(first, np.full(lattice.shape, value, dtype=dtype))
    first = first.at[0, 0, 0].set(9)
    np.testing.assert_array_equal(second, lattice)
    assert np.asarray(first)[0, 0, 0] == 9
