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
