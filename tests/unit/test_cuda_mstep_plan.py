"""Dependency proofs for compact two-step CUDA intermediate storage."""

from types import SimpleNamespace

import numpy as np

from beamz.simulation.cuda.native.mstep import build_mstep_plan


def _plan(shape=(37, 65, 97), groups=(None,) * 9, indices=None):
    # Canonical Yee extents in native H, E order.
    from beamz.lattice import component_shapes

    supports = component_shapes(shape)
    shapes = tuple(supports[c] for c in ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez"))
    cpml = SimpleNamespace(h_terms=(), e_terms=())
    return build_mstep_plan(shapes, cpml, groups, indices), shapes


def test_edge_dependencies_and_scratch_cover_every_read():
    plan, _ = _plan()
    z, y, x = plan.shape
    first = set(plan.first_cells.tolist())
    for row in plan.edge_cells:
        zz, rest = divmod(int(row), y * x)
        yy, xx = divmod(rest, x)
        # H2's E1 neighbours and E1's H1 cross-axis neighbours.
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if 0 <= zz + dz < z and 0 <= yy + dy < y and 0 <= xx + dx < x:
                        assert ((zz + dz) * y + yy + dy) * x + xx + dx in first
    assert plan.core_cells > 0
    large, _ = _plan((65, 129, 193))
    assert large.scratch_cells < np.prod(large.shape)
    assert np.unique(plan.edge_cells).size == plan.edge_cells.size
    zz, rest = np.divmod(plan.first_cells, y * x)
    yy, xx = np.divmod(rest, x)
    spare = plan.scratch_cells // 128 - 1
    assert np.all(plan.slots[zz, yy // 8, xx // 16] < spare)


def test_sources_exclude_their_column_and_two_cell_halo():
    source = SimpleNamespace(
        coeffs=np.ones((1, 1, 1, 1)), starts=np.array([[15, 24, 48]])
    )
    plan, _ = _plan(groups=(source,) + (None,) * 8)
    for yy in range(22, 27):
        for xx in range(46, 51):
            tile_y, tile_x = yy // 16, xx // 32
            quadrant = (yy % 16 >= 8) * 2 + (xx % 32 >= 16)
            assert not (plan.core_masks[tile_y, tile_x] & (1 << quadrant))


def test_monitors_preserve_intermediate_core_samples():
    base, shapes = _plan()
    indices = np.full((1, 6, 1, 1), -1, np.int32)
    indices[0, 0, 0, 0] = (16 * shapes[3][1] + 32) * shapes[3][2] + 48
    plan, _ = _plan(indices=indices)
    ny, nx = plan.shape[1:]
    for zz, yy, xx in ((16, 32, 48), (15, 32, 48), (16, 31, 48), (16, 32, 47)):
        row = (zz * ny + yy) * nx + xx
        assert row in plan.first_cells
    assert plan.core_cells == base.core_cells


def test_monitor_planes_deduplicate_components_and_ignore_invalid_indices():
    base, shapes = _plan((31, 41, 57))
    rng = np.random.default_rng(42)
    indices = np.empty((2, 6, 3, 40), np.int32)
    for component, dims in enumerate((*shapes[3:], *shapes[:3])):
        values = rng.integers(-20, np.prod(dims) + 20, (2, 1, 40))
        values[:, :, :2] = [-1, np.prod(dims)]
        indices[:, component] = values  # Repeated frequencies request the same cells.
    plan, _ = _plan((31, 41, 57), indices=indices)
    expected = set(base.first_cells.tolist())
    nz, ny, nx = plan.shape
    for component, dims in enumerate((*shapes[3:], *shapes[:3])):
        for index in indices[:, component].reshape(-1):
            if not 0 <= index < np.prod(dims):
                continue
            z, rest = divmod(int(index), dims[1] * dims[2])
            y, x = divmod(rest, dims[2])
            for dz, dy, dx in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
                zz, yy, xx = z + dz, y + dy, x + dx
                if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                    expected.add((zz * ny + yy) * nx + xx)
    assert set(plan.first_cells.tolist()) == expected


def test_small_case_has_an_explicit_edge_only_plan():
    plan, _ = _plan((3, 4, 5))
    assert plan.core_cells == 0
    assert plan.edge_cells.size == np.prod(plan.shape)


def test_plan_cache_reuses_values_but_invalidates_geometry():
    base, _ = _plan()
    assert _plan()[0] is base
    source = SimpleNamespace(
        coeffs=np.ones((1, 1, 1, 1)), starts=np.array([[15, 24, 48]])
    )
    sourced, _ = _plan(groups=(source,) + (None,) * 8)
    source.starts[0, 2] += 16
    moved, _ = _plan(groups=(source,) + (None,) * 8)
    assert sourced is not moved
    assert not np.array_equal(sourced.core_masks, moved.core_masks)
    assert not base.slots.flags.writeable
