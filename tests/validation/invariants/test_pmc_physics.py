"""PMC must match the symmetry plane of a doubled Yee domain.

The reference contains no magnetic wall. Comparing all active E/H samples after
several public timesteps tests wall location, signs, staggering and corners.
All grids contain fewer than 1000 material cells.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid, RectilinearGrid
from beamz.lattice import component_axis_offsets_3d

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _mirror_simulations(kind, stretched, cpml, plane="xy"):
    nx, ny, nz = 6, 4, 3
    dx = 0.1 * bz.um
    x = np.arange(nx + 1) * dx
    y = np.arange(ny + 1) * dx
    z = np.arange(nz + 1) * dx
    if stretched:
        x *= 1 + 0.04 * np.arange(nx + 1)
        y *= 1 + 0.03 * np.arange(ny + 1)
        z *= 1 + 0.02 * np.arange(nz + 1)
    # Full-domain midpoint is exactly the PMC plane of the half domain.
    full_x = np.concatenate((x[-1] - x[:0:-1], x[-1] + x))
    boundaries = [bz.PEC()]
    if cpml:
        boundaries.append(
            bz.PML(edges=("top", "bottom"), thickness=dx, formulation="cpml")
        )
    is_3d = kind == "3d"
    polarization = "te" if kind == "te" else "tm"

    def simulation(x_edges, half):
        shape = (nz, ny, len(x_edges) - 1) if is_3d else (ny, len(x_edges) - 1)
        grid = RectilinearGrid(x_edges, y, z if is_3d else np.array([0.0, dx]))
        materials = MaterialGrid(
            np.ones(shape, np.float32),
            np.float32(0),
            np.float32(1),
            dx,
            shape,
            grid=grid,
            polarization=None if is_3d else polarization,
        )
        return bz.Simulation(
            material_grid=materials,
            polarization=polarization,
            plane_2d=plane,
            time=np.arange(9) * (0.15 * dx / bz.LIGHT_SPEED),
            boundaries=[*boundaries, bz.PMC(edges="left")] if half else boundaries,
        )

    return simulation(x, True), simulation(full_x, False)


@pytest.mark.parametrize("kind", ["tm", "te", "3d"])
@pytest.mark.parametrize("stretched", [False, True])
@pytest.mark.parametrize("cpml", [False, True])
def test_pmc_matches_unbounded_mirror_plane(kind, stretched, cpml):
    half, full = _mirror_simulations(kind, stretched, cpml)
    half_state, full_state = half.initial_state(), full.initial_state()
    rng = np.random.default_rng(732)
    half_values, full_values = {}, {}
    nx = half.compile(backend="jax").grid.material_grid.shape[-1]
    masks = half.compile(backend="jax").boundary.metallic
    for component in _COMPONENTS:
        name = component.lower()
        old = getattr(half_state, name)
        if old.shape == (1, 1):
            continue
        values = rng.normal(size=old.shape).astype(np.float32)
        if component.startswith("H"):
            values *= np.sqrt(bz.EPS_0 / bz.MU_0)
        values[np.asarray(getattr(masks, f"{name}_mask"))] = 0
        half_values[name] = jnp.asarray(values)
        parity = -1 if component in ("Ex", "Hy", "Hz") else 1
        on_nodes = component_axis_offsets_3d(component)["x"] == 0
        image = values[..., :0:-1] if on_nodes else values[..., ::-1]
        full_values[name] = jnp.asarray(
            np.concatenate((parity * image, values), axis=-1)
        )
    half_state = half_state._replace(**half_values)
    full_state = full_state._replace(**full_values)
    half_result = half.advance(state=half_state, num_steps=8, backend="jax").state
    full_result = full.advance(state=full_state, num_steps=8, backend="jax").state
    for name in half_values:
        actual = np.asarray(getattr(half_result, name))
        expected = np.asarray(getattr(full_result, name))[..., nx:]
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-7, err_msg=name)


@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_pmc_uses_canonical_face_names_in_every_2d_plane(polarization, plane):
    reference, _ = _mirror_simulations(polarization, False, False)
    rotated, _ = _mirror_simulations(polarization, False, False, plane)
    name = "ez" if polarization == "tm" else "ey"
    state = reference.initial_state()
    values = jnp.ones_like(getattr(state, name))
    state = state._replace(**{name: values})
    expected = reference.advance(state=state, num_steps=2, backend="jax").state
    actual = rotated.advance(state=state, num_steps=2, backend="jax").state
    for component in _COMPONENTS:
        np.testing.assert_array_equal(
            getattr(actual, component.lower()), getattr(expected, component.lower())
        )


@pytest.mark.parametrize("kind", ["tm", "te", "3d"])
def test_all_pmc_faces_preserve_discrete_curl_adjointness(kind):
    """No boundary work: <H, curl E> = <E, curl H> on half-weight wall nodes."""
    from beamz.lattice import (
        build_h_boundary_views_for_e_3d,
        component_shapes,
        curl_e_to_h_3d,
        curl_h_to_e_3d,
    )
    from beamz.simulation.kernels import (
        te_xy_curl_e_to_h_2d,
        te_xy_curl_h_to_e_2d,
        tm_xy_curl_e_to_h_2d,
        tm_xy_curl_h_to_e_2d,
    )

    shape = (3, 4, 5) if kind == "3d" else (4, 5)
    shapes = component_shapes(shape, "te" if kind == "te" else "tm")
    rng = np.random.default_rng(15)
    fields = {
        name: jnp.asarray(rng.normal(size=s), dtype=jnp.float32)
        for name, s in shapes.items()
    }
    edges = frozenset(bz.PMC()._get_edges_for_dimensionality(kind == "3d"))
    if kind == "3d":
        e_names, h_names = _COMPONENTS[:3], _COMPONENTS[3:]
        electric = tuple(fields[name] for name in e_names)
        magnetic = tuple(fields[name] for name in h_names)
        curl_e = curl_e_to_h_3d(*electric, 1.0)
        views = build_h_boundary_views_for_e_3d(*magnetic, magnetic_edges=edges)
        curl_h = curl_h_to_e_3d(*magnetic, 1.0, boundary_views=views)
    elif kind == "tm":
        e_names, h_names = ("Ez",), ("Hx", "Hy")
        curl_e = tm_xy_curl_e_to_h_2d(
            fields["Ez"], 1.0, shapes["Hx"], shapes["Hy"], frozenset()
        )
        curl_h = (
            tm_xy_curl_h_to_e_2d(
                fields["Hx"], fields["Hy"], 1.0, shapes["Ez"], magnetic_edges=edges
            ),
        )
    else:
        e_names, h_names = ("Ex", "Ey"), ("Hz",)
        curl_e = (te_xy_curl_e_to_h_2d(fields["Ex"], fields["Ey"], 1.0, shapes["Hz"]),)
        curl_h = te_xy_curl_h_to_e_2d(
            fields["Hz"],
            1.0,
            shapes["Ex"],
            shapes["Ey"],
            frozenset(),
            magnetic_edges=edges,
        )

    def inner(names, curls):
        total = 0.0
        axes = ("z", "y", "x")[-len(shape) :]
        for name, curl in zip(names, curls, strict=True):
            weights = np.ones(shapes[name])
            offsets = component_axis_offsets_3d(name)
            for axis, coordinate in enumerate(axes):
                if offsets[coordinate] == 0:
                    wall = [slice(None)] * len(shape)
                    wall[axis] = [0, -1]
                    weights[tuple(wall)] *= 0.5
            total += np.sum(weights * np.asarray(fields[name]) * np.asarray(curl))
        return total

    np.testing.assert_allclose(
        inner(h_names, curl_e), inner(e_names, curl_h), rtol=3e-6, atol=3e-6
    )
