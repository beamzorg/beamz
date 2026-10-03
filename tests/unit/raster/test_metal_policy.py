"""Metal point ownership must agree across instantaneous and causal response."""

from dataclasses import replace

import numpy as np
import pytest

import beamz as bz
from beamz.design.raster import Box, Grid, Object, RasterOptions, Scene
from beamz.design.raster.engine import staircase_materials
from beamz.simulation.api import _resolved_raster_options


def metal(epsilon_inf=3):
    return bz.PoleResidue.drude(
        epsilon_inf,
        plasma_frequency=1e16,
        damping=1e14,
        frequency_range=(1e14, 2e15),
    )


def scene_for(medium, host=None):
    return Scene(
        (bz.Material(1) if host is None else host, medium),
        (Object(Box((0.3, -1, -1), (2, 2, 2)), 1),),
    )


def grid_for():
    # Ey x=0.2 differs from its dual-volume center x=0.35.
    return Grid([0, 0.2, 1], [0, 0.5, 1], [0, 0.1])


def sim_for(scene, **kwargs):
    return bz.Simulation(
        scene=scene,
        raster_grid=grid_for(),
        polarization="te",
        time=np.arange(2) * 1e-12,
        **kwargs,
    )


def test_classification_uses_real_optical_permittivity_not_loss_or_epsilon_infinity():
    m = metal()
    lossy_dielectric = bz.Material(4, conductivity=1e6)
    scene = Scene((m, lossy_dielectric, bz.Material(1), bz.Material(0.9)))
    assert staircase_materials(scene, RasterOptions(reference_frequency=1e14)) == (0, 3)
    assert staircase_materials(scene, RasterOptions(reference_frequency=2e15)) == (3,)
    assert staircase_materials(scene, RasterOptions(metal_smoothing="inherit")) == ()


@pytest.mark.parametrize("frequency", [0, -1, float("nan"), float("inf")])
def test_invalid_reference_frequency(frequency):
    with pytest.raises(ValueError, match="positive Hz"):
        RasterOptions(reference_frequency=frequency)


def test_metal_instantaneous_and_pole_ownership_match_at_actual_yee_coordinates():
    sim = sim_for(
        scene_for(metal()), raster_options=RasterOptions(reference_frequency=1e14)
    )
    mg = sim.material_grid
    assert not mg.dispersion_interfaces
    weights = mg.dispersion[0][1]
    # Each Ex midpoint and Ey edge is tested against x>=0.3 independently.
    for component, expected in [("Ex", [0, 1]), ("Ey", [0, 0, 1])]:
        fraction = np.asarray(weights[component])
        np.testing.assert_array_equal(
            fraction, np.broadcast_to(expected, fraction.shape)
        )
        np.testing.assert_array_equal(
            mg.yee_materials["eps_" + component[-1].lower()], 1 + 2 * fraction
        )


def test_identical_static_materials_keep_distinct_causal_ownership():
    host = bz.PoleResidue(3, [(-1e15, 1e14)], frequency_range=(1e14, 2e15))
    sim = sim_for(
        scene_for(metal(), host), raster_options=RasterOptions(reference_frequency=1e14)
    )
    mg = sim.material_grid
    assert not mg.dispersion_interfaces
    for component in ("Ex", "Ey"):
        host_weight, metal_weight = [
            np.asarray(supports[component]) for _, supports in mg.dispersion
        ]
        np.testing.assert_array_equal(host_weight + metal_weight, 1)
        assert set(np.unique(metal_weight)) == {0, 1}
        expected = [0, 1] if component == "Ex" else [0, 0, 1]
        np.testing.assert_array_equal(
            metal_weight, np.broadcast_to(expected, metal_weight.shape)
        )


def test_explicit_inherit_retains_polarized_metal_interfaces():
    scene = scene_for(metal())
    sim = sim_for(scene, raster_options=RasterOptions(metal_smoothing="inherit"))
    assert sim.material_grid.dispersion_interfaces
    assert any(
        np.any((w > 0) & (w < 1)) for w in sim.material_grid.dispersion[0][1].values()
    )


def test_dielectric_default_is_unchanged():
    dielectric = bz.PoleResidue(3, [(-1e15, 1e14)], frequency_range=(1e14, 2e15))
    scene = scene_for(dielectric)
    default = sim_for(scene).material_grid
    old = sim_for(
        scene, raster_options=RasterOptions(metal_smoothing="inherit")
    ).material_grid
    assert default.dispersion_interfaces
    for key in default.yee_materials:
        np.testing.assert_array_equal(
            default.yee_materials[key], old.yee_materials[key]
        )
    for (_, a), (_, b) in zip(default.dispersion, old.dispersion, strict=True):
        for component in a:
            np.testing.assert_array_equal(a[component], b[component])


def test_painter_order_can_remove_metal_completely():
    scene = scene_for(metal())
    painted = replace(
        scene,
        objects=scene.objects + (replace(scene.objects[0], material_id=0, priority=1),),
    )
    mg = sim_for(
        painted, raster_options=RasterOptions(reference_frequency=1e14)
    ).material_grid
    assert not mg.dispersion_interfaces
    for weight in mg.dispersion[0][1].values():
        np.testing.assert_array_equal(weight, 0)


def test_cache_separates_policy_frequency_and_dispersion_with_same_static_epsilon(
    tmp_path,
):
    scene = scene_for(metal())
    low = RasterOptions(reference_frequency=1e14)
    high = replace(low, reference_frequency=2e15)
    a = scene.rasterize(grid_for(), options=low, cache_directory=tmp_path)
    b = scene.rasterize(grid_for(), options=high, cache_directory=tmp_path)
    assert not a.cache_hit and not b.cache_hit
    assert not np.array_equal(a.yee_tensors["epsilon_ey"], b.yee_tensors["epsilon_ey"])
    dielectric = bz.PoleResidue(3, [(-1e15, 1e14)], frequency_range=(1e14, 2e15))
    other = scene_for(dielectric).rasterize(
        grid_for(), options=low, cache_directory=tmp_path
    )
    assert not other.cache_hit
    np.testing.assert_array_equal(
        other.yee_tensors["epsilon_ey"], b.yee_tensors["epsilon_ey"]
    )
    assert scene.rasterize(grid_for(), options=low, cache_directory=tmp_path).cache_hit


def test_source_frequency_drives_scene_classification_and_explicit_override_wins():
    source = bz.PlaneWaveSource(
        center=(0.1, 0.5, 0),
        size=(0, 1, 0),
        direction="+x",
        source_time=bz.GaussianPulse(1e14, 1e13),
    )
    default = sim_for(scene_for(metal()), sources=[source])
    assert not default.material_grid.dispersion_interfaces
    explicit = sim_for(
        scene_for(metal()),
        sources=[source],
        raster_options=RasterOptions(reference_frequency=2e15),
    )
    assert explicit.material_grid.dispersion_interfaces
    resolved = _resolved_raster_options(None, [source])
    assert resolved.reference_frequency == 1e14
    assert _resolved_raster_options(None, []).reference_frequency is None


def test_design_path_and_source_change_invalidate_material_cache():
    design = bz.Design(
        width=1,
        height=1,
        structures=(
            bz.Rectangle(position=(0.325, 0), width=0.675, height=1, material=metal()),
        ),
    )
    source = bz.PlaneWaveSource(
        center=(0.1, 0.5, 0),
        size=(0, 1, 0),
        direction="+x",
        source_time=bz.GaussianPulse(1e14, 1e13),
    )
    low = bz.Simulation(
        design,
        sources=[source],
        resolution=0.1,
        polarization="te",
        time=np.arange(2) * 1e-12,
    )
    high_source = replace(source, source_time=bz.GaussianPulse(2e15, 1e13))
    high = low.updated_copy(sources=[high_source])
    assert low._material_grid_token() != high._material_grid_token()
    assert not low._material_grid().dispersion_interfaces
    assert high._material_grid().dispersion_interfaces
    direct = design.rasterize(0.1, polarization="te", reference_frequency=1e14)
    for key in direct.yee_materials:
        np.testing.assert_array_equal(
            direct.yee_materials[key], low._material_grid().yee_materials[key]
        )


def test_dielectric_interfaces_remain_polarized_in_scene_containing_metal():
    dielectric = bz.PoleResidue(4, [(-1e15, 1e14)], frequency_range=(1e14, 2e15))
    m = metal()
    scene = Scene(
        (bz.Material(1), dielectric, m),
        (
            Object(Box((-1, -1, -1), (0.225, 2, 2)), 1),
            Object(Box((0.625, -1, -1), (2, 2, 2)), 2, id=1),
        ),
    )
    sim = bz.Simulation(
        scene=scene,
        raster_grid=Grid.uniform((0, 0, 0), (1, 1, 0.1), (10, 10, 1)),
        polarization="te",
        time=np.arange(2) * 1e-12,
        raster_options=RasterOptions(reference_frequency=1e14),
    )
    assert sim.material_grid.dispersion_interfaces
    for interface in sim.material_grid.dispersion_interfaces:
        assert m not in interface.materials
        assert dielectric in interface.materials


def test_native_rejects_invalid_material_ids():
    from beamz.design.raster import _native

    compiled = _native.compile_scene(scene_for(metal()).to_json())
    edges = tuple(e.tolist() for e in grid_for().edges)
    with pytest.raises(ValueError, match="out of range"):
        compiled.rasterize(edges, staircase_materials=[2])
    with pytest.raises(ValueError, match="out of range"):
        compiled.interface_samples(edges, "balanced", "all", [2])


@pytest.mark.parametrize("radius", [0.23, 0.001])
def test_curved_metal_staircase_matches_independent_3d_point_oracle(radius):
    from beamz.design.raster import Sphere

    center = np.array([0.55, 0.5, 0.5])
    scene = Scene(
        (bz.Material(1), metal()), (Object(Sphere(tuple(center), radius), 1),)
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (10, 10, 10))
    sim = bz.Simulation(
        scene=scene,
        raster_grid=grid,
        time=np.arange(2) * 1e-12,
        raster_options=RasterOptions(reference_frequency=1e14),
    )
    assert not sim.material_grid.dispersion_interfaces
    for component, fraction in sim.material_grid.dispersion[0][1].items():
        axis = "xyz".index(component[-1].lower())
        coords = [
            0.5 * (e[:-1] + e[1:]) if j == axis else e for j, e in enumerate(grid.edges)
        ]
        xyz = np.meshgrid(*coords, indexing="ij")
        inside = sum((v - c) ** 2 for v, c in zip(xyz, center, strict=True)) < radius**2
        np.testing.assert_array_equal(fraction, inside.transpose(2, 1, 0))


@pytest.mark.parametrize("origin", [-1.0, 0.0, 1.0])
def test_metal_clipped_at_domain_faces_does_not_create_air_seam(origin):
    # All six terminal faces represent the interior domain limit. In particular,
    # periodic joining must not turn a homogeneous metal into a 50/50 air mix.
    grid = Grid.uniform((origin,) * 3, (origin + 1,) * 3, (3, 3, 3))
    scene = Scene(
        (bz.Material(1), metal()), (Object(Box((origin,) * 3, (origin + 1,) * 3), 1),)
    )
    sim = bz.Simulation(
        scene=scene,
        raster_grid=grid,
        time=np.arange(2) * 1e-12,
        raster_options=RasterOptions(reference_frequency=1e14),
        boundaries=[bz.Periodic(axes=("x", "y", "z"))],
    )
    mg = sim.material_grid
    for fraction in mg.dispersion[0][1].values():
        np.testing.assert_array_equal(fraction, 1)
    for component in "xyz":
        np.testing.assert_array_equal(mg.yee_materials["eps_" + component], 3)
