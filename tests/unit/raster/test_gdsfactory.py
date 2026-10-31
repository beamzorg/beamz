from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

import beamz
import beamz.design.raster as raster
from beamz.design import GridSpec, import_component
from beamz.design.gdsfactory import Settings, prepare
from beamz.design.raster.importers import from_gdsfactory
from beamz.design.raster.importers._beamz import from_beamz
from beamz.devices.sources.time import GaussianPulse

pytestmark = pytest.mark.filterwarnings(
    "ignore:Exception ignored in.*gdsfactory/generic_tech/layer_views.yaml:pytest.PytestUnraisableExceptionWarning"
)


class FakeComponent:
    def get_polygons_points(self, by):
        assert by == "tuple"
        return {(1, 0): [np.array([[0, 0], [2, 0], [2, 1], [0, 1]])]}


@dataclass
class FakeLayer:
    layer: tuple[int, int] = (1, 0)
    zmin: float = 0.0
    thickness: float = 0.22
    material: str = "si"
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    z_to_bias: object | None = None
    mesh_order: int = 0
    bias: float = 0.0


@dataclass
class FakeStack:
    layers: dict


def test_material_defaults_require_explicit_opt_in():
    stack = FakeStack({"core": FakeLayer()})
    with pytest.raises(ValueError, match="material_map"):
        from_gdsfactory(FakeComponent(), stack)

    scene = from_gdsfactory(
        FakeComponent(), stack, material_map={"si": beamz.Material(permittivity=12)}
    )
    assert scene.materials[1].epsilon_r[:3] == (12.0, 12.0, 12.0)
    raster.compile_scene(scene)


def test_callable_z_bias_lowers_to_supported_tapered_segments():
    stack = FakeStack(
        {
            "core": FakeLayer(
                z_to_bias=lambda z: 0.05 * z * z,
            )
        }
    )
    scene = from_gdsfactory(
        FakeComponent(), stack, material_map={"si": raster.Material(12)}
    )

    assert len(scene.objects) > 1
    assert all(
        isinstance(
            obj.geometry,
            (raster.ExtrudedPolygon, raster.TaperedExtrudedPolygon),
        )
        for obj in scene.objects
    )
    raster.compile_scene(scene)

    segments = sorted(scene.objects, key=lambda object_: object_.geometry.z_min)
    for segment in segments:
        geometry = segment.geometry
        z0 = geometry.z_min / (0.22e-6)
        z1 = geometry.z_max / (0.22e-6)
        low_bias = -min(x for x, _ in geometry.polygon.exterior)
        high_bias = low_bias
        if isinstance(geometry, raster.TaperedExtrudedPolygon):
            high_bias -= (geometry.z_max - geometry.z_min) * np.tan(
                np.radians(geometry.sidewall_angle_degrees)
            )
        np.testing.assert_allclose(low_bias, 0.05 * z0 * z0 * 1e-6, atol=1e-9)
        np.testing.assert_allclose(high_bias, 0.05 * z1 * z1 * 1e-6, atol=1e-9)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_real_gdsfactory_active_pdk_import_and_rasterization():
    gf = pytest.importorskip("gdsfactory")
    # GDSFactory 8 emits Pydantic-v2 migration warnings while constructing its
    # generic PDK; version 9 requires that PDK to be activated explicitly.
    # Both are upstream lifecycle details, not importer output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            pdk = gf.get_active_pdk()
        except ValueError:
            from gdsfactory.gpdk import PDK

            PDK.activate()
            pdk = gf.get_active_pdk()
        component = gf.components.straight(length=2.0, width=0.5)
        layer_stack = pdk.layer_stack
    material_names = sorted(
        {
            str(level.material)
            for level in layer_stack.layers.values()
            if getattr(level, "material", None)
        }
    )
    material_map = {
        name: raster.Material(2.0 + index) for index, name in enumerate(material_names)
    }

    scene = from_gdsfactory(component, layer_stack, material_map=material_map)
    grid = raster.Grid.uniform(
        (-1e-6, -1e-6, -1e-6),
        (3e-6, 1e-6, 2e-6),
        (8, 4, 3),
    )
    first = scene.rasterize(grid, options=raster.RasterOptions(quality="fast"))
    second = scene.rasterize(grid, options=raster.RasterOptions(quality="fast"))

    assert scene.objects
    assert float(first.tensors["epsilon"].max()) > 1.0
    np.testing.assert_array_equal(first.tensors["epsilon"], second.tensors["epsilon"])


def _stack(**levels):
    from types import SimpleNamespace

    return FakeStack(
        {
            name: SimpleNamespace(**(vars(FakeLayer()) | fields))
            for name, fields in levels.items()
        }
    )


@pytest.fixture
def layout():
    gf = pytest.importorskip("gdsfactory")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            gf.get_active_pdk()
        except ValueError:
            gf.gpdk.PDK.activate()
    c = gf.Component()
    for (x0, y0, x1, y1), layer in [
        ((0, 0, 4, 2), (1, 0)),
        ((0, 0, 4, 0.6), (2, 0)),
        ((0, 1.4, 4, 2), (2, 0)),
        ((-2, 0.5, -1, 1.5), (3, 0)),
    ]:
        c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=layer)
    c.add_port("o1", center=(0, 1), width=0.8, orientation=180, layer=(1, 0))
    return c


def _sample(imported, point):

    center = np.asarray(point) * 1e-6 - imported.source_origin
    grid = raster.Grid.uniform(tuple(center - 5e-10), tuple(center + 5e-10), (1, 1, 1))
    result = from_beamz(imported.design).rasterize(
        grid, options=raster.RasterOptions(smoothing="volume", quality="reference")
    )
    return float(result.tensors["epsilon"].item())


def test_stack_preparation_preserves_etch_fills_and_cached_source(layout, monkeypatch):
    from gdsfactory.technology import LogicalLayer

    calls = []

    def profile(z):
        calls.append(z)
        return 0.05 * z * z

    process = _stack(
        core=dict(
            layer=LogicalLayer(layer=(1, 0)) - LogicalLayer(layer=(2, 0)),
            z_to_bias=profile,
            thickness=0.3,
            mesh_order=1,
        ),
        slab=dict(thickness=0.09, mesh_order=3),
        box=dict(
            layer=(99, 0),
            zmin=-2,
            thickness=2,
            material="sio2",
            background=True,
            background_exclude_layers=[(3, 0)],
            mesh_order=10,
        ),
    )
    materials = {"si": beamz.Material(12), "sio2": beamz.Material(2.25)}
    imported = import_component(layout, layer_stack=process, material_map=materials)
    original = imported.design
    settings = Settings(xy_padding=3e-6, z_padding=1e-6)
    direct = prepare(
        layout, layer_stack=process, material_map=materials, settings=settings
    )
    count = len(calls)
    monkeypatch.setattr(
        LogicalLayer, "get_shapes", lambda *_: pytest.fail("reread source")
    )
    process.layers["core"].zmin = 99
    cached = prepare(imported, settings=settings)
    assert from_beamz(cached.design).to_json() == from_beamz(direct.design).to_json()
    assert len(calls) == count and imported.design is original
    again = prepare(cached.imported, settings=settings)
    assert from_beamz(again.design).to_json() == from_beamz(cached.design).to_json()
    for point, expected in [
        ((2, 0.3, 0.2), 1),
        ((2, 0.3, 0.04), 12),
        ((-2.5, 1, 0.2), 12),
        ((-2.5, 1, -0.5), 2.25),
        ((-1.5, 1, -0.5), 1),
    ]:
        assert _sample(cached.imported, point) == pytest.approx(expected)


@pytest.mark.parametrize("reference", [0, 0.37, 0.5, 1])
@pytest.mark.parametrize("angle", [-20, 20])
def test_downward_taper_keeps_reference_under_crop(layout, reference, angle):

    process = _stack(
        core=dict(zmin=1, thickness=-1, width_to_z=reference, sidewall_angle=angle)
    )
    imported = import_component(
        layout,
        layer_stack=process,
        material_map={"si": 12},
        bounds=((-0.5e-6, -0.5e-6, 0.1e-6), (4.5e-6, 2.5e-6, 0.9e-6)),
    )
    for z in (0.2, 0.8):
        edge = -(z - (1 - reference)) * np.tan(np.radians(angle))
        assert _sample(imported, (2, edge + 0.02, z)) == pytest.approx(12)
        assert _sample(imported, (2, edge - 0.02, z)) == pytest.approx(1)


@pytest.mark.parametrize("unify", [False, True])
def test_native_and_raster_yee_coefficients_match(layout, unify):

    process = _stack(
        core=dict(sidewall_angle=10, width_to_z=0.37),
        slab=dict(thickness=0.09, material="sio2", mesh_order=2),
    )
    materials = {"si": beamz.Material((12, 11, 10)), "sio2": beamz.Material(2.25)}
    bounds = ((-0.2e-6, -0.2e-6, -0.1e-6), (4.2e-6, 2.2e-6, 0.4e-6))
    imported = import_component(
        layout, layer_stack=process, material_map=materials, bounds=bounds, unify=unify
    )
    source = raster.Grid.uniform(*bounds, (1, 6, 6))
    local = raster.Grid(
        *(
            edges - offset
            for edges, offset in zip(source.edges, imported.source_origin, strict=True)
        )
    )
    options = raster.RasterOptions(quality="reference", smoothing="volume")
    expected = from_gdsfactory(
        layout, process, material_map=materials, bounds=bounds
    ).rasterize(source, options=options)
    actual = from_beamz(imported.design).rasterize(local, options=options)
    for collection in ("tensors", "yee_tensors"):
        for name, values in getattr(expected, collection).items():
            np.testing.assert_allclose(
                getattr(actual, collection)[name], values, atol=3e-5, rtol=2e-5
            )


def test_prepared_stack_launches_finite_fields(layout):
    settings = Settings(
        xy_padding=0.7e-6,
        z_padding=0.5e-6,
        pml_thickness=0.2e-6,
        grid_spec=GridSpec.uniform(0.1e-6),
        run_time=2e-15,
        wavelength_points=2,
    )
    setup = prepare(
        layout, layer_stack=_stack(core={}), material_map={"si": 12}, settings=settings
    )
    sim = setup.simulation_for("o1")
    pulse = GaussianPulse(settings.center_frequency, settings.frequency_width, offset=0)
    sim = sim.updated_copy(sources=(sim.sources[0].updated_copy(source_time=pulse),))
    run = sim.advance(num_steps=5, backend="jax", performance=False)
    fields = [
        np.asarray(getattr(run.state, name))
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    ]
    assert all(np.isfinite(field).all() for field in fields)
    assert max(float(np.abs(field).max()) for field in fields) > 0
