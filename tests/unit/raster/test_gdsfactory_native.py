"""Physical import regressions shared by native Design and raster Scene paths."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gdsfactory")

import gdsfactory as gf
from gdsfactory.technology import LogicalLayer

from beamz import Material, Rectangle
from beamz.design import import_component
from beamz.design.dispersion import PoleResidue
from beamz.design.gdsfactory import Settings, prepare
from beamz.design.raster import Grid, RasterOptions, compile_scene
from beamz.design.raster.importers import from_gdsfactory
from beamz.design.raster.importers._beamz import from_beamz
from examples.gdsfactory_imports.cases import BUILDERS, MATERIALS, UM, Probe
from examples.gdsfactory_imports.run import activate_pdk, check_probes, sample_epsilon

pytestmark = [
    pytest.mark.filterwarnings("ignore:Support for class-based.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:Exception ignored in.*gdsfactory/generic_tech/layer_views.yaml:pytest.PytestUnraisableExceptionWarning"
    ),
]


@pytest.fixture
def component():
    activate_pdk()
    c = gf.Component()
    c.add_polygon([(0, 0), (4, 0), (4, 2), (0, 2)], layer=(1, 0))
    return c


def level(**kwargs):
    return SimpleNamespace(
        **dict(
            layer=LogicalLayer(layer=(1, 0)),
            zmin=0.0,
            thickness=1.0,
            material="si",
            mesh_order=3,
        )
        | kwargs
    )


def stack(**levels):
    return SimpleNamespace(layers=levels)


def import_native(component, process, bounds, **kwargs):
    return import_component(
        component,
        layer_stack=process,
        bounds=np.asarray(bounds) * UM,
        material_map=MATERIALS,
        **kwargs,
    )


def native_probe(imported, xyz):
    return sample_epsilon(
        compile_scene(from_beamz(imported.design)),
        np.asarray(xyz) - np.asarray(imported.source_origin) / UM,
    )


@pytest.mark.parametrize("name", BUILDERS)
@pytest.mark.parametrize("unify", [False, True])
def test_native_and_raster_coefficients_match_for_every_example(name, unify):
    activate_pdk()
    case = BUILDERS[name]()
    bounds = np.asarray(case.bounds_um) * UM
    native = import_component(
        case.component,
        layer_stack=case.stack,
        material_map=MATERIALS,
        bounds=bounds,
        unify=unify,
    )
    direct = from_gdsfactory(
        case.component, case.stack, material_map=MATERIALS, bounds=bounds
    )
    probes = [
        Probe(
            p.name,
            tuple(np.asarray(p.xyz_um) - np.asarray(native.source_origin) / UM),
            p.epsilon,
        )
        for p in case.probes
    ]
    assert all(p["passed"] for p in check_probes(from_beamz(native.design), probes))
    # Compare cell and staggered Yee coefficients, including interface supports.
    # A thin x section keeps this a focused integration check, not a full solve.
    lower, upper = bounds.copy()
    x = (lower[0] + upper[0]) / 2
    lower[0], upper[0] = x - 0.001 * UM, x + 0.001 * UM
    source_grid = Grid.uniform(tuple(lower), tuple(upper), (1, 9, 11))
    local_grid = Grid(
        *(
            edges - offset
            for edges, offset in zip(
                source_grid.edges, native.source_origin, strict=True
            )
        )
    )
    options = RasterOptions(quality="reference", smoothing="volume")
    expected = direct.rasterize(source_grid, options=options)
    actual = from_beamz(native.design).rasterize(local_grid, options=options)
    for collection in ("tensors", "yee_tensors"):
        for name, values in getattr(expected, collection).items():
            np.testing.assert_allclose(
                getattr(actual, collection)[name], values, atol=3e-5, rtol=2e-5
            )


@pytest.mark.parametrize("reference", [0, 0.37, 0.5, 1])
@pytest.mark.parametrize("angle", [-20, 20])
def test_downward_extrusion_preserves_original_reference(component, reference, angle):
    process = stack(
        si=level(zmin=1, thickness=-1, width_to_z=reference, sidewall_angle=angle)
    )
    imported = import_native(component, process, ((-0.5, -0.5, -0.1), (4.5, 2.5, 1.1)))
    for z in (0.2, 0.8):
        # Physical signed thickness: reference is at source z=1-reference.
        growth = (z - (1 - reference)) * np.tan(np.radians(angle))
        edge = -growth
        assert native_probe(imported, (2, edge + 0.02, z)) == pytest.approx(12)
        assert native_probe(imported, (2, edge - 0.02, z)) == pytest.approx(1)


def test_clipping_keeps_source_sidewall_reference(component):
    process = stack(si=level(sidewall_angle=20, width_to_z=0.37))
    imported = import_native(component, process, ((1, 0, 0.5), (3, 2, 0.9)))
    edge = (0.7 - 0.37) * np.tan(np.radians(20))
    assert native_probe(imported, (2, edge - 0.02, 0.7)) == pytest.approx(1)
    assert native_probe(imported, (2, edge + 0.02, 0.7)) == pytest.approx(12)
    assert imported.design.depth == pytest.approx(0.4 * UM)
    assert (
        min(s.z for s in imported.design.structures) < 0
    )  # original reference retained


def test_downward_tabulated_bias_and_scalar_bias_compose(component):
    process = stack(
        si=level(zmin=1, thickness=-1, bias=0.1, z_to_bias=((0, 1), (0, 0.2)))
    )
    imported = import_native(component, process, ((-0.5, -0.5, -0.1), (4.5, 2.5, 1.1)))
    assert native_probe(imported, (2, -0.22, 0.2)) == pytest.approx(12)
    assert native_probe(imported, (2, -0.22, 0.8)) == pytest.approx(1)


def test_background_fill_exclusion_taper_and_domain_edge(component):
    component.add_polygon([(1, 0.5), (3, 0.5), (3, 1.5), (1, 1.5)], layer=(2, 0))
    process = stack(
        box=level(
            layer=(99, 0),
            material="sio2",
            background=True,
            background_exclude_layers=[(2, 0)],
            sidewall_angle=10,
        )
    )
    imported = import_native(component, process, ((-1, -1, 0), (5, 3, 1)))
    assert native_probe(imported, (-0.98, 0, 0.9)) == pytest.approx(2.25)
    assert native_probe(imported, (2, 1, 0.5)) == pytest.approx(1)
    # Positive taper erodes the material: excluded holes expand upward.
    assert native_probe(imported, (2, 0.4, 0.1)) == pytest.approx(2.25)
    assert native_probe(imported, (2, 0.4, 0.9)) == pytest.approx(1)


def test_explicit_fills_autobounds_and_skip_report(component):
    process = stack(
        core=level(thickness=0.2),
        box=level(layer=(99, 0), zmin=-10, thickness=10, material="sio2"),
        unused=level(layer=(55, 0), material="unknown"),
    )
    imported = import_component(
        component,
        layer_stack=process,
        material_map=MATERIALS,
        fill_levels=["box"],
        xy_padding=0.2 * UM,
        z_padding=0.3 * UM,
    )
    assert imported.design.depth == pytest.approx(
        0.8 * UM
    )  # unused substrate does not inflate domain
    assert native_probe(imported, (-0.1, 1, -0.2)) == pytest.approx(2.25)
    assert (
        next(r for r in imported.report if r["name"] == "unused")["status"]
        == "empty mask"
    )
    with pytest.raises(TypeError):
        imported.report[0]["status"] = "changed"
    with pytest.raises(ValueError, match="bounds"):
        import_component(
            component,
            layer_stack=stack(box=process.layers["box"]),
            material_map=MATERIALS,
            fill_levels=["box"],
        )


def test_mesh_order_info_ties_and_user_geometry_win(component):
    process = stack(
        first=level(material="si", mesh_order=-5),
        second=level(material="sin", mesh_order=99, info={"mesh_order": -5}),
    )
    imported = import_native(component, process, ((0, 0, 0), (4, 2, 1)))
    assert native_probe(imported, (2, 1, 0.5)) == pytest.approx(4)
    changed = imported.design.with_structure(
        Rectangle(
            position=(0, 0, 0),
            width=4 * UM,
            height=2 * UM,
            depth=UM,
            material=Material(9),
        )
    )
    assert sample_epsilon(
        compile_scene(from_beamz(changed)), (2, 1, 0.5)
    ) == pytest.approx(9)
    scene = from_gdsfactory(component, process, material_map=MATERIALS)
    assert all(obj.priority < 0 for obj in scene.objects)


@pytest.mark.parametrize(
    "material",
    [
        Material((12, 11, 10)),
        PoleResidue(2, [(-1e14, 1e14)], frequency_range=(1e14, 3e14)),
    ],
)
def test_material_objects_survive_native_and_prepare(component, material):
    component.add_port("o1", center=(0, 1), width=0.5, orientation=180, layer=(1, 0))
    imported = import_component(
        component, layer_stack=stack(core=level()), material_map={"si": material}
    )
    assert all(s.material is material for s in imported.design.structures)
    assert any(m is material for m in from_beamz(imported.design).materials)
    source_design = imported.design
    source_origin = imported.source_origin
    prepared = prepare(imported)
    assert imported.design is source_design
    assert imported.source_origin == source_origin
    assert prepared.design.width == pytest.approx(imported.design.width + 6 * UM)
    assert all(s.material is material for s in prepared.design.structures)
    with pytest.raises(ValueError, match="already resolves"):
        prepare(imported, material_map={"si": 12})


def test_multiple_port_heights_require_selection(component):
    component.add_port("o1", center=(0, 1), width=0.5, orientation=180, layer=(1, 0))
    process = stack(lower=level(thickness=0.2), upper=level(zmin=0.5, thickness=0.2))
    with pytest.raises(ValueError, match="multiple heights"):
        import_component(component, layer_stack=process, material_map=MATERIALS)
    imported = import_component(
        component,
        layer_stack=process,
        material_map=MATERIALS,
        port_levels={"o1": "upper"},
    )
    assert imported.port("o1").center[2] + imported.source_origin[2] == pytest.approx(
        0.6 * UM
    )


def test_noncardinal_optical_port_rejected_and_electrical_port_ignored(component):
    component.add_port(
        "heater",
        center=(0, 1),
        width=0.5,
        orientation=45,
        layer=(1, 0),
        port_type="electrical",
    )
    assert (
        import_component(
            component, layer_stack=stack(core=level()), material_map=MATERIALS
        ).ports
        == ()
    )
    component.add_port("o1", center=(0, 1), width=0.5, orientation=45, layer=(1, 0))
    with pytest.raises(ValueError, match="axis-aligned"):
        import_component(
            component, layer_stack=stack(core=level()), material_map=MATERIALS
        )


def test_rib_port_extensions_continue_slab_and_core_without_seam():
    activate_pdk()
    case = BUILDERS["rib_etch"]()
    imported = import_component(
        case.component,
        layer_stack=case.stack,
        material_map=MATERIALS,
        xy_padding=0.5 * UM,
        z_padding=0.2 * UM,
        extend_ports=True,
    )
    for x in (-0.4, -0.01, 0.01, 3.99, 4.01, 4.4):
        assert native_probe(imported, (x, 1, 0.2)) == pytest.approx(12)
        assert native_probe(imported, (x, 0.3, 0.04)) == pytest.approx(12)
        assert native_probe(imported, (x, 0.3, 0.2)) == pytest.approx(1)
    prepared = prepare(
        case.component,
        layer_stack=case.stack,
        material_map=MATERIALS,
        settings=Settings(xy_padding=0.5 * UM, z_padding=0.2 * UM),
    )
    assert native_probe(prepared.imported, (-0.4, 1, 0.2)) == pytest.approx(12)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sidewall_angle": 90}, "invalid"),
        ({"width_to_z": 1.1}, "invalid"),
        ({"bias": (0.1, 0.2)}, "scalar bias"),
        ({"sidewall_angle": 10, "z_to_bias": ((0, 1), (0, 0.1))}, "not both"),
    ],
)
def test_unsupported_process_fields_fail_with_level_name(component, kwargs, match):
    with pytest.raises(ValueError, match=match) as error:
        import_component(
            component,
            layer_stack=stack(problem=level(**kwargs)),
            material_map=MATERIALS,
        )
    assert "problem" in str(error.value)


def test_outside_levels_do_not_require_materials(component):
    process = stack(core=level(), outside=level(zmin=5, material="unmapped"))
    imported = import_native(component, process, ((0, 0, 0), (4, 2, 1)))
    assert imported.report[1]["status"] == "outside bounds"


@pytest.mark.parametrize("reference", [0, 0.37, 0.5, 1])
def test_taper_volume_matches_analytical_integral(component, reference):
    process = stack(
        core=level(sidewall_angle=20, width_to_z=reference, z_to_bias=((0, 1), (0, 0)))
    )
    scene = from_gdsfactory(component, process, material_map=MATERIALS)
    # The thin x section is wholly inside the guide. Integrate the complete
    # trapezoid across y and z: mean width = 2 - tan(theta)*(1 - 2*r).
    grid = Grid.uniform((1.9 * UM, -UM, 0), (2.1 * UM, 3 * UM, UM), (1, 16, 16))
    result = scene.rasterize(
        grid, options=RasterOptions(quality="reference", smoothing="volume")
    )
    width = 2 - np.tan(np.radians(20)) * (1 - 2 * reference)
    expected = 1 + 11 * width / 4
    assert float(result.tensors["epsilon"].mean()) == pytest.approx(expected, abs=1e-3)


def test_tied_background_does_not_override_later_pattern(component):
    process = stack(
        box=level(layer=(99, 0), material="sio2", background=True), core=level()
    )
    imported = import_native(component, process, ((-1, -1, 0), (5, 3, 1)))
    assert native_probe(imported, (2, 1, 0.5)) == pytest.approx(12)
    assert native_probe(imported, (-0.5, 1, 0.5)) == pytest.approx(2.25)


@pytest.mark.parametrize(
    "bounds",
    [((0, 0, 0), (1, 1, 0)), ((0, 0), (1, 1)), ((0, 0, 0), (1, 1, float("inf")))],
)
def test_invalid_domain_rejected(component, bounds):
    with pytest.raises(ValueError, match="bounds"):
        import_native(component, stack(core=level()), bounds)


def test_missing_material_and_unknown_fill_are_explicit(component):
    with pytest.raises(ValueError, match="unmapped.*core"):
        import_component(
            component,
            layer_stack=stack(core=level(material="unmapped")),
            material_map={},
        )
    with pytest.raises(ValueError, match="Unknown fill_levels"):
        import_component(
            component,
            layer_stack=stack(core=level()),
            material_map={},
            fill_levels=["typo"],
        )


def test_fully_excluded_background_does_not_require_material(component):
    component.add_polygon([(-2, -2), (6, -2), (6, 4), (-2, 4)], layer=(2, 0))
    process = stack(
        core=level(),
        excluded=level(
            layer=(99, 0),
            background=True,
            background_exclude_layers=[(2, 0)],
            material="unmapped",
        ),
    )
    imported = import_native(component, process, ((0, 0, 0), (4, 2, 1)))
    assert imported.report[1]["status"] == "outside bounds"


def test_overlap_precedence_is_independent_of_stack_insertion_order():
    activate_pdk()
    case = BUILDERS["stacked_heater"]()
    case.stack.layers = dict(reversed(list(case.stack.layers.items())))
    scene = from_gdsfactory(case.component, case.stack, material_map=MATERIALS)
    checks = check_probes(scene, case.probes)
    assert all(check["passed"] for check in checks), checks


@pytest.mark.parametrize("name", BUILDERS)
def test_gds_roundtrip_preserves_physical_geometry(name, tmp_path):
    activate_pdk()
    case = BUILDERS[name]()
    path = case.component.write_gds(tmp_path / f"{name}.gds")
    # GDS stores the masks, not the process stack. Reapply the same stack.
    scene = from_gdsfactory(gf.import_gds(path), case.stack, material_map=MATERIALS)
    checks = check_probes(scene, case.probes)
    assert all(check["passed"] for check in checks), checks
