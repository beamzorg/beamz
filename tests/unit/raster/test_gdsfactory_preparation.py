"""One preparation policy for raw layouts, cached imports and repeated prepares."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gdsfactory")

import gdsfactory as gf
from gdsfactory.technology import LogicalLayer

from beamz.design import import_component
from beamz.design.gdsfactory import Settings, prepare
from beamz.design.raster.importers._beamz import from_beamz
from examples.gdsfactory_imports.cases import BUILDERS, MATERIALS, UM
from examples.gdsfactory_imports.run import activate_pdk
from tests.unit.raster.test_gdsfactory_native import native_probe

pytestmark = [
    pytest.mark.filterwarnings("ignore:Support for class-based.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:Exception ignored in.*gdsfactory/generic_tech/layer_views.yaml:pytest.PytestUnraisableExceptionWarning"
    ),
]


def assert_same_geometry(first, second):
    assert from_beamz(first.design).to_json() == from_beamz(second.design).to_json()
    assert first.source_origin == second.source_origin
    assert [(p.name, p.center, p.size, p.signed_direction) for p in first.ports] == [
        (p.name, p.center, p.size, p.signed_direction) for p in second.ports
    ]
    assert first.report == second.report


@pytest.mark.parametrize("name", BUILDERS)
def test_one_call_and_two_stage_prepare_match(name):
    activate_pdk()
    case = BUILDERS[name]()
    # Geometry-only examples acquire an explicit test port and level association.
    if not tuple(case.component.ports):
        center = (
            (1, 0.25)
            if name == "hierarchy_holes"
            else (0, 0.5 if name in {"sidewall_references", "bias_profiles"} else 1)
        )
        case.component.add_port(
            "o1", center=center, width=0.4, orientation=180, layer=(1, 0)
        )
    selected = (
        {
            p.name: {
                "sidewall_references": "reference_0.37",
                "bias_profiles": "constant",
                "hierarchy_holes": "ring",
            }[name]
            for p in case.component.ports
        }
        if name in {"sidewall_references", "bias_profiles", "hierarchy_holes"}
        else None
    )
    options = dict(layer_stack=case.stack, material_map=MATERIALS, port_levels=selected)
    settings = Settings(xy_padding=0.5 * UM, z_padding=0.2 * UM)
    geometry = import_component(case.component, **options)
    before = from_beamz(geometry.design).to_json()
    direct = prepare(case.component, settings=settings, **options)
    staged = prepare(geometry, settings=settings)
    assert_same_geometry(direct.imported, staged.imported)
    assert from_beamz(geometry.design).to_json() == before
    changed = settings.updated_copy(xy_padding=0.8 * UM, z_padding=0.4 * UM)
    rebuilt = prepare(staged.imported, settings=changed)
    fresh = prepare(case.component, settings=changed, **options)
    assert_same_geometry(rebuilt.imported, fresh.imported)


def test_simple_defaults_and_prepared_legacy_imports_are_consistent():
    activate_pdk()
    c = gf.components.straight(length=2)
    settings = Settings(xy_padding=0.5 * UM, z_padding=0.2 * UM)
    raw = prepare(c, settings=settings)
    geometry = import_component(c)
    staged = prepare(geometry, settings=settings)
    assert_same_geometry(raw.imported, staged.imported)
    padded = import_component(c, xy_padding=2 * UM, z_padding=UM, extend_ports=True)
    assert_same_geometry(raw.imported, prepare(padded, settings=settings).imported)


def test_cached_import_does_not_read_layout_or_call_profile_again(monkeypatch):
    activate_pdk()
    c = gf.Component()
    c.add_polygon([(0, 0), (4, 0), (4, 2), (0, 2)], layer=(1, 0))
    c.add_port("o1", center=(0, 1), width=0.5, orientation=180, layer=(1, 0))
    calls = []

    def profile(z):
        calls.append(z)
        return 0.1 * z * z

    level = SimpleNamespace(
        layer=LogicalLayer(layer=(1, 0)),
        material="si",
        zmin=0,
        thickness=0.3,
        z_to_bias=profile,
    )
    stack = SimpleNamespace(layers={"core": level})
    materials = {"si": MATERIALS["si"]}
    imported = import_component(c, layer_stack=stack, material_map=materials)
    original_calls = len(calls)

    def fail(*args, **kwargs):
        raise AssertionError("Cached preparation must not read source masks")

    monkeypatch.setattr(LogicalLayer, "get_shapes", fail)
    level.zmin = 99
    materials["si"] = MATERIALS["sin"]
    prepared = prepare(imported, settings=Settings(xy_padding=UM, z_padding=UM))
    assert len(calls) == original_calls
    assert native_probe(prepared.imported, (-0.5, 1, 0.1)) == pytest.approx(12)
    assert prepared.imported.report[0]["z_bounds"] == (0, 0.3 * UM)


def test_background_fill_returns_when_domain_expands():
    activate_pdk()
    c = gf.Component()
    c.add_polygon([(0, 0), (4, 0), (4, 2), (0, 2)], layer=(1, 0))
    c.add_polygon([(-2, 0.5), (-1, 0.5), (-1, 1.5), (-2, 1.5)], layer=(2, 0))
    c.add_port("o1", center=(0, 1), width=0.5, orientation=180, layer=(1, 0))
    stack = SimpleNamespace(
        layers={
            "core": SimpleNamespace(
                layer=LogicalLayer(layer=(1, 0)),
                material="si",
                zmin=0,
                thickness=0.3,
                mesh_order=1,
            ),
            "box": SimpleNamespace(
                layer=(99, 0),
                material="sio2",
                zmin=-2,
                thickness=2,
                background=True,
                background_exclude_layers=[(2, 0)],
                mesh_order=10,
            ),
        }
    )
    imported = import_component(c, layer_stack=stack, material_map=MATERIALS)
    assert imported.report[1]["status"] == "outside bounds"
    prepared = prepare(imported, settings=Settings(xy_padding=3 * UM, z_padding=UM))
    assert native_probe(prepared.imported, (-2.5, 1, -0.5)) == pytest.approx(2.25)
    assert native_probe(prepared.imported, (-1.5, 1, -0.5)) == pytest.approx(1)


def test_explicit_bounds_and_padding_conflict_and_zero_padding_works():
    activate_pdk()
    case = BUILDERS["rib_etch"]()
    options = dict(layer_stack=case.stack, material_map=MATERIALS)
    bounds = np.asarray(case.bounds_um) * UM
    with pytest.raises(ValueError, match="bounds or padding"):
        import_component(case.component, bounds=bounds, xy_padding=UM, **options)
    with pytest.raises(ValueError, match="bounds or padding"):
        prepare(case.component, bounds=bounds, **options)
    settings = Settings(xy_padding=0, z_padding=0)
    direct = prepare(case.component, settings=settings, bounds=bounds, **options)
    geometry = import_component(case.component, bounds=bounds, **options)
    staged = prepare(geometry, settings=settings)
    assert_same_geometry(direct.imported, staged.imported)
    assert_same_geometry(
        direct.imported, prepare(direct.imported, settings=settings).imported
    )
    assert direct.design.width == pytest.approx(bounds[1][0] - bounds[0][0])


def test_process_stack_requires_explicit_materials():
    activate_pdk()
    case = BUILDERS["rib_etch"]()
    with pytest.raises(ValueError, match="explicit material_map"):
        prepare(case.component, layer_stack=case.stack)


def test_edited_design_is_not_silently_replaced_by_cached_masks():
    from dataclasses import replace

    from beamz import Material, Rectangle

    activate_pdk()
    geometry = import_component(gf.components.straight(length=2))
    edited = replace(
        geometry,
        design=geometry.design.with_structure(
            Rectangle(
                position=(0, 0, 0),
                width=UM,
                height=UM,
                depth=0.2 * UM,
                material=Material(2),
            )
        ),
    )
    with pytest.raises(ValueError, match="geometry was edited"):
        prepare(edited)


def test_material_map_without_process_stack_is_not_silently_ignored():
    activate_pdk()
    with pytest.raises(ValueError, match="material_map requires layer_stack"):
        prepare(gf.components.straight(length=2), material_map=MATERIALS)
