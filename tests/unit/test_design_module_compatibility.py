"""Moved design modules preserve public imports and stored type identities."""

import importlib
import inspect
import pickle

import numpy as np
from beamz.design.core import Design as LegacyDesign
from beamz.design.grid_spec import GridSpec as LegacyGridSpec
from beamz.design.structures import Rectangle as LegacyRectangle

from beamz._cache_tokens import cache_token
from beamz.design import Design, GridSpec, Material, Rectangle
from beamz.optimization.optimizer.runtime.identity import _legacy_checkpoint_token


def test_legacy_design_imports_and_pickles_resolve_to_new_implementations():
    values = (
        (Design(width=2.0, height=1.0), "beamz.design.core"),
        (GridSpec.uniform(0.1), "beamz.design.grid_spec"),
        (
            Rectangle(position=(0.0, 0.0), width=1.0, height=1.0),
            "beamz.design.structures",
        ),
    )
    assert LegacyDesign is Design
    assert LegacyGridSpec is GridSpec
    assert LegacyRectangle is Rectangle
    for value, legacy_module in values:
        cls = type(value)
        current = pickle.dumps(value, protocol=0)
        old = current.replace(cls.__module__.encode(), legacy_module.encode())
        assert old != current
        restored = pickle.loads(old)
        assert cache_token(restored) == cache_token(value)
        assert _legacy_checkpoint_token(cache_token(value))[0] == (
            f"{legacy_module}.{cls.__name__}"
        )
        assert inspect.getsource(cls)


def test_grid_spec_remains_usable_through_legacy_module():
    design = Design(width=2.0, height=1.0, material=Material(permittivity=2))
    actual = LegacyGridSpec.uniform(0.25).realize(design)
    expected = GridSpec.uniform(0.25).realize(design)
    np.testing.assert_array_equal(actual.x_edges, expected.x_edges)
    np.testing.assert_array_equal(actual.y_edges, expected.y_edges)


def test_legacy_module_preserves_name_and_forwards_attribute_updates(monkeypatch):
    legacy = importlib.import_module("beamz.design.gdsfactory")
    implementation = importlib.import_module("beamz.design.io.gdsfactory")
    marker = object()
    monkeypatch.setattr(legacy, "s_parameters", marker)
    assert legacy.__name__ == "beamz.design.gdsfactory"
    assert implementation.s_parameters is marker


def test_legacy_module_star_exports_are_unchanged():
    assert importlib.import_module("beamz.design.grid_spec").__all__ == [
        "GridSpec",
        "MeshOverride",
    ]
    assert importlib.import_module("beamz.design.gds").__all__ == [
        "ImportedComponent",
        "export_gds",
        "import_component",
        "import_gds",
    ]
