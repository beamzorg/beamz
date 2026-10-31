"""Both optimization workflows expose the same export controls and fixed cells."""

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from beamz.optimization import (
    InverseDesign,
    InverseDesignResult,
    TopologyDesignRegion,
)
from tests.topology_case import make_problem


@pytest.fixture(scope="module")
def problem():
    return make_problem(target_mode=0)


def core_geometry(design):
    return unary_union(
        [
            Polygon(
                [(x, y) for x, y, _ in s.vertices],
                [[(x, y) for x, y, _ in h] for h in getattr(s, "interiors", ())],
            )
            for s in design.structures
            if s.material.permittivity == 4
        ]
    )


@pytest.mark.parametrize(
    "mode,spacing", [("contour", None), ("contour", 25e-9), ("pixel_exact", None)]
)
def test_workflow_exports_pin_port_footprints(problem, mode, spacing):
    shape = (16, 20)
    solid = np.zeros(shape, dtype=bool)
    solid[4:12, :2] = True
    solid[2:14, -2:] = True
    void = np.zeros(shape, dtype=bool)
    void[:4, :2] = void[12:, :2] = True
    region = TopologyDesignRegion(
        size=(2e-6, 1.6e-6, np.inf),
        center=(3e-6, 2e-6, 0),
        eps_bounds=(1, 4),
        pixel_size=100e-9,
        fixed_solid=solid,
        fixed_void=void,
    )
    invdes = InverseDesign(problem.simulation, region)
    params = np.full(shape, 0.8)
    result = InverseDesignResult(invdes, (params,), None)
    geometry = core_geometry(result.export_design(mode=mode, sample_spacing=spacing))
    assert geometry.geom_type == "Polygon"
    assert geometry.covers(box(1.99e-6, 1.61e-6, 2.19e-6, 2.39e-6))
    assert geometry.intersection(box(2e-6, 1.2e-6, 2.2e-6, 1.6e-6)).area < 1e-25

    global_solid = np.zeros_like(problem.design_region.region_mask)
    global_void = global_solid.copy()
    global_solid[12:28, 20:40] = solid
    global_void[12:28, 20:40] = void
    spec = replace(
        problem.design_region, fixed_solid=global_solid, fixed_void=global_void
    )
    lower = InverseDesign(
        problem.simulation, spec, problem.objective, gradient_backend="autodiff"
    )
    full_params = np.full(spec.region_mask.shape, 0.8)
    result = InverseDesignResult(
        lower, (full_params,), None, fingerprint=lower.fingerprint, _initial_beta=1
    )
    lower_geometry = core_geometry(
        result.export_design(mode=mode, sample_spacing=spacing)
    )
    assert lower_geometry.geom_type == "Polygon"
    assert lower_geometry.covers(box(1.99e-6, 1.61e-6, 2.19e-6, 2.39e-6))
    assert lower_geometry.intersection(box(2e-6, 1.2e-6, 2.2e-6, 1.6e-6)).area < 1e-25
    assert lower.fingerprint != problem.fingerprint


def test_topology_fixed_cells_must_be_inside_region(problem):
    mask = np.ones_like(problem.design_region.region_mask)
    with pytest.raises(ValueError, match="inside"):
        replace(problem.design_region, fixed_solid=mask)
