"""Imported rib stacks reach the modal-source/PML execution path in both axes."""

import numpy as np
import pytest

pytest.importorskip("gdsfactory")

import gdsfactory as gf

from beamz.design.gdsfactory import Settings, prepare
from beamz.design.grid_spec import GridSpec
from beamz.devices.sources.time import GaussianPulse
from examples.gdsfactory_imports.cases import MATERIALS, UM, rib_etch
from examples.gdsfactory_imports.run import activate_pdk

pytestmark = [
    pytest.mark.filterwarnings("ignore:Support for class-based.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:Exception ignored in.*gdsfactory/generic_tech/layer_views.yaml:pytest.PytestUnraisableExceptionWarning"
    ),
]


@pytest.mark.parametrize("rotation", [0, 90])
def test_imported_rib_launches_finite_fields_with_modal_ports(rotation):
    activate_pdk()
    case = rib_etch()
    component = case.component
    if rotation:
        container = gf.Component()
        reference = container.add_ref(component)
        reference.drotate(rotation)
        container.add_ports(reference.ports)
        component = container
    prepared = prepare(
        component,
        layer_stack=case.stack,
        material_map=MATERIALS,
        settings=Settings(
            xy_padding=0.7 * UM,
            z_padding=0.5 * UM,
            pml_thickness=0.2 * UM,
            grid_spec=GridSpec.uniform(0.1 * UM),
            run_time=2e-15,
            wavelength_points=2,
        ),
    )
    simulation = prepared.simulation_for("o1")
    # Center the pulse at startup so this short smoke run actually launches it.
    pulse = GaussianPulse(
        prepared.settings.center_frequency, prepared.settings.frequency_width, offset=0
    )
    simulation = simulation.updated_copy(
        sources=(simulation.sources[0].updated_copy(source_time=pulse),)
    )
    run = simulation.advance(num_steps=5, backend="jax", performance=False)
    fields = [
        np.asarray(getattr(run.state, name))
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    ]
    assert int(run.state.current_step) == 5
    assert all(np.isfinite(field).all() for field in fields)
    assert max(float(np.max(np.abs(field))) for field in fields) > 0
    assert set(run.results.monitors) == {"o1", "o2"}
