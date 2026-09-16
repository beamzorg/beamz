"""Differential validation of the passive-SOI 2x2 MMI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.differential.passive_soi.common import (
    expected_layer_fingerprints,
    generate_layout,
    layer_union_sha256,
    load_passive_soi_case,
    write_layout_gds,
)
from tests.differential.passive_soi.experiments import power_comparison
from tests.differential.passive_soi.mmi2x2 import (
    build_mmi2x2_simulation,
    run_mmi2x2_benchmark,
)


def _port_values(port):
    center = np.asarray(getattr(port, "dcenter", port.center), dtype=float)
    width = float(getattr(port, "dwidth", port.width))
    return center, width, float(port.orientation)


def test_generated_mmi2x2_matches_paper_geometry():
    case = load_passive_soi_case("mmi2x2")
    component = generate_layout(case)

    for layer, expected in expected_layer_fingerprints(case).items():
        assert layer_union_sha256(component, layer) == expected


def test_generated_mmi2x2_preserves_paper_ports():
    case = load_passive_soi_case("mmi2x2")
    component = generate_layout(case)
    expected_ports = case.geometry["ports"]

    assert {port.name for port in component.ports} == set(expected_ports)
    for port in component.ports:
        center, width, orientation = _port_values(port)
        expected = expected_ports[port.name]
        np.testing.assert_allclose(center, expected["center_um"], atol=1e-12, rtol=0.0)
        assert width == pytest.approx(expected["width_um"], abs=1e-12)
        assert orientation == pytest.approx(expected["orientation_deg"], abs=1e-12)


def test_mmi2x2_gds_is_generated_on_demand(tmp_path):
    case = load_passive_soi_case("mmi2x2")
    destination = tmp_path / "mmi2x2.gds"

    written = write_layout_gds(case, destination)

    assert written == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0


def test_mmi2x2_simulation_uses_paper_stack_and_domain():
    case = load_passive_soi_case("mmi2x2")
    simulation, ports, frequencies = build_mmi2x2_simulation(resolution_ppw=6)

    np.testing.assert_allclose(
        (simulation.design.width, simulation.design.height, simulation.design.depth),
        np.asarray([34.0, 7.5, 4.0]) * 1e-6,
        rtol=0.0,
        atol=1e-15,
    )
    assert {port.name for port in ports} == {"o1", "o2", "o3", "o4"}
    assert {port.size[1] / 1e-6 for port in ports} == {3.0}
    assert frequencies.size == 5
    assert simulation.sources[0].mode_spec.num_freqs == 3
    np.testing.assert_allclose(
        np.diff(frequencies), np.diff(frequencies)[0], rtol=1e-12
    )
    assert not simulation.grid.is_uniform
    assert simulation.boundaries[0].formulation == "cpml"
    assert simulation.boundaries[0].thickness == pytest.approx(1.0e-6)
    reference_z_min = case.geometry["simulation"]["domain_bounds_um"]["z"][0]
    assert {
        round(structure.z / 1e-6 + reference_z_min, 12)
        for structure in simulation.design.structures
    } == {0.0}


@pytest.mark.hardware
@pytest.mark.slow
@pytest.mark.parametrize(
    "resolution_ppw",
    [6],
    ids=lambda value: f"{value}ppw",
)
def test_mmi2x2_power_is_physical_and_characterizes_reference(
    resolution_ppw, validation_metrics
):
    case = load_passive_soi_case("mmi2x2")
    artifact_root = os.environ.get("BEAMZ_VALIDATION_ARTIFACT_DIR")
    artifact_dir = (
        Path(artifact_root) / "mmi2x2" / f"{resolution_ppw}ppw"
        if artifact_root
        else None
    )
    result = run_mmi2x2_benchmark(
        resolution_ppw=resolution_ppw,
        progress=True,
        artifact_dir=artifact_dir,
    )
    metadata = {
        "execution_backend": result.backend,
        "through_te0_power": result.through_power,
        "total_output_te0_power": result.total_output_power,
        "excess_loss": result.excess_loss,
        "runtime_s": result.runtime_s,
        "gcups": result.gcups,
        "cells": result.cells,
        "steps": result.steps,
        "grid_shape": result.grid_shape,
        "termination_reason": result.termination_reason,
        "wavelength_span_nm": result.wavelength_span_nm,
    }
    comparison = power_comparison(case, "cross", resolution_ppw, result.cross_power)
    metadata["reference_comparison"] = comparison
    if comparison["reference_eligible"]:
        lower, upper = comparison["converged_range"]
        validation_metrics.check_lower(
            "converted power converged-reference lower bound",
            measured=result.cross_power,
            lower_bound=lower,
            metadata=metadata,
        )
        validation_metrics.check_upper(
            "converted power converged-reference upper bound",
            measured=result.cross_power,
            upper_bound=upper,
            metadata=metadata,
        )
    validation_metrics.check_upper(
        "2x2 MMI maximum selected output power across the band",
        measured=max(result.total_output_power_spectrum),
        upper_bound=1.02,
        unit="fraction",
        resolution=f"{resolution_ppw} cells per wavelength",
        backend="beamz",
        metadata={
            **metadata,
            "rationale": (
                "A passive device cannot create power; 2% allows modal projection "
                "and discretization error."
            ),
        },
    )
