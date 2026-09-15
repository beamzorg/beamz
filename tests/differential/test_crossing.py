"""Differential validation of the passive-SOI waveguide crossing."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tests.differential.passive_soi.common import (
    expected_layer_fingerprints,
    generate_layout,
    layer_union_sha256,
    load_passive_soi_case,
    max_spectrum_difference,
    write_layout_gds,
)
from tests.differential.passive_soi.crossing import (
    build_crossing_simulation,
    run_crossing_benchmark,
)
from tests.differential.passive_soi.four_port import converged_power_reference
from tests.validation.tolerances import Tolerance


def _port_values(port):
    center = np.asarray(getattr(port, "dcenter", port.center), dtype=float)
    width = float(getattr(port, "dwidth", port.width))
    return center, width, float(port.orientation)


def test_generated_crossing_matches_paper_geometry():
    case = load_passive_soi_case("crossing")
    component = generate_layout(case)

    for layer, expected in expected_layer_fingerprints(case).items():
        assert layer_union_sha256(component, layer) == expected


def test_generated_crossing_preserves_paper_ports():
    case = load_passive_soi_case("crossing")
    component = generate_layout(case)
    expected_ports = case.geometry["ports"]

    assert {port.name for port in component.ports} == set(expected_ports)
    for port in component.ports:
        center, width, orientation = _port_values(port)
        expected = expected_ports[port.name]
        np.testing.assert_allclose(center, expected["center_um"], atol=1e-12, rtol=0.0)
        assert width == pytest.approx(expected["width_um"], abs=1e-12)
        assert orientation == pytest.approx(expected["orientation_deg"], abs=1e-12)


def test_crossing_gds_is_generated_on_demand(tmp_path):
    case = load_passive_soi_case("crossing")
    destination = tmp_path / "crossing.gds"

    written = write_layout_gds(case, destination)

    assert written == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0


def test_crossing_simulation_uses_paper_domain_and_both_silicon_layers():
    case = load_passive_soi_case("crossing")
    simulation, ports, frequencies = build_crossing_simulation(resolution_ppw=6)
    wide_simulation, _, wide_frequencies = build_crossing_simulation(
        resolution_ppw=6, wavelength_span_nm=50.0
    )

    np.testing.assert_allclose(
        (simulation.design.width, simulation.design.height, simulation.design.depth),
        np.asarray([10.0, 11.0, 4.0]) * 1e-6,
        rtol=0.0,
        atol=1e-15,
    )
    assert {port.name for port in ports} == {"o1", "o2", "o3", "o4"}
    assert frequencies.size == 5
    assert wide_frequencies.size == 11
    assert simulation.sources[0].mode_spec.num_freqs == 3
    assert wide_simulation.sources[0].mode_spec.num_freqs == 6
    np.testing.assert_allclose(
        np.diff(frequencies), np.diff(frequencies)[0], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.diff(wide_frequencies), np.diff(wide_frequencies)[0], rtol=1e-12
    )
    assert not simulation.grid.is_uniform
    assert simulation.boundaries[0].formulation == "sponge"
    source_time = simulation.sources[0].source_time
    assert source_time.offset / source_time.fwidth < simulation.time[-1]
    source_signal, _ = source_time.sample(simulation.time)
    assert np.max(np.abs(source_signal)) > 0.9
    assert {
        round(structure.depth / 1e-9) for structure in simulation.design.structures
    } >= {150, 220}
    assert {layer["zmin_um"] for layer in case.geometry["layers"].values()} == {0.0}
    reference_z_min = case.geometry["simulation"]["domain_bounds_um"]["z"][0]
    assert {
        round(structure.z / 1e-6 + reference_z_min, 12)
        for structure in simulation.design.structures
    } == {0.0}


def test_crossing_reference_uses_the_solver_spread_at_each_ppw():
    case = load_passive_soi_case("crossing")
    reference_6 = converged_power_reference(
        case,
        "published_converged_through_power_1550nm_span20nm",
        resolution_ppw=6,
    )
    reference_10 = converged_power_reference(
        case,
        "published_converged_through_power_1550nm_span20nm",
        resolution_ppw=10,
    )

    assert reference_6.nominal == pytest.approx(0.9571666667)
    assert reference_6.samples == pytest.approx((0.954, 0.939))
    assert reference_6.absolute_tolerance == pytest.approx(reference_6.nominal - 0.939)
    assert reference_10.samples == pytest.approx((0.954, 0.958))
    assert reference_10.absolute_tolerance == pytest.approx(
        reference_10.nominal - 0.954
    )
    assert reference_6.absolute_tolerance > reference_10.absolute_tolerance


@pytest.mark.hardware
@pytest.mark.slow
@pytest.mark.parametrize(
    "resolution_ppw",
    [
        6,
        pytest.param(
            10,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "BeamZ measures 0.965 TE0 power at 10 ppw, outside the "
                    "0.954--0.960 same-PPW-calibrated acceptance interval."
                ),
            ),
        ),
        15,
        20,
        25,
    ],
    ids=lambda value: f"{value}ppw",
)
def test_crossing_through_power_matches_converged_reference(
    resolution_ppw, validation_metrics
):
    case = load_passive_soi_case("crossing")
    artifact_root = os.environ.get("BEAMZ_VALIDATION_ARTIFACT_DIR")
    artifact_dir = (
        Path(artifact_root) / "crossing" / f"{resolution_ppw}ppw"
        if artifact_root
        else None
    )
    result = run_crossing_benchmark(
        resolution_ppw=resolution_ppw,
        progress=True,
        artifact_dir=artifact_dir,
    )

    reference = converged_power_reference(
        case,
        "published_converged_through_power_1550nm_span20nm",
        resolution_ppw=resolution_ppw,
    )
    validation_metrics.check(
        "crossing TE0 through power at 1550 nm",
        measured=result.through_power,
        reference=reference.nominal,
        tolerance=Tolerance(
            name="same_ppw_reference_deviation",
            absolute=reference.absolute_tolerance,
            relative=0.0,
            rationale=(
                "Maximum deviation from the converged nominal among the "
                "Lumerical and Tidy3D values at the same PPW, with "
                "digitization precision as a floor."
            ),
        ),
        unit="fraction",
        resolution=f"{resolution_ppw} cells per wavelength",
        backend="beamz-vs-resolution-aware-published-reference",
        metadata={
            "published_reference": asdict(reference),
            "execution_backend": result.backend,
            "wavelength_span_nm": result.wavelength_span_nm,
            "runtime_s": result.runtime_s,
            "gcups": result.gcups,
            "cells": result.cells,
            "steps": result.steps,
            "grid_shape": result.grid_shape,
            "termination_reason": result.termination_reason,
        },
    )
    validation_metrics.check_upper(
        "crossing TE0 excess loss at 1550 nm",
        measured=result.excess_loss,
        upper_bound=0.06,
        unit="fraction",
        resolution=f"{resolution_ppw} cells per wavelength",
        backend="beamz",
        metadata={
            "total_output_te0_power": result.total_output_power,
            "rationale": (
                "The paper reports at least 93.9% TE0 through power at every "
                "tested resolution; total TE0 output includes the through port."
            ),
        },
    )
    validation_metrics.check_upper(
        "crossing total output TE0 power at 1550 nm",
        measured=result.total_output_power,
        upper_bound=1.02,
        unit="fraction",
        resolution=f"{resolution_ppw} cells per wavelength",
        backend="beamz",
        metadata={
            "rationale": (
                "A passive device cannot create power; 2% allows modal projection "
                "and discretization error."
            )
        },
    )


@pytest.mark.hardware
@pytest.mark.slow
def test_crossing_spectrum_is_consistent_across_source_bandwidths(
    validation_metrics,
):
    case = load_passive_soi_case("crossing")
    protocol = case.geometry["simulation"]["published_bandwidth_consistency"]
    artifact_root = os.environ.get("BEAMZ_VALIDATION_ARTIFACT_DIR")
    results = []
    for span_nm in protocol["source_spans_nm"]:
        artifact_dir = (
            Path(artifact_root)
            / "crossing"
            / "bandwidth_consistency"
            / f"{span_nm:g}nm"
            if artifact_root
            else None
        )
        results.append(
            run_crossing_benchmark(
                resolution_ppw=int(protocol["resolution_ppw"]),
                wavelength_span_nm=float(span_nm),
                progress=True,
                artifact_dir=artifact_dir,
            )
        )

    difference = max_spectrum_difference(
        results[0].wavelengths_um,
        results[0].through_power_spectrum,
        results[1].wavelengths_um,
        results[1].through_power_spectrum,
    )
    validation_metrics.check_upper(
        "crossing maximum TE0 through-power difference between source bandwidths",
        measured=difference,
        upper_bound=float(protocol["max_abs_difference"]),
        unit="fraction",
        resolution=f"{protocol['resolution_ppw']} cells per wavelength",
        backend="beamz-vs-published-bandwidth-consistency",
        metadata={
            "execution_backends": [result.backend for result in results],
            "source_spans_nm": protocol["source_spans_nm"],
            "wavelengths_um": [result.wavelengths_um for result in results],
            "through_power_spectra": [
                result.through_power_spectrum for result in results
            ],
            "excess_loss_spectra": [result.excess_loss_spectrum for result in results],
            "published_value_basis": protocol["value_basis"],
        },
    )
