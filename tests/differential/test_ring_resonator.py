"""Differential validation of the issue-104 single-bus ring resonator."""

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
from tests.differential.passive_soi.experiments import (
    FieldDecayFailure,
    check_known_failure,
)
from tests.differential.passive_soi.ring_resonator import (
    RING_FIELD_DECAY_THRESHOLD,
    build_ring_resonator_simulation,
    extract_ring_resonances,
    run_ring_resonator_benchmark,
)


def test_ring_fixture_matches_pinned_gds_geometry_and_ports(tmp_path):
    case = load_passive_soi_case("ring_resonator")
    component = generate_layout(case)

    for layer, expected in expected_layer_fingerprints(case).items():
        assert layer_union_sha256(component, layer) == expected
    assert {port.name for port in component.ports} == set(case.geometry["ports"])
    for port in component.ports:
        expected = case.geometry["ports"][port.name]
        np.testing.assert_allclose(port.dcenter, expected["center_um"], atol=1e-12)
        assert port.dwidth == pytest.approx(expected["width_um"])
        assert port.orientation == pytest.approx(expected["orientation_deg"])
    assert write_layout_gds(case, tmp_path / "ring.gds").is_file()


def test_ring_setup_uses_supplementary_lowest_resolution_protocol():
    from beamz import LIGHT_SPEED, µm

    simulation, ports, frequencies = build_ring_resonator_simulation()

    np.testing.assert_allclose(
        (simulation.design.width, simulation.design.height, simulation.design.depth),
        np.asarray((32.0, 24.05, 4.0)) * 1e-6,
        rtol=0.0,
        atol=1e-15,
    )
    assert {port.name for port in ports} == {"o1", "o2"}
    assert frequencies.size == 101
    assert simulation.sources[0].mode_spec.polarization == "te"
    assert simulation.sources[0].mode_spec.num_modes == 5
    assert simulation.sources[0].mode_spec.num_freqs == 3
    assert simulation.sources[0].size == pytest.approx((0.0, 4.5 * µm, 2.0 * µm))
    port_by_name = {port.name: port for port in ports}
    assert port_by_name["o1"].center[0] == pytest.approx(1.5 * µm)
    assert port_by_name["o2"].center[0] == pytest.approx(31.0 * µm)
    assert all(port.size == pytest.approx((0.0, 4.5 * µm, 2.0 * µm)) for port in ports)
    assert simulation.run_time == pytest.approx(30.0 * 32.0 * µm * 2.0 / LIGHT_SPEED)
    assert simulation.boundaries[0].formulation == "sponge"
    assert simulation.boundaries[0].thickness == pytest.approx(1e-6)
    assert not simulation.grid.is_uniform
    with pytest.raises(ValueError, match="unsupported supplementary resolution"):
        build_ring_resonator_simulation(resolution_ppw=7)


def test_ring_resonance_extraction_reports_fsr_q_and_extinction():
    wavelengths = np.linspace(1.54, 1.56, 101)
    power = np.ones_like(wavelengths)
    for center in (1.544, 1.554):
        power -= 0.8 / (1.0 + ((wavelengths - center) / 0.0005) ** 2)

    (
        resonances,
        fsr_nm,
        fwhm_nm,
        loaded_q,
        extinction_db,
        normalized,
    ) = extract_ring_resonances(wavelengths, power)

    np.testing.assert_allclose(resonances, (1.544, 1.554), atol=2e-4)
    assert fsr_nm == pytest.approx(10.0, abs=0.3)
    assert fwhm_nm == pytest.approx(1.0, abs=0.08)
    assert loaded_q > 1_000.0
    assert extinction_db > 6.0
    assert max(normalized) == pytest.approx(1.0)


@pytest.mark.hardware
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=FieldDecayFailure,
    reason=(
        "the pinned supplementary Lumerical-silicon 6.40 ps runtime reaches its time limit before "
        "the 1e-5 field-decay convergence criterion"
    ),
)
def test_ring_repository_runtime_converges_before_resonance_validation(
    validation_metrics,
):
    case = load_passive_soi_case("ring_resonator")
    protocol = case.geometry["simulation"]
    artifact_root = os.environ.get("BEAMZ_VALIDATION_ARTIFACT_DIR")
    result = run_ring_resonator_benchmark(
        resolution_ppw=6,
        progress=True,
        artifact_dir=(
            Path(artifact_root) / "ring_resonator" / "6ppw" if artifact_root else None
        ),
    )
    metadata = {
        "execution_backend": result.backend,
        "resonance_wavelengths_um": result.resonance_wavelengths_um,
        "free_spectral_range_nm": result.free_spectral_range_nm,
        "lowest_resonance_fwhm_nm": result.lowest_resonance_fwhm_nm,
        "loaded_q": result.loaded_q,
        "extinction_db": result.extinction_db,
        "runtime_s": result.runtime_s,
        "gcups": result.gcups,
        "cells": result.cells,
        "steps": result.steps,
        "grid_shape": result.grid_shape,
        "termination_reason": result.termination_reason,
        "terminal_field_decay": result.terminal_field_decay,
        "reference_result_limitation": protocol["reference_result_limitation"],
    }
    assert np.all(np.isfinite(result.through_power_spectrum))
    assert np.all(np.isfinite(result.reflection_power_spectrum))
    check_known_failure(
        validation_metrics.check_upper,
        FieldDecayFailure,
        "ring terminal field-decay ratio",
        measured=result.terminal_field_decay,
        upper_bound=RING_FIELD_DECAY_THRESHOLD,
        unit="ratio",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    validation_metrics.check_upper(
        "ring center-frequency selected modal output power",
        measured=result.center_total_output_power,
        upper_bound=1.02,
        unit="fraction",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    validation_metrics.check_lower(
        "ring resonance count across 20 nm",
        measured=len(result.resonance_wavelengths_um),
        lower_bound=2,
        unit="count",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    fsr_lower, fsr_upper = protocol["expected_fsr_nm_bounds"]
    validation_metrics.check_lower(
        "ring median free spectral range",
        measured=result.free_spectral_range_nm,
        lower_bound=fsr_lower,
        unit="nm",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    validation_metrics.check_upper(
        "ring median free spectral range",
        measured=result.free_spectral_range_nm,
        upper_bound=fsr_upper,
        unit="nm",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    published = protocol["published_ring_results"]["six_ppw_acceptance"]
    validation_metrics.check_lower(
        "ring lowest-resonance FWHM",
        measured=result.lowest_resonance_fwhm_nm,
        lower_bound=published["fwhm_nm"][0],
        unit="nm",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    validation_metrics.check_upper(
        "ring lowest-resonance FWHM",
        measured=result.lowest_resonance_fwhm_nm,
        upper_bound=published["fwhm_nm"][1],
        unit="nm",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    validation_metrics.check_lower(
        "ring through-port extinction",
        measured=result.extinction_db,
        lower_bound=protocol["expected_minimum_extinction_db"],
        unit="dB",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    validation_metrics.check_lower(
        "ring loaded Q",
        measured=result.loaded_q,
        lower_bound=published["q"][0],
        unit="dimensionless",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
    validation_metrics.check_upper(
        "ring loaded Q",
        measured=result.loaded_q,
        upper_bound=published["q"][1],
        unit="dimensionless",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
