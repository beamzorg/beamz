"""Lowest-resolution comparisons for spatial and polarization conversion."""

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
    write_layout_gds,
)
from tests.differential.passive_soi.experiments import (
    OutputPowerFailure,
    check_known_failure,
    power_comparison,
)
from tests.differential.passive_soi.four_port import converged_power_reference
from tests.differential.passive_soi.mode_conversion import (
    DEVICES,
    build_mode_conversion_simulation,
    run_mode_conversion_benchmark,
)


@pytest.mark.parametrize(
    ("name", "observable", "nominal", "samples", "absolute_tolerance"),
    (
        ("crossing", "through", 0.9571666667, (0.954, 0.939), 0.0181666667),
        ("directional_coupler", "cross", 0.4473333333, (0.493, 0.697), 0.2496666667),
        ("mmi2x2", "cross", 0.485, (0.376, 0.358), 0.127),
        ("mode_converter", "conversion", 0.473, (0.967, 0.357), 0.494),
        (
            "polarization_splitter_rotator",
            "conversion",
            0.94925,
            (0.147, 0.051),
            0.89825,
        ),
    ),
)
def test_reference_uses_converged_nominal_and_same_ppw_spread(
    name, observable, nominal, samples, absolute_tolerance
):
    case = load_passive_soi_case(name)
    reference = converged_power_reference(
        case,
        f"published_converged_{observable}_power_1550nm_span20nm",
        resolution_ppw=6,
    )

    assert reference.nominal == pytest.approx(nominal, abs=1e-9)
    assert reference.samples == pytest.approx(samples, abs=1e-9)
    assert reference.absolute_tolerance == pytest.approx(absolute_tolerance, abs=1e-9)


@pytest.mark.parametrize("name", DEVICES)
def test_conversion_layout_matches_reference_gds_and_ports(name, tmp_path):
    case = load_passive_soi_case(name)
    component = generate_layout(case)
    for layer, fingerprint in expected_layer_fingerprints(case).items():
        assert layer_union_sha256(component, layer) == fingerprint
    assert {p.name for p in component.ports} == set(case.geometry["ports"])
    for port in component.ports:
        expected = case.geometry["ports"][port.name]
        np.testing.assert_allclose(port.dcenter, expected["center_um"], atol=1e-12)
        assert port.dwidth == pytest.approx(expected["width_um"])
        assert port.orientation == expected["orientation_deg"]
    assert write_layout_gds(case, tmp_path / f"{name}.gds").is_file()


@pytest.mark.parametrize("name", DEVICES)
def test_conversion_setup_uses_distinct_source_and_output_modes(name):
    simulation, ports, outputs, frequencies = build_mode_conversion_simulation(name)
    protocol = load_passive_soi_case(name).geometry["simulation"]
    ports = {p.name: p for p in ports}
    assert (
        simulation.sources[0].mode_spec.polarization == protocol["source_polarization"]
    )
    assert simulation.sources[0].mode_spec.num_modes == 5
    assert all(m.mode_spec.num_modes == 5 for m in simulation.monitors)
    assert ports["o1"].polarization == protocol["source_polarization"]
    assert ports["conversion"].mode_index == protocol["conversion_mode_index"]
    assert ports["conversion"].polarization == "te"
    assert ports["crosstalk"].polarization == protocol["crosstalk_polarization"]
    assert ports["conversion"].monitor_name == ports["crosstalk"].monitor_name
    assert len(outputs) == len(set(outputs)) == 3
    assert frequencies.size == 5
    assert not simulation.grid.is_uniform
    assert simulation.boundaries[0].thickness == pytest.approx(1e-6)
    assert simulation.boundaries[0].formulation == (
        "sponge" if name == "mode_converter" else "cpml"
    )
    if name == "polarization_splitter_rotator":
        # The cladding starts at the substrate plane, surrounding the core.
        nitride = [
            s for s in simulation.design.structures if s.material.permittivity == 4.0
        ]
        assert nitride
        assert all(s.z == pytest.approx(2e-6) for s in nitride)
    with pytest.raises(ValueError, match="unsupported paper resolution"):
        build_mode_conversion_simulation(name, resolution_ppw=7)


@pytest.mark.hardware
@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "mode_converter",
            marks=pytest.mark.xfail(
                strict=True,
                raises=OutputPowerFailure,
                reason=(
                    "The selected output-power ratio reaches 1.123 across the "
                    "20 nm band, above the 1.02 passivity bound."
                ),
            ),
        ),
        "polarization_splitter_rotator",
    ],
)
@pytest.mark.parametrize("resolution_ppw", [6], ids=["6ppw"])
def test_conversion_power_is_physical_and_characterizes_reference(
    name, resolution_ppw, validation_metrics
):
    case = load_passive_soi_case(name)
    artifact_root = os.environ.get("BEAMZ_VALIDATION_ARTIFACT_DIR")
    result = run_mode_conversion_benchmark(
        name,
        resolution_ppw=resolution_ppw,
        progress=True,
        artifact_dir=Path(artifact_root) / name / "6ppw" if artifact_root else None,
    )
    reference = converged_power_reference(
        case,
        "published_converged_conversion_power_1550nm_span20nm",
        resolution_ppw=resolution_ppw,
    )
    metadata = asdict(result)
    metadata["published_reference"] = asdict(reference)
    metadata["output_power_basis"] = (
        "Selected output modes only; not all guided or radiated power."
    )
    assert np.all(np.isfinite(result.conversion_spectrum))
    assert np.all(np.isfinite(result.crosstalk_spectrum))
    comparison = power_comparison(
        case, "conversion", resolution_ppw, result.conversion_power
    )
    metadata["reference_comparison"] = comparison
    if comparison["reference_eligible"]:
        lower, upper = comparison["converged_range"]
        validation_metrics.check_lower(
            "converted power converged-reference lower bound",
            measured=result.conversion_power,
            lower_bound=lower,
            metadata=metadata,
        )
        validation_metrics.check_upper(
            "converted power converged-reference upper bound",
            measured=result.conversion_power,
            upper_bound=upper,
            metadata=metadata,
        )
    check_known_failure(
        validation_metrics.check_upper,
        OutputPowerFailure,
        f"{name} maximum selected output power across 20 nm",
        measured=max(result.selected_output_spectrum),
        upper_bound=1.02,
        unit="fraction",
        resolution="6 cells per wavelength",
        metadata=metadata,
    )
