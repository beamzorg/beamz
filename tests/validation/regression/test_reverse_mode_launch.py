"""Reciprocal transmission and direction-correct 2D modal extraction (#348)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.modal_sparam_physical_case import (
    StraightWaveguideSParamConfig,
    run_straight_waveguide_sparam_case,
)


@pytest.fixture(
    scope="module",
    params=[("x", "tm", "sponge", 10, 1.1, False)]
    + [
        (axis, polarization, absorber, ppw, offset, True)
        for axis in ("x", "y")
        for polarization in ("tm", "te")
        for absorber, ppw, offset in [
            ("sponge", 10, 1.1),
            ("cpml", 10, 1.1),
            ("cpml", 14, 1.4),
        ]
    ],
    ids=lambda case: "-".join(map(str, case)),
)
def reciprocal_straight_waveguide_results(request):
    """Run the smallest two source excitations once for this module."""
    axis, polarization, absorber, ppw, monitor_offset, mirror_on_grid = request.param
    config = StraightWaveguideSParamConfig(
        axis=axis,
        polarization=polarization,
        monitor_behind_source=True,
        mirror_on_grid=mirror_on_grid,
        pml_formulation=absorber,
        input_monitor_offset_um=monitor_offset,
    )
    forward = run_straight_waveguide_sparam_case(
        resolution_ppw=ppw,
        cfg=config,
        launch_direction="+",
    )
    reverse = run_straight_waveguide_sparam_case(
        resolution_ppw=ppw,
        cfg=config,
        launch_direction="-",
    )
    center = int(np.argmin(np.abs(forward.wavelengths_um - config.wavelength_um)))
    return config, forward, reverse, center


@pytest.mark.simulation
@pytest.mark.compiled
def test_straight_waveguide_complex_transmission_is_reciprocal(
    reciprocal_straight_waveguide_results,
    validation_metrics,
):
    """Opposite source excitations produce the same de-embedded transmission."""
    config, forward, reverse, center = reciprocal_straight_waveguide_results
    s21 = complex(forward.s21_by_monitor["far"][center])
    s12 = complex(reverse.s21_by_monitor["far"][center])
    reciprocity_residual = abs(s21 - s12)

    validation_metrics.check(
        "complex transmission reciprocity |S21-S12|",
        measured=reciprocity_residual,
        reference=0.0,
        tolerance="sparameter_reciprocity",
        resolution=f"{forward.resolution_ppw} ppw",
        metadata={
            "device": "symmetric straight slab waveguide",
            "axis": config.axis,
            "polarization": config.polarization,
            "wavelength_um": config.wavelength_um,
            "reference_plane_separation_um": abs(
                forward.monitor_x_um["far"] - forward.monitor_x_um["o1"]
            ),
            "forward_transmission": {
                "real": float(np.real(s21)),
                "imag": float(np.imag(s21)),
            },
            "reverse_transmission": {
                "real": float(np.real(s12)),
                "imag": float(np.imag(s12)),
            },
        },
    )


@pytest.mark.simulation
@pytest.mark.compiled
@pytest.mark.parametrize("launch_direction", ["+", "-"], ids=["plus", "minus"])
def test_mode_launch_has_low_reflection_and_backward_power(
    reciprocal_straight_waveguide_results,
    launch_direction,
    validation_metrics,
):
    """Check the launched side and behind the source over the full sampled band."""
    config, forward, reverse, _center = reciprocal_straight_waveguide_results
    result = forward if launch_direction == "+" else reverse
    metadata = {
        "axis": config.axis,
        "polarization": config.polarization,
        "direction": launch_direction,
        "absorber": config.pml_formulation,
        "wavelengths_um": result.wavelengths_um.tolist(),
    }
    reflection_db = 20.0 * np.log10(np.maximum(np.abs(result.s11), 1e-12))
    assert np.max(reflection_db) < -35.0, reflection_db
    validation_metrics.check_upper(
        "maximum in-band reflection",
        float(np.max(reflection_db)),
        -35.0,
        unit="dB",
        metadata=metadata,
    )
    assert result.backward_launch is not None
    backward_db = 20.0 * np.log10(np.maximum(np.abs(result.backward_launch), 1e-12))
    assert np.max(backward_db) < -35.0, backward_db
    validation_metrics.check_upper(
        "maximum in-band backward launch",
        float(np.max(backward_db)),
        -35.0,
        unit="dB",
        metadata=metadata,
    )
    for transmission in result.s21_by_monitor.values():
        assert np.all(
            (np.abs(transmission) ** 2 > 0.95) & (np.abs(transmission) ** 2 < 1.05)
        )


@pytest.mark.simulation
@pytest.mark.compiled
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("launch_direction", ["+", "-"], ids=["plus", "minus"])
def test_pec_termination_returns_guided_power_in_every_direction(
    axis,
    polarization,
    launch_direction,
    validation_metrics,
):
    """A real reflected mode must be recovered and cross the injection plane."""
    config = StraightWaveguideSParamConfig(
        axis=axis,
        polarization=polarization,
        pml_formulation="cpml",
        mirror_on_grid=True,
        monitor_behind_source=True,
        reflecting_termination=True,
    )
    result = run_straight_waveguide_sparam_case(
        resolution_ppw=10,
        cfg=config,
        launch_direction=launch_direction,
    )
    reflected_power = np.abs(result.s11) ** 2
    assert np.all((reflected_power > 0.95) & (reflected_power < 1.05)), reflected_power
    assert result.backward_launch is not None
    returned_power = np.abs(result.backward_launch) ** 2
    assert np.all((returned_power > 0.95) & (returned_power < 1.05)), returned_power
    validation_metrics.check_upper(
        "maximum PEC reflected-power error",
        float(np.max(np.abs(reflected_power - 1))),
        0.05,
        metadata={
            "axis": axis,
            "polarization": polarization,
            "direction": launch_direction,
        },
    )
