"""Explicit controls for reproducible passive-SOI diagnostic experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExperimentOptions:
    """Overrides leave the pinned protocol intact unless explicitly selected."""

    run_time_ps: float | None = None
    monitor_offset_um: float | None = None
    boundary_thickness_um: float = 1.0
    transverse_span_um: float | None = None
    source_profiles: int | None = None
    wavelength_step_nm: float | None = None
    exact_center: bool = False
    smoothing: str = "farjadpour_diagonal"
    port_extension_policy: str = "reference"

    def __post_init__(self):
        if self.port_extension_policy not in ("reference", "through_boundary"):
            raise ValueError(
                "port_extension_policy must be reference or through_boundary"
            )
        for name in (
            "run_time_ps",
            "boundary_thickness_um",
            "transverse_span_um",
            "wavelength_step_nm",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.monitor_offset_um is not None and (
            not np.isfinite(self.monitor_offset_um) or self.monitor_offset_um < 0
        ):
            raise ValueError("monitor_offset_um must be finite and nonnegative")
        if self.source_profiles is not None and (
            isinstance(self.source_profiles, bool)
            or int(self.source_profiles) != self.source_profiles
            or self.source_profiles < 1
        ):
            raise ValueError("source_profiles must be a positive integer")

    def frequencies(self, case, span_nm):
        from beamz import LIGHT_SPEED, µm
        from tests.differential.passive_soi.common import reference_frequencies

        frequencies = reference_frequencies(case, span_nm)
        if self.wavelength_step_nm is not None:
            count = max(2, round(span_nm / self.wavelength_step_nm) + 1)
            frequencies = np.linspace(frequencies[0], frequencies[-1], count)
        if self.exact_center:
            center = LIGHT_SPEED / (
                case.geometry["simulation"]["wavelength_center_um"] * µm
            )
            frequencies = np.unique(np.append(frequencies, center))
        return frequencies


class OutputPowerFailure(AssertionError):
    """The known selected-output passivity check failed."""


class FieldDecayFailure(AssertionError):
    """The known finite-duration field-decay check failed."""


def check_known_failure(check, exception, *args, **kwargs):
    """Restrict expected failures to one recorded numerical assertion."""
    try:
        return check(*args, **kwargs)
    except AssertionError as error:
        raise exception(str(error)) from error


DEFAULT_OPTIONS = ExperimentOptions()


def power_comparison(case, observable, resolution_ppw, measured):
    """Keep same-PPW characterization separate from converged-reference evidence."""
    protocol = case.geometry["simulation"]
    config = protocol[f"published_converged_{observable}_power_1550nm_span20nm"]
    series = protocol[config["series_key"]]
    same = [series[str(resolution_ppw)][s] for s in ("lumerical", "tidy3d")]
    converged = [
        sample[s]
        for ppw, sample in series.items()
        if ppw.isdigit() and int(ppw) >= config["convergence_start_ppw"]
        for s in ("lumerical", "tidy3d")
    ]
    uncertainty = config["digitization_absolute_uncertainty"]
    lower, upper = min(converged) - uncertainty, max(converged) + uncertainty
    eligible = resolution_ppw >= config["convergence_start_ppw"]
    return {
        "same_ppw_range": [min(same), max(same)],
        "same_ppw_agreement": min(same) - uncertainty
        <= measured
        <= max(same) + uncertainty,
        "converged_range": [lower, upper],
        "reference_eligible": eligible,
        "reference_agreement": lower <= measured <= upper if eligible else None,
        "mesh_convergence_established": False,
    }


def validate_boundary_clearance(design, source, ports, thickness_um):
    """Changing a fixed-domain absorber must not swallow a source or port center."""
    extent = np.array([design.width, design.height, design.depth])
    thickness = thickness_um * 1e-6
    for plane in (source, *ports):
        center = np.asarray(plane.center)
        clearance = float(np.min(np.minimum(center, extent - center)))
        if clearance + 1e-12 < thickness:
            raise ValueError(
                f"boundary thickness {thickness_um:g} um overlaps {getattr(plane, 'name', 'source')!r} "
                f"at clearance {clearance / 1e-6:g} um; extend the domain and "
                "preserve the interior grid before increasing the absorber"
            )
