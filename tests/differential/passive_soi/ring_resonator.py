"""Pinned single-bus ring-resonator benchmark for issue 104."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tests.differential.passive_soi.common import (
    domain_bounds_um,
    domain_size_um,
    load_passive_soi_case,
    port_center_and_direction,
    reference_absorber_warning_scope,
    reference_frequencies,
)
from tests.differential.passive_soi.four_port import (
    _ported_design,
    _save_four_port_artifacts,
)


@dataclass(frozen=True)
class RingResonatorResult:
    resolution_ppw: int
    backend: str
    wavelengths_um: tuple[float, ...]
    through_power_spectrum: tuple[float, ...]
    reflection_power_spectrum: tuple[float, ...]
    total_output_power_spectrum: tuple[float, ...]
    normalized_through_spectrum: tuple[float, ...]
    resonance_wavelengths_um: tuple[float, ...]
    free_spectral_range_nm: float
    loaded_q: float
    extinction_db: float
    center_total_output_power: float
    runtime_s: float
    gcups: float
    cells: int
    steps: int
    grid_shape: tuple[int, int, int]
    termination_reason: str


def extract_ring_resonances(wavelengths_um, through_power):
    """Extract resonance locations, median FSR, loaded Q, and band extinction."""
    from scipy.signal import find_peaks, peak_widths

    wavelengths = np.asarray(wavelengths_um, dtype=float)
    power = np.asarray(through_power, dtype=float)
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    power = power[order]
    if wavelengths.size < 3 or wavelengths.shape != power.shape:
        raise ValueError("ring spectrum needs at least three wavelength samples")
    if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(power)):
        raise ValueError("ring spectrum must be finite")
    peak_power = float(np.max(power))
    if peak_power <= 0.0:
        raise ValueError("ring through spectrum has no positive power")
    normalized = power / peak_power
    sample_step_nm = float(np.median(np.diff(wavelengths))) * 1e3
    minimum_distance = max(1, round(2.0 / sample_step_nm))
    prominence = max(0.02 * float(np.ptp(normalized)), 1e-4)
    minima, properties = find_peaks(
        -normalized,
        prominence=prominence,
        distance=minimum_distance,
    )
    if minima.size:
        # A single-mode ring produces one dominant resonance family. Retain dips
        # with at least half the strongest prominence so shallow time-window and
        # broadband-normalization ripple does not halve the measured FSR.
        dominant = properties["prominences"] >= 0.5 * np.max(
            properties["prominences"]
        )
        minima = minima[dominant]
    resonances = wavelengths[minima]
    fsr_nm = (
        float(np.median(np.diff(resonances))) * 1e3
        if resonances.size >= 2
        else float("nan")
    )
    loaded_q = float("nan")
    if minima.size:
        deepest = int(minima[np.argmin(normalized[minima])])
        widths, _, left, right = peak_widths(-normalized, [deepest], rel_height=0.5)
        del widths
        sample_axis = np.arange(wavelengths.size, dtype=float)
        left_wavelength = float(np.interp(left[0], sample_axis, wavelengths))
        right_wavelength = float(np.interp(right[0], sample_axis, wavelengths))
        fwhm = right_wavelength - left_wavelength
        if fwhm > 0.0:
            loaded_q = float(wavelengths[deepest] / fwhm)
    floor = max(float(np.min(normalized)), np.finfo(float).tiny)
    extinction_db = float(-10.0 * np.log10(floor))
    normalized_original_order = np.empty_like(normalized)
    normalized_original_order[order] = normalized
    return (
        tuple(float(value) for value in resonances),
        fsr_nm,
        loaded_q,
        extinction_db,
        tuple(float(value) for value in normalized_original_order),
    )


def build_ring_resonator_simulation(*, resolution_ppw: int = 6, diagnostics=False):
    """Build the pinned supplementary ring setup at its lowest resolution."""
    from beamz import (
        LIGHT_SPEED,
        Absorber,
        FieldMonitor,
        GaussianPulse,
        GridSpec,
        ModeSpec,
        Port,
        Simulation,
        µm,
    )
    from beamz.design.raster import RasterOptions

    case = load_passive_soi_case("ring_resonator")
    protocol = case.geometry["simulation"]
    if int(resolution_ppw) not in protocol["resolutions_cells_per_wavelength"]:
        raise ValueError(f"unsupported supplementary resolution {resolution_ppw}")
    design = _ported_design(case)
    bounds = domain_bounds_um(case)
    core = case.geometry["layers"]["core"]
    z_center = (
        float(core["zmin_um"]) + 0.5 * float(core["thickness_m"]) / µm - bounds["z"][0]
    ) * µm
    frequencies = reference_frequencies(case, 20.0)
    mode_spec = ModeSpec(polarization="te", num_modes=int(protocol["mode_candidates"]))
    transverse_span = float(protocol["beamz_port_transverse_span_um"]) * µm
    ports = tuple(
        Port(
            center=port_center_and_direction(
                case, name, inward_offset_um=0.5, z_center=z_center
            )[0],
            size=(0.0, transverse_span, 2.0 * µm),
            name=name,
            direction=port_center_and_direction(
                case, name, inward_offset_um=0.5, z_center=z_center
            )[1],
            mode_spec=mode_spec,
        )
        for name in ("o1", "o2")
    )
    source_center, source_direction = port_center_and_direction(
        case, "o1", inward_offset_um=0.0, z_center=z_center
    )
    source_time = GaussianPulse(
        freq0=LIGHT_SPEED / (float(protocol["wavelength_center_um"]) * µm),
        fwidth=float(np.ptp(frequencies)),
        offset=5.0 / (2.0 * np.pi),
    )
    source = Port(
        center=source_center,
        size=(0.0, transverse_span, 2.0 * µm),
        name="source",
        direction=source_direction,
        mode_spec=mode_spec,
    ).to_source(
        freq0=source_time.freq0,
        fwidth=source_time.fwidth,
        num_freqs=int(protocol["source_mode_profiles"]),
        source_time=source_time,
    )
    monitors = [port.to_monitor(frequencies) for port in ports]
    if diagnostics:
        monitors.append(
            FieldMonitor(
                center=(0.5 * design.width, 0.5 * design.height, z_center),
                size=(design.width, design.height, 0.0),
                freqs=(float(np.median(frequencies)),),
                fields=("Ex", "Ey", "Ez"),
                name="ring_resonator_xy",
            )
        )
    simulation = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[Absorber(edges="all", thickness=1.0 * µm)],
        run_time=15.0 * domain_size_um(case)[0] * µm * 2.0 / LIGHT_SPEED,
        grid_spec=GridSpec.auto(
            min_steps_per_wvl=float(resolution_ppw),
            wavelength=float(protocol["wavelength_center_um"]) * µm,
            courant=0.99,
            max_scale=1.4,
            max_total_cells=None,
        ),
        raster_options=RasterOptions(
            quality="balanced", smoothing="farjadpour_diagonal"
        ),
    )
    return simulation, ports, frequencies


def run_ring_resonator_benchmark(
    *,
    resolution_ppw: int = 6,
    progress: bool = False,
    backend: str | None = None,
    artifact_dir: str | Path | None = None,
) -> RingResonatorResult:
    """Run the ring and extract through, reflection, resonance, FSR, and Q data."""
    from beamz import LIGHT_SPEED, AutoTermination, µm
    from beamz.analysis import s_parameters

    simulation, ports, frequencies = build_ring_resonator_simulation(
        resolution_ppw=resolution_ppw,
        diagnostics=artifact_dir is not None,
    )
    with reference_absorber_warning_scope():
        program = simulation.compile(progress=progress, backend=backend)
        execution_backend = program.config.backend
        results = simulation.run(
            progress=progress,
            backend=execution_backend,
            termination=AutoTermination(
                field_decay=1e-5, monitor_change=None, consecutive_checks=1
            ),
        )
    scattering = s_parameters(
        results,
        source_port="o1",
        ports=ports,
        output_ports=("o1", "o2"),
        frequencies=frequencies,
        min_incident_db=-45.0,
    )
    if not np.all(scattering.diagnostics["valid_mask"]):
        raise ValueError("ring spectrum contains samples with no incident signal")
    if artifact_dir is not None:
        _save_four_port_artifacts(
            Path(artifact_dir),
            simulation,
            results,
            scattering,
            execution_backend=execution_backend,
        )
    wavelengths = LIGHT_SPEED / np.asarray(scattering.frequencies) / µm
    through = np.abs(np.asarray(scattering.s_matrix[("o2", "o1")])) ** 2
    reflection = np.abs(np.asarray(scattering.s_matrix[("o1", "o1")])) ** 2
    total = through + reflection
    resonances, fsr_nm, loaded_q, extinction_db, normalized = extract_ring_resonances(
        wavelengths, through
    )
    center = int(np.argmin(np.abs(wavelengths - 1.55)))
    performance = results.performance
    termination = results.termination
    return RingResonatorResult(
        resolution_ppw=int(resolution_ppw),
        backend=execution_backend,
        wavelengths_um=tuple(float(value) for value in wavelengths),
        through_power_spectrum=tuple(float(value) for value in through),
        reflection_power_spectrum=tuple(float(value) for value in reflection),
        total_output_power_spectrum=tuple(float(value) for value in total),
        normalized_through_spectrum=normalized,
        resonance_wavelengths_um=resonances,
        free_spectral_range_nm=fsr_nm,
        loaded_q=loaded_q,
        extinction_db=extinction_db,
        center_total_output_power=float(total[center]),
        runtime_s=float(performance.runtime_s),
        gcups=float(performance.gcups),
        cells=int(performance.cells),
        steps=int(performance.steps),
        grid_shape=tuple(int(value) for value in simulation.grid.shape),
        termination_reason=termination.reason if termination else "time_limit",
    )
