"""Shared simulation adapter for passive-SOI four-port splitters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tests.differential.case_schema import DifferentialCase
from tests.differential.passive_soi.common import (
    component_polygons,
    domain_bounds_um,
    domain_size_um,
    generate_layout,
    port_center_and_direction,
    reference_absorber_warning_scope,
)
from tests.differential.passive_soi.experiments import (
    DEFAULT_OPTIONS,
    ExperimentOptions,
    validate_boundary_clearance,
)


@dataclass(frozen=True)
class FourPortBenchmarkResult:
    """Paper-comparable splitter observables and execution metadata."""

    resolution_ppw: int
    wavelength_span_nm: float
    backend: str
    wavelengths_um: tuple[float, ...]
    cross_power_spectrum: tuple[float, ...]
    through_power_spectrum: tuple[float, ...]
    total_output_power_spectrum: tuple[float, ...]
    excess_loss_spectrum: tuple[float, ...]
    cross_power: float
    through_power: float
    total_output_power: float
    excess_loss: float
    runtime_s: float
    gcups: float
    cells: int
    steps: int
    grid_shape: tuple[int, int, int]
    termination_reason: str


@dataclass(frozen=True)
class ConvergedPowerReference:
    """Converged nominal with a tolerance calibrated at one resolution."""

    nominal: float
    absolute_tolerance: float
    standard_deviation: float
    minimum: float
    maximum: float
    resolution_ppw: int
    samples: tuple[float, ...]
    converged_resolutions_ppw: tuple[int, ...]
    converged_samples: tuple[float, ...]
    converged_standard_deviation: float
    digitization_uncertainty: float


def converged_power_reference(
    case: DifferentialCase, key: str, *, resolution_ppw: int
) -> ConvergedPowerReference:
    """Estimate a converged nominal and same-PPW reference tolerance."""

    data = case.geometry["simulation"][key]
    series = case.geometry["simulation"][data["series_key"]]
    start = int(data["convergence_start_ppw"])
    resolutions = tuple(
        sorted(
            int(value)
            for value in series
            if str(value).isdigit() and int(value) >= start
        )
    )
    converged_samples = tuple(
        float(series[str(resolution)][solver])
        for resolution in resolutions
        for solver in ("lumerical", "tidy3d")
    )
    nominal = float(np.mean(converged_samples))
    resolution_key = str(int(resolution_ppw))
    if resolution_key not in series:
        raise ValueError(f"no published reference at {resolution_ppw} PPW")
    samples = tuple(
        float(series[resolution_key][solver]) for solver in ("lumerical", "tidy3d")
    )
    uncertainty = float(data.get("digitization_absolute_uncertainty", 0.0))
    return ConvergedPowerReference(
        nominal=nominal,
        absolute_tolerance=max(
            max(abs(value - nominal) for value in samples), uncertainty
        ),
        standard_deviation=float(np.std(samples)),
        minimum=min(samples),
        maximum=max(samples),
        resolution_ppw=int(resolution_ppw),
        samples=samples,
        converged_resolutions_ppw=resolutions,
        converged_samples=converged_samples,
        converged_standard_deviation=float(np.std(converged_samples)),
        digitization_uncertainty=uncertainty,
    )


def paper_cross_power_range(
    case: DifferentialCase, resolution_ppw: int
) -> tuple[float, float]:
    """Return the range bounded by the paper's two solver results."""
    references = case.geometry["simulation"]["published_cross_power_1550nm_span20nm"][
        str(int(resolution_ppw))
    ]
    values = (float(references["lumerical"]), float(references["tidy3d"]))
    return min(values), max(values)


def _ported_design(case: DifferentialCase, *, port_extension_policy="reference"):
    """Extrude the pinned stack, optionally correcting truncated port extensions."""
    from beamz import Design, Material, Polygon, Rectangle, µm

    component = generate_layout(case)
    bounds = domain_bounds_um(case)
    width_um, height_um, depth_um = domain_size_um(case)
    x_offset_um, y_offset_um = -bounds["x"][0], -bounds["y"][0]
    silicon = Material(case.materials["silicon_n_at_1p55_um"] ** 2)
    silica = Material(case.materials["silica_n_at_1p55_um"] ** 2)
    design = Design(
        width=width_um * µm,
        height=height_um * µm,
        depth=depth_um * µm,
        background=silica,
    )

    if "top_cladding" in case.geometry:
        cladding = case.geometry["top_cladding"]
        cladding_z = float(cladding["zmin_um"])
        design += Rectangle(
            position=(0.0, 0.0, (cladding_z - bounds["z"][0]) * µm),
            width=width_um * µm,
            height=height_um * µm,
            depth=(bounds["z"][1] - cladding_z) * µm,
            material=Material(case.materials[cladding["material"]] ** 2),
        )

    for layer in case.geometry["layers"].values():
        thickness = float(layer["thickness_m"])
        beamz_z_um = float(layer["zmin_um"]) - bounds["z"][0]
        for points in component_polygons(component, tuple(layer["gds"])):
            design += Polygon(
                vertices=tuple(
                    (
                        (float(x) + x_offset_um) * µm,
                        (float(y) + y_offset_um) * µm,
                    )
                    for x, y in np.asarray(points, dtype=float)[:, :2]
                ),
                z=beamz_z_um * µm,
                depth=thickness,
                material=silicon,
            )

    core = case.geometry["layers"]["core"]
    core_z = (float(core["zmin_um"]) - bounds["z"][0]) * µm
    core_depth = float(core["thickness_m"])
    extension = float(case.geometry["simulation"]["port_extension_um"]) * µm
    for port in case.geometry["ports"].values():
        x_um, y_um = port["center_um"]
        x = (float(x_um) + x_offset_um) * µm
        y = (float(y_um) + y_offset_um) * µm
        width = float(port["width_um"]) * µm
        orientation = int(round(float(port["orientation_deg"]))) % 360
        if orientation == 180:
            extension_width = (
                max(extension, x + µm)
                if port_extension_policy == "through_boundary"
                else extension
            )
            position = (x - extension_width, y - width / 2)
        elif orientation == 0:
            extension_width = (
                max(extension, design.width - x + µm)
                if port_extension_policy == "through_boundary"
                else extension
            )
            position = (x, y - width / 2)
        else:
            raise ValueError(f"unsupported port orientation {orientation}")
        design += Rectangle(
            position=(*position, core_z),
            width=extension_width,
            height=width,
            depth=core_depth,
            material=silicon,
        )
    return design.unified_polygons()


def build_four_port_simulation(
    case: DifferentialCase,
    *,
    resolution_ppw: int,
    wavelength_span_nm: float = 20.0,
    diagnostics: bool = False,
    options: ExperimentOptions = DEFAULT_OPTIONS,
):
    """Build a paper-matched BeamZ four-port simulation without executing it."""
    from beamz import (
        LIGHT_SPEED,
        PML,
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

    protocol = case.geometry["simulation"]
    if int(resolution_ppw) not in protocol["resolutions_cells_per_wavelength"]:
        raise ValueError(f"unsupported paper resolution {resolution_ppw}")
    if float(wavelength_span_nm) not in protocol["wavelength_spans_nm"]:
        raise ValueError(f"unsupported paper wavelength span {wavelength_span_nm}")

    design = _ported_design(case, port_extension_policy=options.port_extension_policy)
    bounds = domain_bounds_um(case)
    core = case.geometry["layers"]["core"]
    z_center = (
        float(core["zmin_um"]) + 0.5 * float(core["thickness_m"]) / µm - bounds["z"][0]
    ) * µm
    wavelength_center = float(protocol["wavelength_center_um"]) * µm
    frequencies = options.frequencies(case, wavelength_span_nm)
    grid_spec = GridSpec.auto(
        min_steps_per_wvl=float(resolution_ppw),
        wavelength=wavelength_center,
        courant=0.99,
        max_scale=1.4,
        max_total_cells=None,
    )

    source_name = protocol.get("source_port", "o1")
    mode_candidates = int(protocol.get("mode_candidates", 1))
    mode_spec = ModeSpec(
        polarization=protocol.get("source_polarization", "te"),
        num_modes=mode_candidates,
    )
    transverse_span = (
        float(options.transverse_span_um or protocol["beamz_port_transverse_span_um"])
        * µm
    )
    z_span = 2.0 * µm
    ports = tuple(
        Port(
            center=port_center_and_direction(
                case,
                name,
                inward_offset_um=(
                    0.5
                    if options.monitor_offset_um is None
                    else options.monitor_offset_um
                ),
                z_center=z_center,
            )[0],
            size=(0.0, transverse_span, z_span),
            name=name,
            direction=port_center_and_direction(
                case,
                name,
                inward_offset_um=(
                    0.5
                    if options.monitor_offset_um is None
                    else options.monitor_offset_um
                ),
                z_center=z_center,
            )[1],
            mode_spec=(
                mode_spec
                if name == source_name
                else ModeSpec(polarization="te", num_modes=mode_candidates)
            ),
        )
        for name in case.geometry["ports"]
    )
    source_center, source_direction = port_center_and_direction(
        case, source_name, inward_offset_um=0.0, z_center=z_center
    )
    source_port = Port(
        center=source_center,
        size=(0.0, transverse_span, z_span),
        name="source",
        direction=source_direction,
        mode_spec=mode_spec,
    )
    frequency_width = float(np.ptp(frequencies))
    source_time = GaussianPulse(
        freq0=LIGHT_SPEED / wavelength_center,
        fwidth=frequency_width,
        offset=5.0 / (2.0 * np.pi),
    )
    source = source_port.to_source(
        freq0=source_time.freq0,
        fwidth=frequency_width,
        num_freqs=options.source_profiles
        or round(float(wavelength_span_nm) / 10.0) + 1,
        source_time=source_time,
    )
    validate_boundary_clearance(design, source, ports, options.boundary_thickness_um)
    monitors = [port.to_monitor(frequencies) for port in ports]
    if diagnostics:
        monitors.append(
            FieldMonitor(
                center=(0.5 * design.width, 0.5 * design.height, z_center),
                size=(design.width, design.height, 0.0),
                freqs=(float(np.median(frequencies)),),
                fields=("Ex", "Ey", "Ez"),
                name=f"{case.name.removeprefix('passive_soi_')}_xy",
            )
        )
    boundary = (
        Absorber(edges="all", thickness=options.boundary_thickness_um * µm)
        if protocol.get("absorber_from_ppw") is not None
        and int(resolution_ppw) >= int(protocol["absorber_from_ppw"])
        else PML(
            edges="all",
            thickness=options.boundary_thickness_um * µm,
            formulation="cpml",
        )
    )
    simulation = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[boundary],
        run_time=(
            options.run_time_ps * 1e-12
            if options.run_time_ps is not None
            else 15.0 * domain_size_um(case)[0] * µm * 2.0 / LIGHT_SPEED
        ),
        grid_spec=grid_spec,
        raster_options=RasterOptions(quality="balanced", smoothing=options.smoothing),
    )
    return simulation, ports, frequencies


def _save_four_port_artifacts(
    directory: Path,
    simulation,
    results,
    scattering,
    *,
    execution_backend: str,
    options: ExperimentOptions = DEFAULT_OPTIONS,
) -> None:
    """Persist raw and visual evidence for one four-port run."""
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    field_monitor = next(
        monitor for monitor in simulation.monitors if monitor.name.endswith("_xy")
    )
    cross_section = {
        "z": field_monitor.center[2],
        "y": field_monitor.center[1],
    }
    fig, _ = simulation.plot(
        **cross_section, source_markers=False, monitor_markers=False
    )
    fig.savefig(directory / "geometry_cross_sections.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, _ = simulation.plot(**cross_section)
    fig.savefig(directory / "simulation_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, _ = results.plot_field(
        monitor_name=field_monitor.name,
        field_name="E",
        frequency=float(np.median(scattering.frequencies)),
        val="abs^2",
    )
    fig.savefig(directory / "field_E_abs2_1550nm.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    source_time = simulation.sources[0].source_time
    source_signal, source_quadrature = source_time.sample(simulation.time)
    arrays: dict[str, np.ndarray] = {
        "frequencies_hz": np.asarray(scattering.frequencies),
        "source_time_s": np.asarray(simulation.time),
        "source_signal": np.asarray(source_signal),
        "source_quadrature": np.asarray(source_quadrature),
        "valid_mask": np.asarray(scattering.diagnostics["valid_mask"]),
        "incident_power": np.asarray(scattering.diagnostics["P_in"]),
        "guided_output_power": np.asarray(scattering.diagnostics["P_guided_out"]),
        "power_sum": np.asarray(scattering.diagnostics["power_sum"]),
        "loss_estimate": np.asarray(scattering.diagnostics["loss_est"]),
    }
    for (output, source), values in scattering.s_matrix.items():
        arrays[f"S_{output}_{source}"] = np.asarray(values)
    for port_name, wave in scattering.diagnostics["waves"].items():
        for diagnostic_name in (
            "a_plus",
            "a_minus",
            "P_plus",
            "P_minus",
            "mode_neff",
            "projection_residual",
            "condition_number",
            "projected_signed_power",
        ):
            if diagnostic_name in wave:
                arrays[f"diagnostic_{port_name}__{diagnostic_name}"] = np.asarray(
                    wave[diagnostic_name]
                )
    for port_name, flux in scattering.diagnostics["monitor_flux_checks"].items():
        for diagnostic_name in (
            "monitor_flux",
            "P_modal_sum",
            "P_modal_net",
            "P_selected",
            "P_rejected",
            "P_selected_modal_net",
        ):
            arrays[f"flux_{port_name}__{diagnostic_name}"] = np.asarray(
                flux[diagnostic_name]
            )
    for monitor_name, monitor_results in results.monitors.items():
        arrays[f"{monitor_name}__frequencies_hz"] = np.asarray(
            monitor_results.get_dft_frequencies()
        )
        for component in monitor_results.dft_fields:
            arrays[f"{monitor_name}__{component}"] = np.asarray(
                monitor_results.get_dft_component(component)
            )
    for axis in ("x", "y", "z"):
        arrays[f"grid_{axis}_boundaries_m"] = np.asarray(
            simulation.grid.axis_edges(axis)
        )
    np.savez_compressed(directory / "monitor_data.npz", **arrays)

    performance = results.performance
    termination = results.termination
    metadata = {
        "experiment_options": asdict(options),
        "run_time_s": float(simulation.run_time),
        "boundary_thickness_m": float(simulation.boundaries[0].thickness),
        "monitor_centers_m": {m.name: list(m.center) for m in simulation.monitors},
        "execution_backend": execution_backend,
        "resolution_m": float(simulation.resolution),
        "grid_shape": list(simulation.grid.shape),
        "grid_is_uniform": bool(simulation.grid.is_uniform),
        "raster_quality": simulation.raster_options.quality
        if simulation.raster_options
        else "pre_rasterized",
        "raster_smoothing": simulation.raster_options.smoothing
        if simulation.raster_options
        else simulation.material_grid.smoothing,
        "boundary_formulation": simulation.boundaries[0].formulation,
        "steps": int(performance.steps if performance else simulation.num_steps),
        "runtime_s": float(performance.runtime_s if performance else float("nan")),
        "gcups": float(performance.gcups if performance else float("nan")),
        "cells": int(performance.cells if performance else 0),
        "termination_reason": termination.reason if termination else "time_limit",
        "termination_field_decay": termination.field_decay if termination else None,
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_four_port_benchmark(
    case: DifferentialCase,
    *,
    resolution_ppw: int,
    wavelength_span_nm: float = 20.0,
    progress: bool = False,
    backend: str | None = None,
    artifact_dir: str | Path | None = None,
) -> FourPortBenchmarkResult:
    """Execute one four-port case and extract the two forward TE0 powers."""
    from beamz import LIGHT_SPEED, AutoTermination, µm
    from beamz.analysis import s_parameters

    simulation, ports, frequencies = build_four_port_simulation(
        case,
        resolution_ppw=resolution_ppw,
        wavelength_span_nm=wavelength_span_nm,
        diagnostics=artifact_dir is not None,
    )
    with reference_absorber_warning_scope():
        program = simulation.compile(progress=progress, backend=backend)
        execution_backend = program.config.backend
        results = simulation.run(
            progress=progress,
            backend=execution_backend,
            termination=AutoTermination(
                field_decay=1e-5,
                monitor_change=None,
                consecutive_checks=1,
            ),
        )
    scattering = s_parameters(
        results,
        source_port="o1",
        ports=ports,
        output_ports=["o3", "o4"],
        frequencies=frequencies,
        min_incident_db=-45.0,
    )
    if artifact_dir is not None:
        _save_four_port_artifacts(
            Path(artifact_dir),
            simulation,
            results,
            scattering,
            execution_backend=execution_backend,
        )
    wavelengths_um = LIGHT_SPEED / np.asarray(scattering.frequencies) / µm
    center = int(np.argmin(np.abs(wavelengths_um - 1.55)))
    cross_spectrum = np.abs(np.asarray(scattering.s_matrix[("o3", "o1")])) ** 2
    through_spectrum = np.abs(np.asarray(scattering.s_matrix[("o4", "o1")])) ** 2
    total_output_spectrum = cross_spectrum + through_spectrum
    excess_loss_spectrum = 1.0 - total_output_spectrum
    performance = results.performance
    termination = results.termination
    return FourPortBenchmarkResult(
        resolution_ppw=int(resolution_ppw),
        wavelength_span_nm=float(wavelength_span_nm),
        backend=execution_backend,
        wavelengths_um=tuple(float(value) for value in wavelengths_um),
        cross_power_spectrum=tuple(float(value) for value in cross_spectrum),
        through_power_spectrum=tuple(float(value) for value in through_spectrum),
        total_output_power_spectrum=tuple(
            float(value) for value in total_output_spectrum
        ),
        excess_loss_spectrum=tuple(float(value) for value in excess_loss_spectrum),
        cross_power=float(cross_spectrum[center]),
        through_power=float(through_spectrum[center]),
        total_output_power=float(total_output_spectrum[center]),
        excess_loss=float(excess_loss_spectrum[center]),
        runtime_s=float(performance.runtime_s if performance else float("nan")),
        gcups=float(performance.gcups if performance else float("nan")),
        cells=int(performance.cells if performance else 0),
        steps=int(performance.steps if performance else simulation.num_steps),
        grid_shape=tuple(int(value) for value in simulation.grid.shape),
        termination_reason=termination.reason if termination else "time_limit",
    )
