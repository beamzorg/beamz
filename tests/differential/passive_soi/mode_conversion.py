"""Modal conversion benchmarks using the paper's pinned silicon layouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tests.differential.passive_soi.common import (
    load_passive_soi_case,
    reference_absorber_warning_scope,
)
from tests.differential.passive_soi.four_port import (
    _save_four_port_artifacts,
    build_four_port_simulation,
)

DEVICES = ("mode_converter", "polarization_splitter_rotator")


@dataclass(frozen=True)
class ModeConversionResult:
    resolution_ppw: int
    backend: str
    wavelengths_um: tuple[float, ...]
    conversion_spectrum: tuple[float, ...]
    crosstalk_spectrum: tuple[float, ...]
    selected_output_spectrum: tuple[float, ...]
    conversion_power: float
    crosstalk_power: float
    selected_output_power: float
    runtime_s: float
    cells: int
    steps: int
    grid_shape: tuple[int, int, int]
    termination_reason: str


def build_mode_conversion_simulation(
    name: str, *, resolution_ppw: int = 6, diagnostics: bool = False
):
    """Build physical monitors and separate conversion/crosstalk projections."""
    from beamz import ModeSpec

    if name not in DEVICES:
        raise ValueError(f"unsupported mode conversion device {name!r}")
    case = load_passive_soi_case(name)
    protocol = case.geometry["simulation"]
    simulation, physical_ports, frequencies = build_four_port_simulation(
        case, resolution_ppw=resolution_ppw, diagnostics=diagnostics
    )
    ports = {port.name: port for port in physical_ports}
    target = ports[protocol["conversion_port"]]
    # Both projections read the same DFT aperture. Five candidates match the
    # reference setup and allow polarization ordering to distinguish TE1 from TM0.
    projections = tuple(
        replace(
            target,
            name=channel,
            monitor_name=target.name,
            mode_spec=ModeSpec(
                polarization=protocol[f"{channel}_polarization"],
                mode_index=protocol[f"{channel}_mode_index"],
                num_modes=int(protocol["mode_candidates"]),
            ),
        )
        for channel in ("conversion", "crosstalk")
    )
    analysis_ports = (
        tuple(port for port in physical_ports if port != target) + projections
    )
    outputs = tuple(
        port for port in protocol["output_ports"] if port != target.name
    ) + ("conversion", "crosstalk")
    return simulation, analysis_ports, outputs, frequencies


def run_mode_conversion_benchmark(
    name: str,
    *,
    resolution_ppw: int = 6,
    progress: bool = False,
    backend: str | None = None,
    artifact_dir: str | Path | None = None,
) -> ModeConversionResult:
    """Measure converted power and residual mode power from one broadband run."""
    from beamz import LIGHT_SPEED, AutoTermination, µm
    from beamz.analysis import s_parameters

    simulation, ports, outputs, frequencies = build_mode_conversion_simulation(
        name, resolution_ppw=resolution_ppw, diagnostics=artifact_dir is not None
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
        output_ports=outputs,
        frequencies=frequencies,
        min_incident_db=-45.0,
    )
    if not np.all(scattering.diagnostics["valid_mask"]):
        raise ValueError("conversion spectrum contains samples with no incident signal")
    if artifact_dir is not None:
        _save_four_port_artifacts(
            Path(artifact_dir),
            simulation,
            results,
            scattering,
            execution_backend=execution_backend,
        )
    wavelengths = LIGHT_SPEED / np.asarray(scattering.frequencies) / µm
    powers = {key: np.abs(value) ** 2 for key, value in scattering.s_matrix.items()}
    conversion = powers[("conversion", "o1")]
    crosstalk = powers[("crosstalk", "o1")]
    selected_output = sum(powers.values())
    center = int(np.argmin(np.abs(wavelengths - 1.55)))
    performance = results.performance
    termination = results.termination
    return ModeConversionResult(
        resolution_ppw=resolution_ppw,
        backend=execution_backend,
        wavelengths_um=tuple(float(v) for v in wavelengths),
        conversion_spectrum=tuple(float(v) for v in conversion),
        crosstalk_spectrum=tuple(float(v) for v in crosstalk),
        selected_output_spectrum=tuple(float(v) for v in selected_output),
        conversion_power=float(conversion[center]),
        crosstalk_power=float(crosstalk[center]),
        selected_output_power=float(selected_output[center]),
        runtime_s=float(performance.runtime_s),
        cells=int(performance.cells),
        steps=int(performance.steps),
        grid_shape=tuple(simulation.grid.shape),
        termination_reason=termination.reason if termination else "time_limit",
    )
