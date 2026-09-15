"""Geometry, simulation, and observables for the passive-SOI directional coupler."""

from pathlib import Path

from tests.differential.passive_soi.common import load_passive_soi_case
from tests.differential.passive_soi.four_port import (
    FourPortBenchmarkResult as DirectionalCouplerBenchmarkResult,
)
from tests.differential.passive_soi.four_port import (
    build_four_port_simulation,
    paper_cross_power_range,
    run_four_port_benchmark,
)

__all__ = [
    "DirectionalCouplerBenchmarkResult",
    "build_directional_coupler_simulation",
    "paper_cross_power_range",
    "run_directional_coupler_benchmark",
]


def build_directional_coupler_simulation(
    *, resolution_ppw: int, wavelength_span_nm: float = 20.0, diagnostics: bool = False
):
    """Build the paper-matched simulation without executing it."""
    return build_four_port_simulation(
        load_passive_soi_case("directional_coupler"),
        resolution_ppw=resolution_ppw,
        wavelength_span_nm=wavelength_span_nm,
        diagnostics=diagnostics,
    )


def run_directional_coupler_benchmark(
    *,
    resolution_ppw: int,
    wavelength_span_nm: float = 20.0,
    progress: bool = False,
    backend: str | None = None,
    artifact_dir: str | Path | None = None,
) -> DirectionalCouplerBenchmarkResult:
    """Execute the benchmark and extract forward TE0 powers."""
    return run_four_port_benchmark(
        load_passive_soi_case("directional_coupler"),
        resolution_ppw=resolution_ppw,
        wavelength_span_nm=wavelength_span_nm,
        progress=progress,
        backend=backend,
        artifact_dir=artifact_dir,
    )
