"""Portable performance records with controlled-hardware regression policy."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "beamz.performance/v3"


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One reproducible compile/warm-runtime/memory measurement.

    ``warm_runtime_samples_s`` measures the already-compiled device executable.
    ``warm_end_to_end_samples_s`` additionally includes BeamZ's placement and result
    materialization.  Keeping both avoids presenting dispatch overhead as stencil
    throughput while still tracking the latency users experience.
    """

    beamz_commit: str
    beamz_version: str
    python_version: str
    jax_version: str
    jaxlib_version: str
    workload: str
    backend: str
    device: str
    device_count: int
    precision: str
    grid_dimensions: tuple[int, ...]
    timesteps: int
    boundaries: tuple[str, ...]
    sources: tuple[str, ...]
    monitors: tuple[str, ...]
    trace_lower_s: float
    compile_s: float
    warm_runtime_samples_s: tuple[float, ...]
    warm_end_to_end_samples_s: tuple[float, ...]
    peak_memory_bytes: int
    cpml_psi_precision: str = "float32"
    cuda_component_version: str | None = None
    cuda_abi_version: int | None = None
    cuda_flags: int = 0

    def __post_init__(self) -> None:
        text_fields = (
            self.beamz_commit,
            self.beamz_version,
            self.python_version,
            self.jax_version,
            self.jaxlib_version,
            self.workload,
            self.backend,
            self.device,
            self.precision,
        )
        if not all(str(value).strip() for value in text_fields):
            raise ValueError("benchmark identity fields must be non-empty")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        if self.backend not in {"jax", "cuda_streamed"}:
            raise ValueError("unknown benchmark backend")
        if self.cpml_psi_precision not in {"float32", "bfloat16"}:
            raise ValueError("unknown CPML state precision")
        if self.cuda_flags < 0:
            raise ValueError("cuda_flags must be non-negative")
        if self.backend.startswith("cuda") and (
            not self.cuda_component_version or self.cuda_abi_version is None
        ):
            raise ValueError("CUDA records require native component provenance")
        if self.backend == "jax" and (
            self.cuda_component_version is not None
            or self.cuda_abi_version is not None
            or self.cuda_flags != 0
        ):
            raise ValueError("JAX records must not claim CUDA component provenance")
        if self.device_count <= 0:
            raise ValueError("device_count must be positive")
        if len(self.grid_dimensions) not in {2, 3} or any(
            int(value) <= 0 for value in self.grid_dimensions
        ):
            raise ValueError("grid_dimensions must contain two or three positive sizes")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        durations = (
            self.trace_lower_s,
            self.compile_s,
            *self.warm_runtime_samples_s,
            *self.warm_end_to_end_samples_s,
        )
        if len(self.warm_runtime_samples_s) < 3:
            raise ValueError("at least three warm runtime samples are required")
        if len(self.warm_end_to_end_samples_s) < 3:
            raise ValueError("at least three warm end-to-end samples are required")
        if any(not math.isfinite(value) or value <= 0.0 for value in durations):
            raise ValueError("benchmark durations must be positive and finite")
        if self.peak_memory_bytes <= 0:
            raise ValueError("peak_memory_bytes must be positive")

    @property
    def material_cells(self) -> int:
        return math.prod(self.grid_dimensions)

    @property
    def median_warm_runtime_s(self) -> float:
        return float(statistics.median(self.warm_runtime_samples_s))

    @property
    def median_warm_end_to_end_s(self) -> float:
        return float(statistics.median(self.warm_end_to_end_samples_s))

    @property
    def updated_cells(self) -> int:
        """Return the conventional FDTD cell-update count for the run."""
        return self.material_cells * self.timesteps

    @property
    def kernel_gcups(self) -> float:
        return self.updated_cells / self.median_warm_runtime_s / 1e9

    @property
    def end_to_end_gcups(self) -> float:
        return self.updated_cells / self.median_warm_end_to_end_s / 1e9

    @property
    def workload_identity(self) -> tuple[Any, ...]:
        """Return the physical workload, independent of its implementation."""
        return (
            self.workload,
            self.precision,
            self.cpml_psi_precision,
            self.grid_dimensions,
            self.timesteps,
            self.boundaries,
            self.sources,
            self.monitors,
        )

    @property
    def execution_identity(self) -> tuple[Any, ...]:
        """Return implementation and hardware inputs for regression gating."""
        return (
            self.backend,
            self.device,
            self.device_count,
            self.cuda_component_version,
            self.cuda_abi_version,
            self.cuda_flags,
        )

    @property
    def comparison_identity(self) -> tuple[Any, ...]:
        return (*self.workload_identity, *self.execution_identity)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "material_cells": self.material_cells,
                "updated_cells": self.updated_cells,
                "median_warm_runtime_s": self.median_warm_runtime_s,
                "median_warm_end_to_end_s": self.median_warm_end_to_end_s,
                "kernel_gcups": self.kernel_gcups,
                "end_to_end_gcups": self.end_to_end_gcups,
            }
        )
        json.dumps(payload, allow_nan=False)
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Relative changes and optional controlled-machine gate result."""

    warm_runtime_change: float
    warm_end_to_end_change: float
    compile_change: float
    memory_change: float
    controlled_hardware: bool
    passed: bool | None


@dataclass(frozen=True, slots=True)
class BackendSpeedup:
    """Relative performance of two backends on one physical workload and machine."""

    kernel_speedup: float
    end_to_end_speedup: float
    compile_speedup: float
    memory_ratio: float


def _relative_change(current: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline benchmark values must be positive")
    return current / baseline - 1.0


def compare_benchmarks(
    baseline: BenchmarkRecord,
    current: BenchmarkRecord,
    *,
    controlled_hardware: bool,
    runtime_limit: float = 0.05,
    memory_limit: float = 0.05,
    compile_limit: float = 0.10,
) -> BenchmarkComparison:
    """Compare records; gate only when the machine is explicitly controlled."""
    if baseline.comparison_identity != current.comparison_identity:
        raise ValueError("benchmark records do not describe the same workload")
    runtime_change = _relative_change(
        current.median_warm_runtime_s,
        baseline.median_warm_runtime_s,
    )
    compile_change = _relative_change(current.compile_s, baseline.compile_s)
    end_to_end_change = _relative_change(
        current.median_warm_end_to_end_s,
        baseline.median_warm_end_to_end_s,
    )
    memory_change = _relative_change(
        float(current.peak_memory_bytes),
        float(baseline.peak_memory_bytes),
    )
    passed = (
        runtime_change <= runtime_limit
        and end_to_end_change <= runtime_limit
        and memory_change <= memory_limit
        and compile_change <= compile_limit
        if controlled_hardware
        else None
    )
    return BenchmarkComparison(
        warm_runtime_change=runtime_change,
        warm_end_to_end_change=end_to_end_change,
        compile_change=compile_change,
        memory_change=memory_change,
        controlled_hardware=controlled_hardware,
        passed=passed,
    )


def compare_backend_speedup(
    baseline: BenchmarkRecord,
    candidate: BenchmarkRecord,
) -> BackendSpeedup:
    """Compare backend implementations while holding physics and hardware fixed."""
    if baseline.workload_identity != candidate.workload_identity:
        raise ValueError("benchmark records do not describe the same physical workload")
    if (baseline.device, baseline.device_count) != (
        candidate.device,
        candidate.device_count,
    ):
        raise ValueError("backend speedup requires the same hardware and device count")
    return BackendSpeedup(
        kernel_speedup=baseline.median_warm_runtime_s / candidate.median_warm_runtime_s,
        end_to_end_speedup=baseline.median_warm_end_to_end_s
        / candidate.median_warm_end_to_end_s,
        compile_speedup=baseline.compile_s / candidate.compile_s,
        memory_ratio=candidate.peak_memory_bytes / baseline.peak_memory_bytes,
    )
