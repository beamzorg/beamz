from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from scripts.rtx3090_benchmark import (
    BackendMeasurement,
    RTX3090Comparison,
    RTX3090Matrix,
    source_environment,
    summarize_timings,
    write_matrix_artifacts,
    write_report_artifacts,
)


def test_source_environment_prioritizes_both_checkouts_and_disables_cache(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PYTHONPATH", "/unrelated")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/cached")
    env = source_environment(tmp_path / "source", tmp_path / "runner")
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "source"),
        str(tmp_path / "runner"),
        "/unrelated",
    ]
    assert env["BEAMZ_DISABLE_JAX_PERSISTENT_CACHE"] == "1"
    assert "JAX_COMPILATION_CACHE_DIR" not in env


def _measurement(label: str, samples: tuple[float, ...]) -> BackendMeasurement:
    return BackendMeasurement(
        label=label,
        revision="abc123",
        backend="jax" if label == "origin/main JAX/XLA" else "cuda_streamed",
        device="NVIDIA GeForce RTX 3090",
        grid_zyx=(32, 48, 64),
        timesteps=20,
        trace_lower_s=0.1,
        compile_s=1.2,
        warm_runtime_samples_s=samples,
        profile="heterogeneous_cpml",
        field_precision="float32",
        cpml_psi_precision="float32",
        python_version="3.11.15",
        jax_version="0.9.0",
        jaxlib_version="0.9.0",
        beamz_version="0.4.3",
        cuda_component_version=(None if label == "origin/main JAX/XLA" else "0.12.0"),
        cuda_abi_version=None if label == "origin/main JAX/XLA" else 12,
        cuda_flags=0 if label == "origin/main JAX/XLA" else 128,
        driver_version="610.43.03",
        cuda_version="12.9",
    )


def test_timing_summary_reports_distribution_and_deterministic_interval():
    summary = summarize_timings((1.0, 1.1, 1.2, 1.3, 1.4))

    assert summary.count == 5
    assert summary.minimum_s == pytest.approx(1.0)
    assert summary.p25_s == pytest.approx(1.1)
    assert summary.median_s == pytest.approx(1.2)
    assert summary.p75_s == pytest.approx(1.3)
    assert summary.maximum_s == pytest.approx(1.4)
    assert summary.median_ci95_low_s <= summary.median_s <= summary.median_ci95_high_s
    assert summary.coefficient_of_variation > 0.0


def test_measurement_rejects_insufficient_or_invalid_samples():
    with pytest.raises(ValueError, match="three warm"):
        _measurement("origin/main JAX/XLA", (1.0, 1.1))
    with pytest.raises(ValueError, match="positive finite"):
        _measurement("origin/main JAX/XLA", (1.0, 0.0, 1.1))


def test_report_writes_machine_readable_statistics_markdown_and_graph(tmp_path):
    report = RTX3090Comparison(
        _measurement("origin/main JAX/XLA", (2.0, 2.1, 2.2, 2.1, 2.0)),
        _measurement("PR CUDA streamed", (1.0, 1.1, 1.0, 1.1, 1.0)),
    )

    paths = write_report_artifacts(report, tmp_path)
    payload = json.loads(paths["json"].read_text())

    assert payload["schema_version"] == "beamz.performance/rtx3090-v2"
    assert payload["cuda"]["cuda_component_version"] == "0.12.0"
    assert payload["cuda"]["cuda_abi_version"] == 12
    assert payload["cuda"]["cuda_flags"] == 128
    assert payload["cuda"]["cpml_psi_precision"] == "float32"
    assert payload["runtime_speedup"] == pytest.approx(2.1)
    assert payload["runtime_speedup_ci95"][0] > 1.0
    assert payload["runtime_speedup_ci95"][0] <= payload["runtime_speedup"]
    assert payload["runtime_speedup_ci95"][1] >= payload["runtime_speedup"]
    assert payload["cuda_is_faster"]
    assert "Custom CUDA speedup: **2.100×**" in paths["markdown"].read_text()
    assert paths["graph"].read_bytes().startswith(b"\x89PNG")


def test_comparison_rejects_unlabeled_mixed_precision():
    baseline = _measurement("origin/main JAX/XLA", (2.0, 2.1, 2.2))
    cuda = replace(
        _measurement("PR CUDA streamed", (1.0, 1.1, 1.2)),
        cpml_psi_precision="bfloat16",
    )

    with pytest.raises(ValueError, match="same numerical precision"):
        RTX3090Comparison(baseline, cuda)


def test_matrix_report_writes_profile_statistics_and_graph(tmp_path):
    comparison = RTX3090Comparison(
        _measurement("origin/main JAX/XLA", (2.0, 2.1, 2.2, 2.1, 2.0)),
        _measurement("PR CUDA streamed", (1.0, 1.1, 1.0, 1.1, 1.0)),
    )
    matrix = RTX3090Matrix(
        (("uniform_pec", comparison), ("heterogeneous_cpml", comparison))
    )

    paths = write_matrix_artifacts(matrix, tmp_path)
    payload = json.loads(paths["json"].read_text())

    assert payload["schema_version"] == "beamz.performance/rtx3090-matrix-v2"
    assert payload["profiles"]["uniform_pec"]["runtime_speedup"] == pytest.approx(2.1)
    assert "heterogeneous_cpml" in paths["markdown"].read_text()
    assert paths["graph"].read_bytes().startswith(b"\x89PNG")
