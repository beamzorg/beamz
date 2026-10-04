"""Capacity bounds require explicit GPU OOM, never arbitrary child failures."""

import json
from argparse import Namespace

import numpy as np
import pytest

from scripts import benchmark_cosine_h100 as benchmark


def options(tmp_path):
    return Namespace(
        output_dir=tmp_path,
        resolutions=[36, 30, 25, 21],
        refinements=3,
        full_resolutions=[],
        steps=256,
        samples=5,
    )


def test_capacity_refinement_preserves_success_failure_bracket(tmp_path, monkeypatch):
    attempted = []

    def trial(args, resolution):
        attempted.append(resolution)
        return {
            "status": "ok" if resolution >= 28 else "gpu_oom",
            "resolution_nm": resolution,
        }

    monkeypatch.setattr(benchmark, "trial", trial)
    benchmark.sweep(options(tmp_path))
    result = json.loads((tmp_path / "summary.json").read_text())
    assert attempted == [36, 30, 25, 27.5, 28.75, 28.125]
    assert result["last_success_nm"] >= 28 > result["first_gpu_oom_nm"]


def test_non_gpu_failure_does_not_claim_capacity(tmp_path, monkeypatch):
    attempted = []

    def trial(args, resolution):
        attempted.append(resolution)
        return {
            "status": "ok" if resolution >= 30 else "error",
            "resolution_nm": resolution,
        }

    monkeypatch.setattr(benchmark, "trial", trial)
    benchmark.sweep(options(tmp_path))
    result = json.loads((tmp_path / "summary.json").read_text())
    assert attempted == [36, 30, 25]
    assert result["first_gpu_oom_nm"] is None


def test_refinement_preserves_physical_workload_and_lifts_grid_budget():
    coarse, fine = benchmark.crossing(36), benchmark.crossing(18)
    assert coarse.grid_spec.max_total_cells is None
    assert fine.grid_spec.max_total_cells is None
    assert coarse.boundaries[0].thickness == fine.boundaries[0].thickness
    assert coarse.run_time == fine.run_time == 1e-12
    assert len(coarse.sources) == len(fine.sources) == 1
    assert len(coarse.monitors) == len(fine.monitors) == 3
    for left, right in zip(coarse.monitors, fine.monitors, strict=True):
        np.testing.assert_array_equal(left.freqs, right.freqs)
    for sim, dx in ((coarse, 36e-9), (fine, 18e-9)):
        np.testing.assert_allclose(sim.sources[0].size, (0, 1.4e-6, 0.644e-6))
        assert np.all(np.asarray(sim.size) >= np.array([13.7e-6, 13.7e-6, 1.965e-6]))
        assert np.all(
            np.asarray(sim.size) < np.array([13.7e-6, 13.7e-6, 1.965e-6]) + dx
        )


@pytest.mark.parametrize("devices, suffix", [(1, ""), (8, "-8gpu-z")])
def test_trial_refuses_stale_results(tmp_path, devices, suffix):
    target = tmp_path / f"throughput-cuda_streamed-36nm{suffix}.json"
    target.write_text('{"status": "ok", "gcups": 999}')
    args = Namespace(
        output_dir=tmp_path, backend="cuda_streamed", devices=devices, shard_axis="z"
    )
    with pytest.raises(FileExistsError, match="fresh output directory"):
        benchmark.trial(args, 36)
    assert json.loads(target.read_text())["gcups"] == 999
