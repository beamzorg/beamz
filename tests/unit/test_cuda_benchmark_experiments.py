"""Accounting checks for the CUDA profiler artifact reader."""

import gzip
import json

import pytest

from scripts.profile_cuda_workload import summarize_trace


def test_profile_accounting_excludes_host_events_and_preserves_kernel_cost(tmp_path):
    path = tmp_path / "trace.json.gz"
    events = [
        {"ph": "X", "name": "host work", "dur": 9000, "args": {}},
        {
            "ph": "X",
            "name": "MstepEdgePhase",
            "dur": 300,
            "args": {"kernel_details": "edge"},
        },
        {
            "ph": "X",
            "name": "MstepPackedColumnCore",
            "dur": 100,
            "args": {"kernel_details": "core"},
        },
        {"ph": "X", "name": "Memcpy D2D", "dur": 1000, "args": {}},
    ]
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    result = summarize_trace(path)
    assert result["total_kernel_s"] == 0.0004
    assert [(r["percent"], r["count"]) for r in result["groups"]] == [(75, 1), (25, 1)]
    assert result["top_kernels"][0]["details"] == "edge"


def test_cpu_only_profile_is_not_reported_as_gpu_evidence(tmp_path):
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": [{"ph": "X", "name": "host", "dur": 100}]}, stream)
    with pytest.raises(ValueError, match="No GPU kernel events"):
        summarize_trace(path)


def test_compact_edge_stage_assignment_requires_verified_launch_sequence(tmp_path):
    path = tmp_path / "trace.json.gz"
    events = [
        {
            "ph": "X",
            "name": f"MstepEdgePhaseILi{phase}ELb1EEv",
            "ts": index,
            "dur": duration,
            "args": {"kernel_details": "edge"},
        }
        for index, (phase, duration) in enumerate(
            zip((0, 1, 0, 1), (100, 200, 300, 400), strict=True)
        )
    ]
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    result = summarize_trace(path, steps=2)
    groups = {row["group"]: row["duration_us"] for row in result["groups"]}
    assert groups == {"first-step compact edges": 300, "second-step compact edges": 700}
    assert result["compact_first_second_assignment"].startswith("Verified")
    events[1]["name"] = "MstepEdgePhaseILi0ELb1EEv"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    result = summarize_trace(path, steps=2)
    assert result["compact_first_second_assignment"] == "Not assigned"


def test_full_edge_templates_are_never_hidden_in_other_kernels(tmp_path):
    path = tmp_path / "trace.json.gz"
    names = [
        "MstepFullEdgePhaseILi0ELb1ELb0ELb1ELb1EEv",
        "MstepFullEdgePhaseILi1ELb1ELb1ELb1ELb0EEv",
        "MstepFullEdgePhase_unknown_template",
    ]
    events = [
        {"ph": "X", "name": name, "dur": 100, "args": {"kernel_details": "edge"}}
        for name in names
    ]
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    result = summarize_trace(path)
    assert {row["group"] for row in result["groups"]} == {
        "first-step full edges and core copies",
        "full edges (stage unassigned)",
        "full edges (unparsed template)",
    }


@pytest.mark.parametrize("suffix", ["Ev", "Li0ELb0EEv", "Lin1ELb1EEv"])
def test_full_edges_without_copycore_still_split_by_verified_step_order(
    tmp_path, suffix
):
    path = tmp_path / "trace.json.gz"
    events = [
        {
            "ph": "X",
            "name": f"MstepFullEdgePhaseILi{phase}ELb1ELb0ELb1ELb0E{suffix}",
            "ts": i,
            "dur": duration,
            "args": {"kernel_details": "full edge"},
        }
        for i, (phase, duration) in enumerate(
            zip((0, 1, 0, 1), (10, 20, 30, 40), strict=True)
        )
    ]
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    result = summarize_trace(path, steps=2)
    assert {r["group"]: r["duration_us"] for r in result["groups"]} == {
        "first-step full edges": 30,
        "second-step full edges": 70,
    }
