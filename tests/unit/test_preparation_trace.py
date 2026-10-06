import json
import runpy
import sys
from pathlib import Path

import pytest

from beamz.simulation.preparation_trace import preparation_phase


def test_nested_preparation_trace_records_failure(tmp_path, monkeypatch):
    path = tmp_path / "preparation.jsonl"
    monkeypatch.setenv("BEAMZ_TRACE_PREPARATION", str(path))
    with pytest.raises(ValueError), preparation_phase("outer"):
        with preparation_phase("inner"):
            pass
        raise ValueError("failed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["phase"], r["event"]) for r in rows] == [
        ("outer", "start"),
        ("inner", "start"),
        ("inner", "end"),
        ("outer", "error"),
    ]
    assert rows[-1]["error_type"] == "ValueError"
    if sys.platform == "win32":
        assert rows[-1]["host_peak_bytes"] is None
    else:
        assert rows[-1]["host_peak_bytes"] > 0
    assert rows[-1]["elapsed_s"] >= 0


def test_trace_import_and_timings_without_resource(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)
    module = runpy.run_path(
        str(Path(__file__).parents[2] / "beamz/simulation/preparation_trace.py")
    )
    monkeypatch.delenv("BEAMZ_TRACE_PREPARATION", raising=False)
    with module["preparation_phase"]("disabled"):
        pass

    path = tmp_path / "preparation.jsonl"
    monkeypatch.setenv("BEAMZ_TRACE_PREPARATION", str(path))
    with module["preparation_phase"]("enabled"):
        pass
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "end"]
    assert all(row["host_peak_bytes"] is None for row in rows)
    assert all(row["host_rss_bytes"] is None for row in rows)
    assert rows[-1]["elapsed_s"] >= 0
