import json

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
    assert rows[-1]["host_peak_bytes"] > 0
    assert rows[-1]["elapsed_s"] >= 0
