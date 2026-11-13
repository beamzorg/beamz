"""Promotion must reject changed physics, precision, ownership and partial output."""
import copy
import json

import pytest

from scripts.cuda_research_compare import compare, digest


def make_run(tmp_path, label, *, flags=128, dt=0.25, dtype="<f4", ownership="preserve"):
    manifest = tmp_path / "arrays.json"
    manifest.write_text('{"same_immutable_packet": true}')
    state = [{"name": f"leaf{i}", "shape": [1], "dtype": dtype,
              "reference_key": f"ref{i}"} for i in range(30)]
    fixture = {"attributes": {"cuda_flags": flags, "nsteps": 8, "dt": dt,
                              "cpml_enabled": 1},
               "data_file": manifest.name, "data_sha256": digest(manifest),
               "state_leaves": state}
    fp = tmp_path / f"{label}-fixture.json"
    fp.write_text(json.dumps(fixture))
    parity = {"passed": True, "exact": True, "bitwise_exact": True,
              "leaves": [{**{k: x[k] for k in ("name", "shape", "dtype")},
                          "passed": True, "bitwise_exact": True} for x in state]}
    run = {"status": "ok", "host_inputs_unchanged": True, "fixture": str(fp),
           "fixture_sha256": digest(fp), "ownership": ownership, "chunks": [4, 4],
           "median_wall_s": 1.0, "parity": parity, "repeat_parity": copy.deepcopy(parity)}
    rp = tmp_path / f"{label}.json"
    rp.write_text(json.dumps(run))
    return rp


def test_launch_change_allowed_but_physics_precision_and_ownership_rejected(tmp_path):
    base = make_run(tmp_path, "base")
    launch = make_run(tmp_path, "launch", flags=128 | (1 << 24))
    assert compare([base], [launch])["median_paired_speedup"] == 1
    for label, kwargs in [("clock", {"dt": .125}), ("bf16", {"flags": 128 | 64}),
                          ("dtype", {"dtype": "<f8"}), ("owned", {"ownership": "owned"})]:
        other = make_run(tmp_path, label, **kwargs)
        with pytest.raises(ValueError, match="contract mismatch"):
            compare([base], [other])


@pytest.mark.parametrize("damage", ["missing_leaf", "wrong_name", "short_schedule", "nonfinite"])
def test_invalid_timing_or_partial_validation_never_promoted(tmp_path, damage):
    base = make_run(tmp_path, "base")
    other = make_run(tmp_path, "other")
    report = json.loads(other.read_text())
    if damage == "missing_leaf":
        report["parity"]["leaves"].pop()
    elif damage == "wrong_name":
        report["repeat_parity"]["leaves"][0]["name"] = "unrelated"
    elif damage == "short_schedule":
        report["chunks"] = [4]
    else:
        report["median_wall_s"] = float("nan")
    other.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare([base], [other])


def test_explicit_control_rejects_mislabeled_or_replaced_binary(tmp_path):
    baseline = make_run(tmp_path, "baseline")
    candidate = make_run(tmp_path, "candidate")
    module = tmp_path / "control.so"
    module.write_bytes(b"verified control binary")
    expected = digest(module)
    report = json.loads(baseline.read_text())
    report.update(module=str(module), module_sha256=expected)
    baseline.write_text(json.dumps(report))
    result = compare([baseline], [candidate], expected_control_sha256=expected)
    assert result["control_identity_verified"]
    assert result["expected_control_sha256"] == expected
    with pytest.raises(ValueError, match="Control module SHA mismatch"):
        compare([baseline], [candidate], expected_control_sha256="0" * 64)
    module.write_bytes(b"a different binary at the same path")
    with pytest.raises(ValueError, match="Control binary changed or missing"):
        compare([baseline], [candidate], expected_control_sha256=expected)


def rolling_run(tmp_path, label, *, depth=2, changed=None):
    path = make_run(tmp_path, label)
    report = json.loads(path.read_text())
    fixture_path = tmp_path / f"{label}-fixture.json"
    fixture = json.loads(fixture_path.read_text())
    fixture["attributes"]["rolling_plan_depth"] = depth
    fixture["data_format"] = "npy_directory_v1"
    names = {129: "plan_bounds", 130: "plan_core_masks",
             132: "plan_first_cells", 133: "plan_edge_cells"}
    arrays = {}
    fixture["inputs"] = []
    for i in range(134):
        value = {"shape": [1], "dtype": "<f4", "sha256": f"{i:064x}"}
        if i == changed:
            value["sha256"] = "f" * 64
        key = f"input_{i}"
        arrays[key] = value
        fixture["inputs"].append({"index": i, "name": names.get(i, f"input{i}"),
                                   "key": key, **value})
    for i, leaf in enumerate(fixture["state_leaves"]):
        leaf["initial_key"] = f"initial{i}"
        for key in (leaf["initial_key"], leaf["reference_key"]):
            arrays[key] = {"shape": [1], "dtype": "<f4", "sha256": "a" * 64}
    manifest = tmp_path / f"{label}-arrays.json"
    manifest.write_text(json.dumps({"arrays": arrays}))
    fixture.update(data_file=manifest.name, data_sha256=digest(manifest))
    fixture_path.write_text(json.dumps(fixture))
    report["fixture_sha256"] = digest(fixture_path)
    path.write_text(json.dumps(report))
    return path


def test_d4_only_operational_plan_changes_are_explicitly_admitted(tmp_path):
    base = rolling_run(tmp_path, "base")
    candidate = rolling_run(tmp_path, "d4", depth=4, changed=132)
    with pytest.raises(ValueError, match="contract mismatch"):
        compare([base], [candidate])
    assert compare([base], [candidate], allow_rolling4_plan=True)["median_paired_speedup"] == 1
    for i in (0, 6, 74, 113, 131):
        other = rolling_run(tmp_path, f"damage{i}", depth=4, changed=i)
        with pytest.raises(ValueError, match="contract mismatch"):
            compare([base], [other], allow_rolling4_plan=True)


def test_d4_reference_bytes_must_still_be_identical(tmp_path):
    base = rolling_run(tmp_path, "base")
    candidate = rolling_run(tmp_path, "d4", depth=4, changed=132)
    report = json.loads(candidate.read_text())
    fp = tmp_path / "d4-fixture.json"
    fixture = json.loads(fp.read_text())
    manifest = tmp_path / fixture["data_file"]
    data = json.loads(manifest.read_text())
    data["arrays"]["ref0"]["sha256"] = "b" * 64
    manifest.write_text(json.dumps(data))
    fixture["data_sha256"] = digest(manifest)
    fp.write_text(json.dumps(fixture))
    report["fixture_sha256"] = digest(fp)
    candidate.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="contract mismatch"):
        compare([base], [candidate], allow_rolling4_plan=True)
