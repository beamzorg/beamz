#!/usr/bin/env python3
"""Gate private grouped queues, coefficient fallbacks and fail-closed inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cuda_native_runtime import logical_native_state
from scripts.export_cuda_native_fixture import buffer_digest
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_hybrid_cpml import exact_state


def axis_partition(rows, shapes, thickness):
    """Independent NumPy coordinate classification, preserving order per bucket."""
    extent = np.max(shapes, axis=0)
    minimum = np.min(shapes, axis=0)
    coordinates = np.unravel_index(rows, tuple(extent))
    masks = np.zeros(rows.size, np.int8)
    for axis, coordinate in enumerate(coordinates):
        masks |= (
            (coordinate < thickness) | (coordinate >= minimum[axis] - thickness)
        ).astype(np.int8) << axis
    counts = np.bincount(masks, minlength=8)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int32)
    ordered = np.concatenate([rows[masks == mask] for mask in range(8)])
    if not np.array_equal(np.sort(ordered), rows):
        raise AssertionError("Reference partition changed point set")
    return offsets, ordered


def exact_rank2(inputs, component):
    result = list(inputs)
    table = inputs[37 + component]
    words = inputs[40 + component].view(np.uint32).reshape(-1)
    count = inputs[3 + component].size
    ids = ((words[:, None] >> (8 * np.arange(4, dtype=np.uint32))) & 255).reshape(-1)[
        :count
    ]
    blocks = (count + 31) // 32
    dense = np.zeros(blocks * 32, np.float32)
    dense[:count] = table[ids]
    result[37 + component] = dense
    result[40 + component] = (np.arange(blocks, dtype=np.int32) * 32).reshape(1, -1)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError("Requires JAX absent")
    spec = importlib.util.spec_from_file_location("_cuda_native", a.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    f = json.loads(a.fixture.read_text())
    data = np.load(a.fixture.parent / f["data_file"], allow_pickle=False)
    original = [data[x["key"]] for x in f["inputs"]]
    expected = {x["name"]: data[x["reference_key"]] for x in f["state_leaves"]}
    shapes = np.array([x.shape for x in original[:6]])
    thickness = f["attributes"]["boundary_code"] >> 8
    expected_offsets = [
        axis_partition(original[132 + q], shapes, thickness)[0] for q in range(2)
    ]
    variants = [("normal", original)]
    allrank = original
    for c in range(3):
        allrank = exact_rank2(allrank, c)
    variants += [("all_rank2", allrank), ("mixed_rank2", exact_rank2(original, 0))]
    dense = list(original)
    dense[9] = np.full(original[0].shape, original[9].item(), np.float32)
    variants.append(("dense_H_fallback", dense))
    rows = []
    for name, inputs in variants:
        before = [buffer_digest(x) for x in inputs]
        session = module.NativeSession(inputs, f["attributes"], preserve_inputs=True)
        try:
            result = session.run_grouped([7])
            actual = logical_native_state(
                f, data, session.download(), session.clocks(), result["output_bank"]
            )
            parity = exact_state(actual, expected)
            metadata = session.metadata()
            offsets = all(
                metadata["grouped_" + label + "_offsets"]
                == expected_offsets[q].tolist()
                for q, label in enumerate(["first", "edge"])
            )
            rows.append(
                {
                    "case": name,
                    "parity": parity,
                    "offsets_match_independent_numpy": offsets,
                    "host_unchanged": before == [buffer_digest(x) for x in inputs],
                    "zero_extra_gpu_bytes": metadata["grouped_extra_gpu_bytes"] == 0,
                    "metadata": metadata,
                }
            )
        finally:
            session.close()
    # Arbitrary PEC configurations take the existing full-edge fallback. Compare
    # directly against an ordinary native run with the same changed physical BC.
    attrs = {**f["attributes"], "boundary_code": f["attributes"]["boundary_code"] | 63}
    values = []
    for grouped in [False, True]:
        session = module.NativeSession(original, attrs, preserve_inputs=True)
        try:
            result = (session.run_grouped if grouped else session.run_schedule)([7])
            values.append(
                logical_native_state(
                    f, data, session.download(), session.clocks(), result["output_bank"]
                )
            )
        finally:
            session.close()
    rows.append(
        {
            "case": "PEC_fallback_same_physics_control",
            "parity": exact_state(values[1], values[0]),
        }
    )
    negatives = []
    for name, modify in [
        ("duplicate", lambda x: x.__setitem__(1, x[0])),
        ("reversed", lambda x: x.__setitem__(slice(None), x[::-1].copy())),
        (
            "out_of_range",
            lambda x: x.__setitem__(-1, int(np.prod(np.max(shapes, axis=0)))),
        ),
    ]:
        inputs = list(original)
        inputs[132] = np.array(original[132], copy=True)
        modify(inputs[132])
        session = module.NativeSession(inputs, f["attributes"], preserve_inputs=True)
        try:
            try:
                session.run_grouped([7])
            except ValueError as error:
                negatives.append(
                    {
                        "case": name,
                        "passed": "sorted unique" in str(error),
                        "error": str(error),
                        "ready": session.metadata()["grouped_axes_ready"],
                    }
                )
            else:
                negatives.append({"case": name, "passed": False})
        finally:
            session.close()
    for first in ["grouped", "control"]:
        session = module.NativeSession(original, f["attributes"], preserve_inputs=False)
        try:
            (session.run_grouped if first == "grouped" else session.run_schedule)([3])
            before = session.clocks()
            try:
                (session.run_schedule if first == "grouped" else session.run_grouped)(
                    [1]
                )
            except ValueError as error:
                negatives.append(
                    {
                        "case": first + "_mode_switch",
                        "passed": "queue order is fixed" in str(error)
                        and session.clocks() == before,
                        "error": str(error),
                    }
                )
            else:
                negatives.append({"case": first + "_mode_switch", "passed": False})
        finally:
            session.close()
    passed = all(
        r["parity"]["all30_bitwise"]
        and r.get("offsets_match_independent_numpy", True)
        and r.get("host_unchanged", True)
        and r.get("zero_extra_gpu_bytes", True)
        for r in rows
    ) and all(r["passed"] for r in negatives)
    report = {
        "schema": "beamz.grouped-axis-gates.v1",
        "module": str(a.module),
        "module_sha256": file_digest(a.module),
        "fixture": str(a.fixture),
        "fixture_sha256": file_digest(a.fixture),
        "worker_sha256": file_digest(__file__),
        "parity_cases": rows,
        "negative_cases": negatives,
        "passed": passed,
        "queue_set_evidence": "Independent NumPy stable partition is an exact permutation of canonical sorted rows; nativeoffsetcounts match; fullstatebitwise checks exerciseactualdeviceexecution. DirectprivateGPUqueue readback not implemented.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": passed, "cases": len(rows), "negatives": len(negatives)},
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
