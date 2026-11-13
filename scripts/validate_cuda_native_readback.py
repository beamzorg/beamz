#!/usr/bin/env python3
"""GPU acceptance gate for bounded native readback; use a small mmap fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import ArrayReader, compare_arrays, digest_array
from scripts.export_fdtdx_comparison import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError("Run with JAX absent")
    f = json.loads(a.fixture.read_text())
    store = ArrayReader(a.fixture.parent / f["data_directory"])
    inputs = [store[x["key"]] for x in f["inputs"]]
    if sum(x.nbytes for x in inputs[:6]) > 256 << 20:
        raise MemoryError("This bulk-API equivalence smoke requires a small fixture")
    before = [digest_array(x) for x in inputs]
    assert before == [x["sha256"] for x in f["inputs"]]
    spec = importlib.util.spec_from_file_location("_cuda_native", a.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    work = a.output.parent / (a.output.stem + ".readbacks")
    work.mkdir(exist_ok=True)
    session = module.NativeSession(inputs, f["attributes"], preserve_inputs=True)
    try:
        result = session.run_schedule([f["attributes"]["nsteps"]])
        old = session.download()
        rows = []
        for index, reference in enumerate(old):
            dest = np.lib.format.open_memmap(
                work / f"output-{index}.npy",
                mode="w+",
                dtype=reference.dtype,
                shape=reference.shape,
            )
            session.download_into(index, dest)
            exact = np.array_equal(
                dest.reshape(-1).view(np.uint8), reference.reshape(-1).view(np.uint8)
            )
            assert exact, index
            dest.flush()
            rows.append(
                {"index": index, "old_API_bytes_equal": True, "bytes": dest.nbytes}
            )
        leaves = []
        for leaf in f["state_leaves"]:
            m = leaf["mapping"]
            if m["kind"] == "field":
                value = old[(18 if result["output_bank"] else 0) + m["component"]]
            elif m["kind"] == "output":
                value = old[m["index"]]
            elif m["kind"] == "clock":
                value = np.asarray(
                    session.clocks()["time" if m["name"] == "t" else "current_step"],
                    dtype=np.dtype(leaf["dtype"]),
                )
            else:
                value = store[leaf["initial_key"]]
            check = compare_arrays(value, store[leaf["reference_key"]])
            assert check["passed"], leaf["name"]
            leaves.append(dict(check, name=leaf["name"]))
        ref = old[0]
        sentinel = np.full(ref.shape, 123.25, np.float32)
        wrong_shape = np.empty((ref.size,), np.float32)
        wrong_type = np.empty(ref.shape, np.int32)
        strided = np.empty((*ref.shape[:-1], ref.shape[-1] * 2), np.float32)[..., ::2]
        readonly = np.empty(ref.shape, np.float32)
        readonly.flags.writeable = False
        negatives = []
        for label, index, dest in [
            ("index-negative", -1, sentinel),
            ("index-past-end", len(old), sentinel),
            ("shape", 0, wrong_shape),
            ("dtype", 0, wrong_type),
            ("noncontiguous", 0, strided),
            ("readonly", 0, readonly),
            ("not-array", 0, []),
        ]:
            try:
                session.download_into(index, dest)
            except (ValueError, TypeError) as error:
                negatives.append({"case": label, "error": str(error)})
            else:
                raise AssertionError(f"Must reject {label}")
        assert np.all(sentinel == np.float32(123.25))

        def threaded(index):
            dest = np.empty_like(old[index])
            session.download_into(index, dest)
            return np.array_equal(dest, old[index])

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert all(pool.map(threaded, [0, 1]))
        assert before == [digest_array(x) for x in inputs]
    finally:
        session.close()
    try:
        session.download_into(0, sentinel)
    except RuntimeError as error:
        negatives.append({"case": "closed", "error": str(error)})
    else:
        raise AssertionError("Closed session must reject readback")
    report = {
        "status": "ok",
        "module_sha256": file_digest(a.module),
        "fixture_sha256": file_digest(a.fixture),
        "raw_outputs": rows,
        "all30": leaves,
        "negative_controls": negatives,
        "two_thread_serialization": True,
        "host_inputs_unchanged": True,
        "jax_absent": True,
    }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "ok",
                "raw_outputs": len(rows),
                "state_leaves": len(leaves),
                "negative_controls": len(negatives),
            }
        )
    )


if __name__ == "__main__":
    main()
