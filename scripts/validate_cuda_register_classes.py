"""Check admitted/fallback H coefficient classes against ordinary CUDA steps.

Unlike fixture-preserving tests, these deliberately alter the physical H
coefficients. Each altered packet gets an independent ordinary-step reference;
the unchanged-physics reference is never reused for a changed material.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cuda_native_runtime import logical_native_state
from scripts.cuda_fixture_store import ArrayReader, compare_arrays, digest_array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError("Use the native-only interpreter")
    spec = importlib.util.spec_from_file_location("_cuda_native", a.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = json.loads(a.fixture.read_text())
    data = (ArrayReader(a.fixture.parent / fixture["data_directory"])
            if fixture.get("data_format") == "npy_directory_v1" else
            np.load(a.fixture.parent / fixture["data_file"], allow_pickle=False))
    original = [data[x["key"]] for x in fixture["inputs"]]
    if any(x.shape != () or x.dtype != np.dtype("float32") for x in original[6:12]):
        raise ValueError("Coefficient-class gate requires scalar FP32 H fixture inputs")
    rows = []
    for name in ("lossless_isotropic", "anisotropic_source", "mixed_decay",
                 "uniform_lossy", "signed_zero_sources", "pec_intersections"):
        inputs = list(original)
        attributes = dict(fixture["attributes"])
        if name == "anisotropic_source":
            inputs[10] = np.asarray(inputs[10] * np.float32(1.125), dtype=np.float32)
        elif name == "mixed_decay":
            inputs[6] = np.asarray(.99, dtype=np.float32)
        elif name == "uniform_lossy":
            inputs[6:9] = [np.asarray(.97, dtype=np.float32) for _ in range(3)]
        elif name == "signed_zero_sources":
            inputs[9:12] = [np.asarray(v, dtype=np.float32) for v in (0., -0., 0.)]
        elif name == "pec_intersections":
            attributes["boundary_code"] |= 63
        before = [digest_array(x) for x in inputs]
        reference = None
        for kind in ("program", "mstep"):
            session = module.NativeSession(inputs, attributes,
                                           preserve_inputs=True, kind=kind)
            try:
                output = session.run_schedule([3, 4])
                state = logical_native_state(fixture, data, session.download(),
                                              session.clocks(), output["output_bank"])
                if kind == "program":
                    reference = state
                else:
                    leaves = [{"name": field, **compare_arrays(value, reference[field])}
                              for field, value in state.items()]
                    assert len(leaves) == 30 and all(x["passed"] and x["bitwise_exact"] for x in leaves), name
                    rows.append({"case": name, "all30_bitwise": True, "leaves": leaves,
                                 "input_sha256": before, "chunks": [3, 4],
                                 "attributes": attributes,
                                 "reference_kind": "ordinary in-place CUDA program"})
            finally:
                session.close()
        assert before == [digest_array(x) for x in inputs], name
    a.output.write_text(json.dumps({"status": "ok", "module": str(a.module),
                                   "module_sha256": hashlib.sha256(a.module.read_bytes()).hexdigest(),
                                   "fixture": str(a.fixture),
                                   "fixture_sha256": hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
                                   "cases": rows}, indent=2) + "\n")
    print(json.dumps({"status": "ok", "cases": len(rows)}))


if __name__ == "__main__":
    main()
