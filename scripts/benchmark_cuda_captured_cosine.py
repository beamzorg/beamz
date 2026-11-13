"""Screen a native candidate on an unchanged captured cosine input packet.

Exports one array at a time through mapped files to bound host memory. Full
notebook runtime and process memory remain separate acceptance measurements.
"""
from pathlib import Path
import argparse
import ctypes
import importlib.util
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cuda_fixture_store import ArrayReader, ArrayWriter, drop_pages, compare_arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--warm-runs", type=int, default=2)
    p.add_argument("--depth", type=int, choices=(4, 16), default=16)
    p.add_argument("--method", choices=("moving", "ordinary"), default="moving")
    p.add_argument("--reference-state", type=Path)
    p.add_argument("--profile-library", type=Path)
    a = p.parse_args()
    if a.warm_runs < 0:
        p.error("Warm sample count must be nonnegative")
    a.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((a.capture / "results.json").read_text())
    reader = ArrayReader(a.capture / "prepared-inputs")
    inputs = [reader[str(i)] for i in range(134)]
    spec = importlib.util.spec_from_file_location("_cuda_native", a.module.resolve())
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    remaining = a.steps; chunks = []
    for n in source["runs"][0]["native"]["chunks"]:
        take = min(n, remaining); chunks.append(take); remaining -= take
        if not remaining: break
    if remaining or a.steps < 1: raise ValueError("Steps outside captured horizon")
    start = time.perf_counter()
    session = module.NativeSession(inputs, source["packet_attributes"], preserve_inputs=False,
                                   kind="mstep" if a.method == "moving" else "program",
                                   initialize_scratch=False)
    upload_s = time.perf_counter() - start
    for value in inputs: drop_pages(value)
    samples = []
    def execute():
        return (session.run_moving(chunks, a.depth) if a.method == "moving"
                else session.run_schedule(chunks))
    try:
        for sample in range(a.warm_runs + 1):
            session.reset(); start = time.perf_counter()
            timing = execute()
            samples.append(dict(wall_s=time.perf_counter() - start, timing=timing))
        profile = None
        if a.profile_library:
            from scripts.profile_cuda_native_phases import summarize
            session.reset()
            library = ctypes.CDLL(str(a.profile_library.resolve()))
            library.beamz_profile_start.argtypes = [ctypes.c_char_p]
            library.beamz_profile_error.restype = ctypes.c_char_p
            trace = a.output / "activity.jsonl"
            if library.beamz_profile_start(str(trace.resolve()).encode()):
                raise RuntimeError(str(library.beamz_profile_error()))
            start = time.perf_counter()
            execute()
            elapsed = time.perf_counter() - start
            if library.beamz_profile_stop():
                raise RuntimeError(str(library.beamz_profile_error()))
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            profile = dict(wall_s=elapsed, activity=summarize(records))
        writer = ArrayWriter(a.output / "state")
        reference = ArrayReader(a.reference_state) if a.reference_state else None
        parity = {}
        names = ["hx", "hy", "hz", "ex", "ey", "ez"] + [f"psi_h.{i}" for i in range(6)] + [f"psi_e.{i}" for i in range(6)] + ["dft_re", "dft_im", "dft_weight"]
        for i, name in enumerate(names):
            index = i if i < 18 else i + 12
            shape = inputs[i] if i < 6 else inputs[31 + i - 6] if i < 12 else inputs[62 + i - 12] if i < 18 else inputs[120 + i - 18]
            temporary = a.output / "leaf.npy"
            value = np.lib.format.open_memmap(temporary, mode="w+", dtype=shape.dtype, shape=shape.shape)
            session.download_into(index, value)
            writer.put(name, value, require_finite=True)
            if reference is not None:
                reference_name = name
                if name.startswith("psi_"):
                    family, term = name.split(".")
                    reference_name = f"cpml_{family}_terms.{term}"
                elif name.startswith("dft_"):
                    reference_name = {"dft_re": "dft_vec_re", "dft_im": "dft_vec_im",
                                      "dft_weight": "dft_weight_sum"}[name]
                if name in reference.metadata:
                    reference_name = name
                expected = reference[reference_name]
                parity[name] = compare_arrays(value, expected)
                drop_pages(expected)
            drop_pages(value); del value; temporary.unlink()
        writer.finish()
        result = dict(module=str(a.module.resolve()), steps=a.steps, depth=a.depth, method=a.method,
            chunks=chunks, upload_s=upload_s, cold=samples[0], warm=samples[1:],
            median_wall_s=float(np.median([x["wall_s"] for x in samples[1:]])) if a.warm_runs else None,
            metadata=session.metadata(), clocks=session.clocks(), parity=parity,
            profile=profile,
            all_bitwise=all(row["bitwise_exact"] for row in parity.values()) if parity else None)
        (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("steps", "median_wall_s", "all_bitwise")}))
    finally:
        session.close()


if __name__ == "__main__": main()
