"""Gate cyclic layout against a fresh unrotated ordinary CUDA world oracle.

The oracle executes in a separate process using an explicitly hashed module.
All actual downloaded fields/psi/DFT are inverse-mapped to world storage before
all30 bitwise comparison. This is a correctness gate, not a timing benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_cyclic_fixture import leaf_mapping, restore_world_leaves, rotate_packet
from scripts.cuda_fixture_store import digest_array
from scripts.export_fdtdx_comparison import file_digest
from scripts.prove_cuda_cyclic_axis_equivariance import cyclic
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_native_source_plane import make_packet
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of

CASES = ("normal-plane", "direct-source-DFT", "count3", "nx2", "cropped-ZY",
         "shifted-ZY", "caller-shift-after-true", "caller-equalize-after-false", "nonisotropic-H")


def internal_fixture(world, rotation):
    result = copy.deepcopy(world)
    result["cyclic_layout"] = {"schema": "beamz.cyclic-storage-layout.v1", "xyz_map": rotation.xyz}
    for leaf in result["state_leaves"]:
        leaf["world_shape"] = list(leaf["shape"])
        leaf["world_mapping"] = dict(leaf["mapping"])
        leaf["mapping"] = leaf_mapping(leaf["mapping"], rotation)
        if leaf["mapping"]["kind"] == "field" or (leaf["mapping"]["kind"] == "output" and 6 <= leaf["mapping"]["index"] < 18):
            leaf["shape"] = list(rotation.vector(leaf["shape"]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--oracle-module", type=Path)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=CASES)
    p.add_argument("--reference-only", action="store_true")
    p.add_argument("--steps", type=int, choices=(17, 35), default=35)
    p.add_argument("--ownership", choices=("both", "owned", "preserve"), default="both")
    p.add_argument("--schedule", choices=("both", "whole", "split"), default="both")
    p.add_argument("--repeats", type=int, default=2)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    world, data, inputs = load_fixture(a.fixture)
    if a.reference_only:
        if a.case is None:
            p.error("One case required for world oracle")
        module = module_at(a.module)
        values, attrs = make_packet(inputs, world["attributes"], a.case, a.steps, synthetic=True)
        expected = ordinary(module, world, data, values, attrs, a.steps)
        quiet = list(values)
        for group in (5, 7):
            quiet[86 + 3 * group] = np.zeros_like(values[86 + 3 * group])
        disabled = ordinary(module, world, data, quiet, attrs, a.steps)
        np.savez(a.output, **{"world__" + k: v for k, v in expected.items()},
                 **{"quiet__" + k: v for k, v in disabled.items()})
        return 0
    report = {"schema": "beamz.cyclic-layout-world-validation.v1", "passed": False,
              "module": str(a.module), "module_sha256": file_digest(a.module),
              "oracle_module": str(a.oracle_module or a.module),
              "oracle_module_sha256": file_digest(a.oracle_module or a.module),
              "world_fixture_sha256": file_digest(a.fixture), "worker_sha256": file_digest(__file__), "cases": []}
    try:
        cases = (a.case,) if a.case else CASES
        references = {}
        for case in cases:
            target = a.output.with_name(a.output.stem + "-" + case + ".npz")
            command = [sys.executable, __file__, "--module", str(a.oracle_module or a.module),
                       "--fixture", str(a.fixture), "--output", str(target), "--case", case,
                       "--steps", str(a.steps), "--reference-only"]
            subprocess.run(command, check=True, timeout=120)
            references[case] = target
        module = module_at(a.module)
        rotation = cyclic(2)
        mapped = internal_fixture(world, rotation)
        ownerships = (False, True) if a.ownership == "both" else (a.ownership == "preserve",)
        split = [17, 18] if a.steps == 35 else [16, 1]
        schedules = ([a.steps], split) if a.schedule == "both" else (([a.steps],) if a.schedule == "whole" else (split,))
        for case in cases:
            with np.load(references[case], allow_pickle=False) as ref:
                expected = {k[7:]: np.array(ref[k]) for k in ref.files if k.startswith("world__")}
                quiet = {k[7:]: np.array(ref[k]) for k in ref.files if k.startswith("quiet__")}
            field_live = any(not np.array_equal(expected[k], quiet[k]) for k in ("hx", "hy", "hz", "ex", "ey", "ez"))
            dft_live = any(not np.array_equal(expected[k], quiet[k]) for k in ("dft_vec_re", "dft_vec_im"))
            if not field_live or (case == "direct-source-DFT" and not dft_live):
                raise AssertionError("World oracle must be source-sensitive, with a live direct-DFT case")
            for preserve in ownerships:
                for chunks in schedules:
                    world_values, attrs = make_packet(inputs, world["attributes"], case, a.steps, synthetic=True)
                    values, attrs = rotate_packet(world_values, attrs)
                    session = module.NativeSession(values, attrs, preserve_inputs=preserve)
                    try:
                        if case in ("caller-shift-after-true", "caller-equalize-after-false"):
                            for group in (4, 6):
                                values[88 + 3 * group][:, 1:] += 1 if case == "caller-shift-after-true" else -1
                        before = [digest_array(v) for v in values]
                        for repeat in range(a.repeats):
                            if repeat and not preserve:
                                session.reset()
                            result = session.run_moving(chunks, 16)
                            meta = session.metadata()
                            actual = restore_world_leaves(state_of(mapped, data, session, result), mapped, values[117])
                            parity = exact(actual, expected)
                            want_class = case not in ("count3", "nx2", "cropped-ZY", "shifted-ZY", "caller-equalize-after-false")
                            want_dispatch = want_class and case != "nonisotropic-H"
                            class_ok = bool(meta["moving_source_plane_pair_class"]) == want_class
                            dispatch_ok = bool(meta["moving_source_plane_pair_dispatch_active"]) == want_dispatch
                            row = {"case": case, "preserve": preserve, "chunks": chunks, "repeat": repeat,
                                   "world_all30": parity, "field_source_sensitivity": field_live,
                                   "DFT_source_sensitivity": dft_live, "expected_class": want_class,
                                   "expected_dispatch": want_dispatch, "metadata": meta,
                                   "passed": parity["all30_bitwise"] and class_ok and dispatch_ok}
                            report["cases"].append(row)
                            if not row["passed"]:
                                raise AssertionError(f"Cyclic world parity/admission failure:{case}")
                        if before != [digest_array(v) for v in values]:
                            raise AssertionError("Caller inputs changed")
                    finally:
                        session.close()
        report["passed"] = len(report["cases"]) == len(cases) * len(ownerships) * len(schedules) * a.repeats
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "states": len(report["cases"]), "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
