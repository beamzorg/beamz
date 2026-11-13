#!/usr/bin/env python3
"""Exercise certified observer-only dispatch using a separate nearby source.

Packets are changed correctness witnesses, not benchmark scenes. Every packet
receives a fresh ordinary CUDA Program oracle in the NumPy-only environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.cuda_native_block_cases import rank2_e_packet
from scripts.cuda_native_probe_cases import point_probe_packet
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_plane_retirement import module_at, moving_repeats, ordinary


def separated_probe_packet(inputs, attributes, source, observer, steps):
    changed, attrs = point_probe_packet(inputs, attributes, source, steps)
    observed, _ = point_probe_packet(inputs, attributes, observer, steps)
    changed[113], changed[114] = observed[113], observed[114]
    return changed, attrs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--case-index", type=int)
    p.add_argument("--ownership", choices=("both", "owned", "preserve"), default="both")
    p.add_argument("--source-zyx", type=int, nargs=3, default=(32, 32, 14))
    a = p.parse_args()
    report = {"schema": "beamz.native-observer-dispatch.v1", "passed": False,
              "module": str(a.module), "module_sha256": file_digest(a.module),
              "fixture": str(a.fixture), "fixture_sha256": file_digest(a.fixture),
              "worker_sha256": file_digest(__file__), "steps": a.steps,
              "depth": a.depth, "oracle": "fresh ordinary CUDA Program", "cases": []}

    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        if a.steps < a.depth or a.repeats < 1:
            raise ValueError("Require an actual macro and >=1 repeat")
        module = module_at(a.module)
        if not (getattr(module, "moving_observer_uniform_1024", False)
                and getattr(module, "moving_compact_observer_dispatch", False)):
            raise ValueError("Candidate lacks certified compact observer dispatch")
        fixture, data, inputs = load_fixture(a.fixture)
        original_hashes = [digest_array(x) for x in inputs]
        # At depth16, owner X31 has trajectory minimum15, excluding source14.
        # Thus observer15 (and16) reaches policy2, rather than coincident IO4.
        source = tuple(a.source_zyx)
        near = (source[0], source[1], source[2] + 1)
        farther = (source[0], source[1], source[2] + 2)
        cases = (("observer15-positive", near),
                 ("observer16-changed-same-shape", farther),
                 ("rank2-fallback", near),
                 ("nonisotropic-H-fallback", near))
        if a.case_index is not None:
            cases = (cases[a.case_index],)
        ownerships = ((False, True) if a.ownership == "both"
                      else (a.ownership == "preserve",))
        for name, observer in cases:
            changed, attrs = separated_probe_packet(
                inputs, fixture["attributes"], source, observer, a.steps)
            unmodified = ordinary(module, fixture, data, changed, attrs, a.steps)
            expected_active = not name.endswith("fallback")
            if name == "rank2-fallback":
                changed = rank2_e_packet(changed)
                # Newer candidates certify exact block values; older modules
                # deliberately retain the original rank-two fallback gate.
                expected_active = bool(getattr(module, "moving_block_uniform_e", False))
            elif name == "nonisotropic-H-fallback":
                changed[9] = np.array(changed[9], copy=True)
                changed[9][...] *= np.float32(1.03125)
            before = [digest_array(x) for x in changed]
            expected = ordinary(module, fixture, data, changed, attrs, a.steps)
            if name == "rank2-fallback" and not exact(expected, unmodified)["all30_bitwise"]:
                raise AssertionError("Rank2 transform changed physical coefficients")
            quiet = list(changed)
            for group in (5, 7):
                quiet[86 + 3 * group] = np.zeros_like(changed[86 + 3 * group])
            disabled = ordinary(module, fixture, data, quiet, attrs, a.steps)
            field_live = any(not np.array_equal(expected[k], disabled[k])
                             for k in ("ex", "ey", "ez", "hx", "hy", "hz"))
            dft_live = any(not np.array_equal(expected[k], disabled[k])
                           for k in expected if "dft" in k)
            if not (field_live and dft_live):
                raise AssertionError("Separate source/observer witness is not live")
            for preserve in ownerships:
                rows = moving_repeats(module, fixture, data, changed, attrs,
                                      expected, a, preserve)
                for row in rows:
                    count = int(row["metadata"].get("moving_uniform_observer_tiles", 0))
                    positive = row["branch_witness"]["positive_macro"]
                    branch_correct = positive and ((count > 0) if expected_active else count == 0)
                    row.update(case=name, source_zyx=source, observer_zyx=observer,
                               certified_observer_tiles=count,
                               expected_observer_dispatch=expected_active,
                               observer_dispatch_correct=branch_correct,
                               source_disabled_field_sensitivity=field_live,
                               source_disabled_DFT_sensitivity=dft_live)
                    row["passed"] = (row["parity"]["all30_bitwise"] and branch_correct
                                     and row["source_observation_counters_correct"])
                    report["cases"].append(row)
                    save()
                    if not row["passed"]:
                        raise AssertionError(f"Observer dispatch failed: {name}")
            if before != [digest_array(x) for x in changed]:
                raise AssertionError("Native solve modified caller packet")
        report["original_fixture_inputs_unchanged"] = original_hashes == [
            digest_array(x) for x in inputs]
        report["jax_imported"] = "jax" in sys.modules
        report["positive_macro"] = all(x["branch_witness"]["positive_macro"]
                                       for x in report["cases"])
        report["positive_observer_branch"] = any(
            x["expected_observer_dispatch"] and x["certified_observer_tiles"] > 0
            and x["branch_witness"]["positive_macro"] for x in report["cases"])
        report["passed"] = (len(report["cases"]) == len(cases) * len(ownerships) * a.repeats
                            and report["original_fixture_inputs_unchanged"]
                            and all(x["passed"] for x in report["cases"]))
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    save()
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
