#!/usr/bin/env python3
"""Check immutable, phase-local CPML profile facts against ordinary CUDA Program.

These changed-profile packets are correctness witnesses, not benchmark scenes.
Run with the NumPy-only interpreter and the coordinator's exclusive GPU grant.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.cuda_native_probe_cases import branch_witness, point_probe_packet
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of

PROFILE_INDICES = tuple(range(13, 31)) + tuple(range(44, 62))
PAIRS = ((0, 5), (1, 2), (3, 4))


def equal_axis_profiles(inputs):
    """Independent bitwise oracle; phase boundaries and signed zero matter."""
    for base in (13, 44):
        for first, second in PAIRS:
            for kind in range(3):
                a, b = inputs[base + 3 * first + kind], inputs[base + 3 * second + kind]
                if (a.dtype != np.float32 or b.dtype != np.float32
                        or a.shape != b.shape or not np.isfinite(a).all()
                        or not np.isfinite(b).all() or a.tobytes() != b.tobytes()):
                    return False
    return True


def profile_packet(inputs, case):
    changed = list(inputs)
    for index in PROFILE_INDICES:
        changed[index] = np.array(inputs[index], copy=True, order="C")
    if case in ("unequal-paired-H-profile", "immutable-unequal-profile-upload-capture"):
        changed[13 + 3 * 5] *= np.float32(.75)
    elif case == "signed-zero-paired-mismatch":
        changed[13].reshape(-1)[0] = np.float32(0)
        changed[13 + 3 * 5].reshape(-1)[0] = np.float32(-0.)
    elif case == "distinct-H-E-phase-profiles":
        for term in range(6):
            changed[44 + 3 * term] *= np.float32(.875)
            changed[45 + 3 * term] *= np.float32(.9375)
    elif case not in ("equal-axis-positive", "immutable-profile-upload-capture"):
        raise ValueError(f"Unknown profile case: {case}")
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--require-alias", action="store_true")
    a = p.parse_args()
    report = {"schema": "beamz.native-profile-facts.v1", "passed": False,
              "module": str(a.module), "module_sha256": file_digest(a.module),
              "fixture": str(a.fixture), "fixture_sha256": file_digest(a.fixture),
              "worker_sha256": file_digest(__file__), "depth": a.depth,
              "steps": a.steps, "oracle": "fresh ordinary CUDA Program", "cases": []}

    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        if a.steps < a.depth or a.repeats < 1:
            raise ValueError("Require an actual moving macro and >=1 repeat")
        if importlib.util.find_spec("jax") is not None:
            raise RuntimeError("Use the NumPy-only interpreter")
        module = module_at(a.module)
        alias_capability = bool(getattr(module, "moving_runtime_profile_alias", False))
        report["alias_capability"] = alias_capability
        if a.require_alias and not alias_capability:
            raise ValueError("Candidate does not advertise runtime profile aliasing")
        fixture, data, inputs = load_fixture(a.fixture)
        original_hashes = [digest_array(x) for x in inputs]
        point = tuple(min(lookup, min(v.shape[axis] for v in inputs[:6]) - 1)
                      for axis, lookup in enumerate((14, 16, 32)))
        packet, attrs = point_probe_packet(inputs, fixture["attributes"], point, a.steps)
        baseline = ordinary(module, fixture, data, packet, attrs, a.steps)
        cases = ("equal-axis-positive", "unequal-paired-H-profile",
                 "signed-zero-paired-mismatch", "distinct-H-E-phase-profiles",
                 "immutable-profile-upload-capture",
                 "immutable-unequal-profile-upload-capture")
        for name in cases:
            changed = profile_packet(packet, name)
            expected_class = name not in ("unequal-paired-H-profile",
                                          "signed-zero-paired-mismatch",
                                          "immutable-unequal-profile-upload-capture")
            if equal_axis_profiles(changed) != expected_class:
                raise AssertionError(f"Independent profile class mismatch: {name}")
            expected = ordinary(module, fixture, data, changed, attrs, a.steps)
            sensitive = any(not np.array_equal(expected[key], baseline[key])
                            for key in expected)
            if (name in ("unequal-paired-H-profile", "distinct-H-E-phase-profiles")
                    and not sensitive):
                raise AssertionError(f"Changed profile has no physical witness: {name}")
            for preserve in (False, True):
                working = profile_packet(changed, "equal-axis-positive")
                before = [digest_array(x) for x in working]
                session = module.NativeSession(working, attrs, preserve_inputs=preserve)
                try:
                    if name == "immutable-profile-upload-capture":
                        for index in PROFILE_INDICES:
                            working[index][...] *= np.float32(.5)
                        working[13 + 3 * 5].reshape(-1)[0] = np.float32(.125)
                        if equal_axis_profiles(working):
                            raise AssertionError("Caller mutation failed to change class")
                    elif name == "immutable-unequal-profile-upload-capture":
                        for index in PROFILE_INDICES:
                            working[index][...] *= np.float32(.5)
                        for base in (13, 44):
                            for first, second in PAIRS:
                                for kind in range(3):
                                    working[base + 3 * second + kind][...] = working[
                                        base + 3 * first + kind]
                        if not equal_axis_profiles(working):
                            raise AssertionError("Caller mutation failed to equalize class")
                    after_intentional_mutation = [digest_array(x) for x in working]
                    for repeat in range(a.repeats):
                        if repeat and not preserve:
                            session.reset()
                        result = session.run_moving([a.steps], a.depth)
                        metadata = session.metadata()
                        clocks = session.clocks()
                        counters_correct = (
                            clocks["current_step"] == int(working[127]) + a.steps
                            and clocks["elapsed_steps"] == int(working[128]) + a.steps)
                        witness = branch_witness(metadata, result)
                        parity = exact(state_of(fixture, data, session, result), expected)
                        observed_class = metadata.get("moving_equal_axis_profiles")
                        aliased_tiles = int(metadata.get("moving_profile_alias_noio_tiles", 0))
                        class_correct = (not alias_capability or
                                         (observed_class is not None
                                          and bool(observed_class) == expected_class))
                        branch_correct = (not alias_capability or
                                          ((aliased_tiles > 0) if expected_class
                                           else aliased_tiles == 0))
                        host_correct = after_intentional_mutation == [
                            digest_array(x) for x in working]
                        row = {"case": name, "preserve": preserve, "repeat": repeat,
                               "expected_profile_class": expected_class,
                               "observed_profile_class": observed_class,
                               "profile_alias_noio_tiles": aliased_tiles,
                               "profile_class_correct": class_correct,
                               "alias_branch_correct": branch_correct,
                               "source_observation_counters_correct": counters_correct,
                               "physical_sensitivity_to_changed_profiles": sensitive,
                               "branch_witness": witness, "parity": parity,
                               "host_unchanged_after_intentional_mutation": host_correct,
                               "caller_profiles_intentionally_mutated": before !=
                               after_intentional_mutation}
                        row["passed"] = (parity["all30_bitwise"] and host_correct
                                         and witness["positive_noio_branch"]
                                         and class_correct and branch_correct
                                         and counters_correct)
                        report["cases"].append(row)
                        save()
                        if not row["passed"]:
                            raise AssertionError(f"Profile witness failed: {name}")
                finally:
                    session.close()
        report["original_fixture_inputs_unchanged"] = original_hashes == [
            digest_array(x) for x in inputs]
        report["jax_imported"] = "jax" in sys.modules
        report["passed"] = (len(report["cases"]) == len(cases) * 2 * a.repeats
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
