"""Check one temporal method against ordinary CUDA across source and grid facts.

All cases use live seeded state, arbitrary component/count combinations, both
lossless material formats and the same full-state oracle. Source and grid facts
vary independently; none selects a benchmark-specific numerical implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.cuda_native_const_block_cases import constant_rank2_e_packet
from scripts.cuda_native_probe_cases import point_probe_packet
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of


def main() -> int:
    """Run general source/geometry cases with independent ordinary-step oracles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="")
    args = parser.parse_args()
    module = module_at(args.module)
    assert getattr(module, "unified_temporal", False)
    fixture, data, inputs = load_fixture(args.fixture)
    records = []
    cases = [
        ("no-sources", (0, 0, 0, 0, 0, 0), (1, 1, 1), (29, 32, 31), False, False, ()),
        *[(f"component-{c}", tuple((1 if c % 2 == 0 else 3) if k == c else 0 for k in range(6)),
           (1, 2, 1), (29, 32, 31), bool(c % 2), False, ()) for c in range(6)],
        ("two-component-plane", (0, 0, 2, 0, 2, 0), (2, 3, 1), (29, 32, 31), False, False, ()),
        ("four-component-plane", (0, 6, 6, 0, 6, 6), (2, 3, 1), (29, 32, 31), True, False, ()),
        ("unequal-counts-volume", (1, 2, 3, 4, 5, 7), (2, 3, 4), (29, 32, 31), True, False, ()),
        ("low-clipped-volume", (1, 2, 3, 4, 5, 7), (2, 3, 4), (-1, 15, 31), True, False, ()),
        ("high-clipped-volume", (1, 2, 3, 4, 5, 7), (2, 3, 4), (62, 94, 126), True, False, ()),
        ("y-normal-plane", (2, 4, 0, 3, 0, 5), (3, 1, 2), (29, 32, 31), False, False, ()),
        ("z-normal-plane", (0, 2, 1, 3, 5, 0), (1, 3, 2), (29, 32, 31), True, False, ()),
        ("nonisotropic-H", (1, 2, 3, 4, 5, 7), (2, 3, 4), (29, 32, 31), True, True, ()),
        ("new-rectangle-a", (1, 2, 3, 4, 5, 7), (2, 3, 4), (29, 32, 31), False, False, (57, 73, 105)),
        ("new-rectangle-b", (0, 3, 2, 1, 0, 5), (3, 1, 2), (29, 32, 31), True, False, (73, 65, 97)),
    ]
    if args.case:
        selected = set(args.case.split(","))
        cases = [row for row in cases if row[0] in selected]
        if not cases:
            parser.error("Unknown generality case")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, counts, extent, start, block, nonisotropic, shape in cases:
        values = list(inputs)
        if shape:
            staggering = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
            for c, add in enumerate(staggering):
                target = tuple(n + extra for n, extra in zip(shape, add))
                values[c] = np.resize(inputs[c], target)
                values[74 + c], values[80 + c] = np.zeros(target, np.float32), np.zeros(target, np.float32)
            for family, offset in enumerate((31, 62)):
                for term, axis in enumerate((1, 0, 0, 2, 2, 1)):
                    target = list(values[3 * family + term // 2].shape)
                    target[axis] = 24
                    values[offset + term] = np.resize(inputs[offset + term], target)
            for c in range(3):
                coefficient = np.float32(inputs[37 + c][0])
                values[37 + c] = np.array([coefficient, coefficient * np.float32(1.03125)], np.float32)
                size = values[3 + c].size
                ids = np.zeros((size + 3) // 4 * 4, np.uint8)
                ids[size // 2:size // 2 + 37] = 1
                values[40 + c] = ids.view(np.int32)
        observer = tuple(max(0, min(start[a], min(x.shape[a] for x in values[:6]) - 1)) for a in range(3))
        values, attrs = point_probe_packet(values, fixture["attributes"], observer, 35)
        attrs["coincident_source_group_mask"] = sum(1 << (3 + c) for c, count in enumerate(counts) if count > 1)
        for g in range(9):
            count = counts[g - 3] if g >= 3 else 0
            coeff = np.empty((count, *extent), np.float32)
            wave = np.empty((count, 35), np.float32)
            for term in range(count):
                coeff[term] = np.float32(((-1) ** term) * .17320507764816284 * (term + 1))
                wave[term] = np.float32(.3141592741012573) * ((np.arange(35) + term) % 7 - 3)
            values[86 + 3 * g], values[87 + 3 * g] = coeff, wave
            values[88 + 3 * g] = np.broadcast_to(np.asarray(start, np.int32), (count, 3)).copy()
        if block:
            values = constant_rank2_e_packet(values)
        if nonisotropic:
            values[9] = np.asarray(values[9] * np.float32(1.03125))
        before = [digest_array(x) for x in values]
        expected = ordinary(module, fixture, data, values, attrs, 35)
        if any(counts):
            quiet = list(values)
            for g in range(3, 9):
                quiet[86 + 3 * g] = np.zeros_like(values[86 + 3 * g])
            disabled = ordinary(module, fixture, data, quiet, attrs, 35)
            assert any(not np.array_equal(expected[k], disabled[k]) for k in ("ex", "ey", "ez", "hx", "hy", "hz")), name
        for preserve in (False, True):
            # The established native session contract fixes depth at first
            # preparation. Test both supported depths in independent sessions.
            for depth in (16, 4):
                session = module.NativeSession(values, attrs, preserve_inputs=preserve, kind="mstep", initialize_scratch=False)
                try:
                    for chunks in (([35], [17, 18], [3, 13, 19]) if depth == 16 else ([5, 30],)):
                        session.reset()
                        result = session.run_moving(chunks, depth)
                        parity = exact(state_of(fixture, data, session, result), expected)
                        row = dict(case=name, counts=counts, extent=extent, start=start, block=block, shape=shape,
                            preserve=preserve, depth=depth, chunks=chunks, all30_bitwise=parity["all30_bitwise"],
                            positive_macro=result["moving_macro_count"] > 0, metadata=session.metadata(), parity=parity)
                        records.append(row)
                        args.output.write_text(json.dumps(dict(passed=False, records=records), indent=2) + "\n")
                        assert row["all30_bitwise"] and row["positive_macro"], (name, depth, chunks)
                    if not preserve and depth == 16:
                        session.reset()
                        session.run_moving([17], 16)
                        result = session.run_moving([18], 16)
                        parity = exact(state_of(fixture, data, session, result), expected)
                        assert parity["all30_bitwise"], (name, "cross-call continuation")
                        records.append(dict(case=name, cross_call_continuation=True, all30_bitwise=True, parity=parity))
                finally:
                    session.close()
        assert before == [digest_array(x) for x in values], name
    # Bounds hoisting requires a proved coincident topology, independent of
    # which components/counts happen to be present in the scene.
    rejected = []
    group = next((3 + c for c, count in enumerate(counts) if count > 1), -1)
    for invalid in (("different-starts", "missing-coincident-fact") if group >= 0 else ()):
        changed, changed_attrs = list(values), dict(attrs)
        if invalid == "different-starts":
            changed[88 + 3 * group] = values[88 + 3 * group].copy()
            changed[88 + 3 * group][1, 0] += 1
        else:
            changed_attrs["coincident_source_group_mask"] &= ~(1 << group)
        session = module.NativeSession(changed, changed_attrs, preserve_inputs=False, kind="mstep", initialize_scratch=False)
        try:
            try:
                session.run_moving([16], 16)
            except ValueError as error:
                assert "coincident" in str(error)
                rejected.append(invalid)
            else:
                raise AssertionError(f"Unproved source topology accepted: {invalid}")
        finally:
            session.close()
    args.output.write_text(json.dumps(dict(passed=True, module=str(args.module.resolve()),
        module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), records=records, rejected=rejected,
        source_cases=len(cases), jax_imported="jax" in sys.modules), indent=2) + "\n")
    print(json.dumps(dict(passed=True, source_cases=len(cases), records=len(records))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
