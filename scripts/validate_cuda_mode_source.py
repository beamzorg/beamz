"""Live complete-state CUDA oracle checks for mode-source specialization.

Includes negative dispatch controls, changed planes, CPML, lossless block
materials, odd tails, continuation and resets. Use the NumPy-only interpreter.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_native_const_block_cases import constant_rank2_e_packet
from scripts.cuda_native_probe_cases import point_probe_packet
from scripts.cuda_fixture_store import digest_array
from scripts.validate_cuda_native_moving import load_fixture, exact
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of


def mode_packet(inputs, attrs, point, steps, *, terms=6, width=1, rank2=True):
    changed, attrs = point_probe_packet(inputs, attrs, point, steps)
    active = (4, 5, 7, 8)
    attrs["coincident_source_group_mask"] = sum(1 << g for g in active)
    for group in range(9):
        offset = 86 + 3 * group
        count = terms if group in active else 0
        coeff = np.empty((count, 1, 2, width), np.float32)
        wave = np.empty((count, steps), np.float32)
        for term in range(count):
            coeff[term] = np.float32(((-1) ** term) * (term + 1) / 32)
            wave[term] = np.float32(.125) * ((np.arange(steps) + term) % 7 - 3)
        changed[offset], changed[offset + 1] = coeff, wave
        changed[offset + 2] = np.broadcast_to(np.asarray(point, np.int32), (count, 3)).copy()
    if rank2:
        changed = constant_rank2_e_packet(changed)
    return changed, attrs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", help="Select one live case for focused sanitizer runs")
    a = p.parse_args()
    module = module_at(a.module)
    assert (getattr(module, "moving_mode_source54_six", False)
            or getattr(module, "moving_unified_sources", False))
    fixture, data, inputs = load_fixture(a.fixture)
    records = []
    cases = [
        ("block-interior", (29, 32, 31), {}),
        ("block-low-cpml", (1, 15, 31), {}),
        ("block-high-cpml", (62, 94, 126), {}),
        ("byte-interior", (29, 32, 31), dict(rank2=False)),
        ("five-terms-fallback", (29, 32, 31), dict(terms=5)),
        ("wide-plane-fallback", (29, 32, 31), dict(width=2)),
        ("anisotropic-H-fallback", (29, 32, 31), {}),
    ]
    if a.case:
        cases = [row for row in cases if row[0] == a.case]
        if not cases:
            p.error("Unknown mode-source case")
    for name, point, options in cases:
        values, attrs = mode_packet(inputs, fixture["attributes"], point, 35, **options)
        if name == "anisotropic-H-fallback":
            values[9] = np.asarray(values[9] * np.float32(1.03125))
        hashes = [digest_array(x) for x in values]
        expected = ordinary(module, fixture, data, values, attrs, 35)
        quiet = list(values)
        for g in (4, 5, 7, 8): quiet[86 + 3 * g] = np.zeros_like(values[86 + 3 * g])
        disabled = ordinary(module, fixture, data, quiet, attrs, 35)
        assert any(not np.array_equal(expected[k], disabled[k]) for k in ("ex", "ey", "ez", "hx", "hy", "hz"))
        assert any(not np.array_equal(expected[k], disabled[k]) for k in expected if "dft" in k)
        session = module.NativeSession(values, attrs, preserve_inputs=False, kind="mstep", initialize_scratch=False)
        try:
            for chunks in ([35], [16, 19], [17, 18]):
                session.reset()
                result = session.run_moving(chunks, 16)
                parity = exact(state_of(fixture, data, session, result), expected)
                row = dict(case=name, chunks=chunks, all30_bitwise=parity["all30_bitwise"],
                    positive_macro=result["moving_macro_count"] > 0, metadata=session.metadata(), parity=parity)
                records.append(row)
                a.output.write_text(json.dumps(dict(passed=False, records=records), indent=2) + "\n")
                assert row["all30_bitwise"] and row["positive_macro"], (name, chunks)
        finally: session.close()
        assert hashes == [digest_array(x) for x in values]
    # Certificate input bytes must be the private upload snapshot, even when
    # the caller corrupts its coefficient pool and headers before first use.
    values, attrs = mode_packet(inputs, fixture["attributes"], (29, 32, 31), 35)
    expected = ordinary(module, fixture, data, values, attrs, 35)
    session = module.NativeSession(values, attrs, preserve_inputs=False, kind="mstep", initialize_scratch=False)
    try:
        for i in range(37, 40): values[i].fill(np.nan)
        for i in range(40, 43): values[i].fill(np.iinfo(np.int32).max)
        result = session.run_moving([17, 18], 16)
        parity = exact(state_of(fixture, data, session, result), expected)
        assert parity["all30_bitwise"]
        records.append(dict(case="private-block-capture", all30_bitwise=True, parity=parity))
    finally: session.close()
    a.output.write_text(json.dumps(dict(passed=True, records=records, jax_imported="jax" in sys.modules), indent=2) + "\n")
    print(json.dumps(dict(passed=True, cases=len(records))))


if __name__ == "__main__": main()
