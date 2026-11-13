"""Token and full-mask participation proof for the paper-shaped warp-X cache.

No CUDA execution, arithmetic parity, compiler register count, or speed claim.
All X shuffle operands are snapshotted for every lane before owned/valid tests.
Centers/Z stay in owner registers; only Y reads use a two-component shared frame.
"""

from __future__ import annotations

import itertools
import json
import math

try:
    from scripts.cuda_stencil_trace import trace_yee_step
    from scripts.prove_cuda_systolic_schedule import component_shapes
except ModuleNotFoundError:
    from cuda_stencil_trace import trace_yee_step
    from prove_cuda_systolic_schedule import component_shapes


ZERO = ("zero",)


def prove(cache=(16, 16, 32), threads=512, steps=4, owner=(1, 1, 1),
          *, branch_shuffle=False, bad_wrap=False, stale_e_shuffle=False,
          stale_h_shuffle=False, omit_shared_component=False):
    assert cache[2] == 32 and math.prod(cache[1:]) == threads
    padded = math.prod(cache)
    assert padded % threads == 0
    slots = padded // threads
    tile = tuple(p - 1 for p in cache)
    shape = (35, 37, 71)
    shapes = component_shapes(shape)
    origin = tuple(a * b for a, b in zip(owner, tile, strict=True))
    registers = {}
    shared = {}
    shared_reads = shuffle_reads = register_reads = participation_checks = 0
    excluded_source_reads = invalid_participants = 0

    def points(low, extent):
        return itertools.product(*(range(a, a + b) for a, b in zip(low, extent, strict=True)))

    def slot(p):
        z, y, x = (a % b for a, b in zip(p, cache, strict=True))
        return (z * cache[1] + y) * cache[2] + x

    def valid(field, p):
        return all(0 <= a < b for a, b in zip(p, shapes[field], strict=True))

    def token(field, version, p):
        return (field, version, p) if valid(field, p) else ZERO

    def publish(family):
        for c in (0, 2):
            if omit_shared_component and c == 2:
                continue
            for q in range(padded):
                shared[c, q] = registers[family + "xyz"[c], q]

    fields = tuple(f + c for f in "HE" for c in "xyz")
    for p in points(tuple(a - 1 for a in origin), cache):
        for field in fields:
            registers[field, slot(p)] = token(field, 0, p)
    publish("E")
    for n in range(1, steps + 1):
        low = tuple(a - n for a in origin)
        owned = set(points(low, tile))
        p_by_slot = {slot(p): p for p in owned}
        cache_low = tuple(a - 1 for a in low)
        incoming = set(points(cache_low, cache)) - owned
        assert {slot(p) for p in incoming}.isdisjoint(p_by_slot)
        for phase, family in enumerate("HE"):
            if phase:
                # Full H update loop has retired old E register reads first.
                for p in incoming:
                    for field in fields:
                        registers[field, slot(p)] = token(field, n, p)
                # Barrier1 retires all old E shared reads before H replaces it.
                publish("H")
                # Barrier2 makes current H visible before E reads/shuffles.
            opposite = "E" if family == "H" else "H"
            shift = 1 if family == "H" else -1
            shuffled = {}
            for j in range(slots):
                for warp in range(threads // 32):
                    qs = [j * threads + warp * 32 + lane for lane in range(32)]
                    participants = set(range(32))
                    if branch_shuffle:
                        # Deliberately places full-mask shuffles inside owned.
                        participants = {lane for lane, q in enumerate(qs) if q in p_by_slot}
                    assert participants == set(range(32)), "Full-mask shuffle excludes a named lane"
                    participation_checks += 1
                    for lane, q in enumerate(qs):
                        invalid_participants += q not in p_by_slot or not any(valid(opposite + c, p_by_slot[q]) for c in "xyz")
                        src_lane = (lane + (-shift if bad_wrap else shift)) % 32
                        source_q = qs[src_lane]
                        # Every lane executes the same two shuffle instructions.
                        for c in (1, 2):
                            field = opposite + "xyz"[c]
                            actual = registers[field, source_q]
                            stale = (phase == 0 and stale_e_shuffle) or (phase == 1 and stale_h_shuffle)
                            if stale and actual != ZERO:
                                actual = (actual[0], actual[1] - 1, actual[2])
                            shuffled[field, q] = actual
            equations = trace_yee_step(n - 1)
            for update in equations:
                if not update.output.field.startswith(family):
                    continue
                for p in owned:
                    q = slot(p)
                    if valid(update.output.field, p):
                        for read in update.reads:
                            source = tuple(a + b for a, b in zip(p, read.offset, strict=True))
                            source_q = slot(source)
                            if read.field.startswith(family) or not any(read.offset) or read.offset[0]:
                                assert source_q % threads == q % threads, "Register operand changes thread owner"
                                actual = registers[read.field, source_q]
                                register_reads += 1
                            elif read.offset[2]:
                                assert read.field[1] in "yz"
                                actual = shuffled[read.field, q]
                                shuffle_reads += 1
                                excluded_source_reads += source_q not in p_by_slot
                            else:
                                assert read.offset[1] and read.field[1] in "xz"
                                key = "xyz".index(read.field[1]), source_q
                                assert key in shared, "Missing required Y-shared component"
                                actual = shared[key]
                                shared_reads += 1
                            expected = token(read.field, read.version, source)
                            assert actual == expected, (family, n, p, read, actual, expected)
                    registers[update.output.field, q] = token(update.output.field, n, p)
        # Barrier3 retires current-H shared reads before all E registers replace it.
        publish("E")
        # Barrier4 completes E staging before the next H phase.
        for p in points(cache_low, cache):
            for field in fields:
                assert registers[field, slot(p)] == token(field, n, p)
    return {"cache_zyx": cache, "owned_zyx": tile, "threads": threads,
            "steps": steps, "owner": owner, "state_register_scalars_per_thread": 6 * slots,
            "shared_frame_bytes": 2 * padded * 4, "shared_base_table_bytes": 64,
            "phase_metadata_bytes": 4, "total_shared_bytes": 2 * padded * 4 + 68,
            "shared_scalar_writes_per_step": 4 * padded,
            "shuffle_scalar_transfers_per_step": 4 * padded,
            "warp_shuffle_instructions_per_step": 4 * padded // 32,
            "shared_read_scalars": shared_reads, "shuffle_read_scalars": shuffle_reads,
            "register_read_scalars": register_reads, "full_warp_participation_checks": participation_checks,
            "ghost_or_unowned_lane_participations": invalid_participants,
            "shuffle_reads_from_unowned_slots": excluded_source_reads,
            "source_level_shared_loads_per_interior_update": 4,
            "source_level_shared_writes_per_cache_slot_step": 4,
            "result": "pass"}


def main():
    cases = [prove(steps=n, owner=o) for n in (2, 4, 8) for o in ((0, 0, 0), (1, 1, 1), (2, 2, 2))]
    negatives = []
    for label, kw in (("shuffle inside owned branch", {"branch_shuffle": True}),
                      ("wrong warp wrap direction", {"bad_wrap": True}),
                      ("stale H-stage old-E shuffle", {"stale_e_shuffle": True}),
                      ("stale E-stage current-H shuffle", {"stale_h_shuffle": True}),
                      ("missing tangential Y component", {"omit_shared_component": True})):
        try:
            prove(steps=2, **kw)
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError(f"Negative control escaped: {label}")
    print(json.dumps({"scope": __doc__, "cases": cases, "negative_controls_rejected": negatives}, indent=2))


if __name__ == "__main__":
    main()
