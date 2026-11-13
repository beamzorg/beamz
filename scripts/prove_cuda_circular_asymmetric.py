"""Independent token/lifetime proof for permanent shared x/z and register y.

CacheZYX=6x16x32, owned5x15x31; 512 threads retain Hy/Ey at six fixed Z slots.
No GPU arithmetic, memory-order execution, resource fit, or performance claim.
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


def prove_phase_metadata(steps, *, counter_before_b1=False, omit_e_minus_one=False):
    """Adversarial warp order; only the three named CTA barriers order readers."""
    counter = 1
    warps = 16
    for n in range(1, steps + 1):
        h_reads = []
        for warp in range(warps):
            if counter_before_b1 and warp == warps - 1:
                # A fast warp publishes while the last warp has not read H time.
                counter = n + 1
            h_reads.append(counter)
        assert h_reads == [n] * warps, "Counter write overtakes an H reader before B1"
        # B1 retires every H counter/old-E reader, including the slow warp.
        counter = n + 1
        # B2 publishes counter and imports. No next write until next B1;
        # reaching that B1 requires first crossing this iteration's B3.
        e_reads = [counter - (0 if omit_e_minus_one else 1) for _ in reversed(range(warps))]
        assert e_reads == [n] * warps, "E must read published counter minus one"
        # B3 retires all E readers before any warp enters the next iteration.
    return {"warp_clock_checks": steps * warps * 2,
            "cta_barriers": steps * 3, "counter_write_interval": "B1 to B2",
            "H_clock": "counter", "E_clock": "counter-1"}


def prove(steps=4, owner=(1, 1, 1), *, early_import=False, stale_h=False,  # noqa: C901
          branch_shuffle=False, wrong_z_mask=False, publish_all_owned=False,
          counter_before_b1=False, omit_e_minus_one=False,
          family_reordered=False):
    metadata = prove_phase_metadata(steps, counter_before_b1=counter_before_b1,
                                    omit_e_minus_one=omit_e_minus_one)
    if family_reordered:
        metadata = {"cta_barriers": 2 * steps, "counter_protocol": "separate H/E words; next E word published at next B1"}
    cache, tile, threads = (6, 16, 32), (5, 15, 31), 512
    shape = (19, 37, 71)
    shapes = component_shapes(shape)
    origin = tuple(a * b for a, b in zip(owner, tile, strict=True))
    padded = math.prod(cache)
    shared, registers = {}, {}
    fields = tuple(f + c for f in "HE" for c in "xyz")
    previous_publication = set()
    reads = shuffles = ghost_participants = excluded_sources = 0

    def points(low, extent):
        return itertools.product(*(range(a, a + b) for a, b in zip(low, extent, strict=True)))

    def slot(p):
        z, y, x = ((a - o) % b for a, o, b in zip(p, origin, cache, strict=True))
        return (z * cache[1] + y) * cache[2] + x

    def valid(field, p):
        return all(0 <= a < b for a, b in zip(p, shapes[field], strict=True))

    def token(field, version, p):
        return (field, version, p) if valid(field, p) else ZERO

    def bank(field):
        return registers if field[1] == "y" else shared

    def store(field, version, p):
        bank(field)[field, slot(p)] = token(field, version, p)

    for p in points(tuple(a - 1 for a in origin), cache):
        for field in fields:
            store(field, 0, p)
    for n in range(1, steps + 1):
        low = tuple(a - n for a in origin)
        owned = set(points(low, tile))
        owned_slots = {slot(p) for p in owned}
        incoming = set(points(tuple(a - 1 for a in low), cache)) - owned
        assert owned_slots.isdisjoint(slot(p) for p in incoming)
        # Publication loads are restricted to the previous outgoing high union.
        # Next-H shared writes touch Q(n), disjoint from that union's slots.
        if not family_reordered:
            assert owned_slots.isdisjoint(previous_publication), "Pending publication overlaps next H writes"

        def import_current(n=n, incoming=incoming, family=None):
            for p in incoming:
                producer = tuple((a + n) // b for a, b in zip(p, tile, strict=True))
                if any(valid(f, p) for f in fields):
                    assert all(a <= b for a, b in zip(producer, owner, strict=True))
                    assert sum(producer) < sum(owner)
                for field in fields:
                    if family is None or field.startswith(family):
                        store(field, n - 1 if stale_h and field.startswith("H") else n, p)

        if early_import:
            import_current()
        elif family_reordered:
            import_current(family="H")
        for phase, family in enumerate("HE"):
            if phase and not early_import:
                # B1 retires old-E shared reads and pending prior publication.
                import_current(family="E" if family_reordered else None)
                # In reordered mode B1 follows all H imports+updates; E-low
                # writes are disjoint from E-owned old-E reads, so no B2 is
                # needed before E arithmetic. The region pass checks overlap.
                # In the original mode B2 publishes all current imports.
            opposite = "E" if family == "H" else "H"
            shift = 1 if family == "H" else -1
            shuffled = {}
            for j in range(cache[0]):
                for warp in range(threads // 32):
                    qs = [j * threads + warp * 32 + lane for lane in range(32)]
                    participants = {lane for lane, q in enumerate(qs) if q in owned_slots} if branch_shuffle else set(range(32))
                    assert participants == set(range(32)), "Full-mask shuffle excludes a named lane"
                    shuffles += 1
                    for lane, q in enumerate(qs):
                        ghost_participants += q not in owned_slots
                        source_q = qs[(lane + shift) % 32]
                        shuffled[q] = registers[opposite + "y", source_q]
            for update in trace_yee_step(n - 1):
                if not update.output.field.startswith(family):
                    continue
                for p in owned:
                    q = slot(p)
                    if valid(update.output.field, p):
                        for read in update.reads:
                            source = tuple(a + b for a, b in zip(p, read.offset, strict=True))
                            source_q = slot(source)
                            if read.field[1] != "y":
                                actual = shared[read.field, source_q]
                            elif read.offset[2]:
                                actual = shuffled[q]
                                excluded_sources += source_q not in owned_slots
                            else:
                                assert not read.offset[1], "Canonical Y curls must not consume component y"
                                if wrong_z_mask and read.offset[0]:
                                    j, lane = divmod(q, threads)
                                    source_q = ((j + read.offset[0]) & 5) * threads + lane
                                assert source_q % threads == q % threads
                                actual = registers[read.field, source_q]
                            expected = token(read.field, read.version, source)
                            assert actual == expected, (family, n, p, read, actual, expected)
                            reads += 1
                    store(update.output.field, n, p)
            if not phase and family_reordered:
                # B1 has completed all current H imports and owned updates.
                # H publication can now run concurrently with E-family work.
                for p in points(tuple(a - 1 for a in low), cache):
                    for c in "xyz":
                        assert bank("H" + c)["H" + c, slot(p)] == token("H" + c, n, p)
        # Original B3 or reordered B2 retires E stencils and H publication.
        # E publication may overlap next H, which writes only H and reads E.
        for p in points(tuple(a - 1 for a in low), cache):
            for field in fields:
                assert bank(field)[field, slot(p)] == token(field, n, p)
        high = tuple(a + b - 1 for a, b in zip(low, tile, strict=True))
        outgoing = {p for p in owned if any(a == b for a, b in zip(p, high, strict=True))}
        if publish_all_owned:
            outgoing = owned
        previous_publication = {slot(p) for p in outgoing}
        # Original three-barrier whole-output reads require the final step.
        # Reordered H/E publications retire before their own family is written
        # again, so even whole-owned publication is safe (though wasteful).
    return {"cache_zyx": cache, "owned_zyx": tile, "owner": owner, "steps": steps,
            "threads": threads, "owned_cells": math.prod(tile),
            "state_register_scalars_per_thread": 12, "permanent_shared_field_bytes": 4 * padded * 4,
            "total_shared_bytes": 4 * padded * 4 + 64 + (8 if family_reordered else 4),
            "nominal_ctas_if_registers_at_most64": 2, "nominal_resident_warps": 32,
            "versioned_operand_checks": reads, "warp_shuffle_instructions": shuffles,
            "unowned_lane_participations": ghost_participants,
            "shuffle_reads_from_unowned_slots": excluded_sources,
            "phase_metadata": metadata, "family_reordered": family_reordered, "result": "pass"}


def main():
    cases = [prove(steps=n, owner=o) for n in (2, 8, 16) for o in ((0, 0, 0), (1, 1, 1), (3, 2, 2))]
    negatives = []
    for label, kw in (("import before old E reads retire", {"early_import": True}),
                      ("H import at previous version", {"stale_h": True}),
                      ("divergent full-mask shuffle", {"branch_shuffle": True}),
                      ("mask wrapping nonpower Z6", {"wrong_z_mask": True}),
                      ("all-owned shared publication loads", {"publish_all_owned": True}),
                      ("counter write before B1", {"counter_before_b1": True}),
                      ("E counter read without minus one", {"omit_e_minus_one": True})):
        try:
            prove(**kw)
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError(f"Negative control escaped: {label}")
    print(json.dumps({"scope": __doc__, "cases": cases, "negative_controls_rejected": negatives}, indent=2))


if __name__ == "__main__":
    main()
