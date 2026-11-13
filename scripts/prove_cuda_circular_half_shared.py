"""Register-owned H/E plus one-family shared circular-cache token proof.

Every fixed cache slot has one thread/register owner. Opposite neighbors use
shared memory; own values (and optional centers) use only same-thread registers.
This checks state/phase/slot ownership, not CUDA execution or compiler resources.
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


def prove(cache_shape=(4, 4, 4), threads=4, steps=4, *, local_centers=False,
          overwrite_E_early=False, overwrite_H_early=False, stale_import_H=False,
          register_neighbors=False, local_z=False, wrong_z_direction=False):
    tile = tuple(p - 1 for p in cache_shape)
    shape = tuple(3 * p + 1 for p in cache_shape)
    shapes = component_shapes(shape)
    origin = tile
    padded = math.prod(cache_shape)
    assert padded % threads == 0
    slots_per_thread = padded // threads
    fields = tuple(f + c for f in "HE" for c in "xyz")
    registers = {}
    shared = {}
    shared_reads = register_reads = 0

    def points(low, sizes):
        return itertools.product(*(range(a, a + b) for a, b in zip(low, sizes, strict=True)))

    def slot(p):
        z, y, x = (a % b for a, b in zip(p, cache_shape, strict=True))
        return (z * cache_shape[1] + y) * cache_shape[2] + x

    def valid(field, p):
        return all(0 <= a < b for a, b in zip(p, shapes[field], strict=True))

    def publish(family):
        for c in "xyz":
            for q in range(padded):
                shared[c, q] = registers[family + c, q]

    initial_low = tuple(a - 1 for a in origin)
    for p in points(initial_low, cache_shape):
        for field in fields:
            registers[field, slot(p)] = field, 0, p
    publish("E")
    for n in range(1, steps + 1):
        low = tuple(a - n for a in origin)
        owned = set(points(low, tile))
        cache_low = tuple(a - 1 for a in low)
        incoming = set(points(cache_low, cache_shape)) - owned
        assert {slot(p) for p in owned}.isdisjoint({slot(p) for p in incoming})
        if overwrite_E_early:
            publish("H")  # A possible bad ordering without barrier1.
        equations = trace_yee_step(n - 1)
        for phase, family in enumerate("HE"):
            if phase:
                # Imports modify ONLY each slot owner's private registers.
                # Old E owned values and all E shared reads are unaffected.
                for p in incoming:
                    q = slot(p)
                    thread, j = q % threads, q // threads
                    assert thread + j * threads == q and j < slots_per_thread
                    for field in fields:
                        clock = n - 1 if stale_import_H and field.startswith("H") else n
                        registers[field, q] = field, clock, p
                # Barrier1 retires Eshared reads; H is now complete in regs.
                publish("H")
                # Barrier2 makes current H shared visible before any E read.
                if overwrite_H_early:
                    publish("E")  # Bad ordering without barrier3.
            for update in equations:
                if not update.output.field.startswith(family):
                    continue
                for p in owned:
                    q = slot(p)
                    owner_thread = q % threads
                    if valid(update.output.field, p):
                        for read in update.reads:
                            source = tuple(a + b for a, b in zip(p, read.offset, strict=True))
                            if not valid(read.field, source):
                                continue
                            is_z = read.offset[0] != 0 and read.offset[1:] == (0, 0)
                            actual_source = source
                            if local_z and is_z and wrong_z_direction:
                                actual_source = (p[0] - read.offset[0], p[1], p[2])
                            source_slot = slot(actual_source)
                            own_family = read.field.startswith(family)
                            use_register = own_family or (local_centers and not any(read.offset)) or (local_z and is_z) or register_neighbors
                            if use_register:
                                assert source_slot % threads == owner_thread, "Cross-thread register read"
                                actual = registers[read.field, source_slot]
                                register_reads += 1
                            else:
                                actual = shared[read.field[1], source_slot]
                                shared_reads += 1
                            assert actual == (read.field, read.version, source), (n, phase, actual, read, source)
                    registers[update.output.field, q] = update.output.field, n, p
        # Barrier3 retires Hshared reads. All E registers can replace shared.
        publish("E")
        # Barrier4 completes publication before the next H phase.
        for p in points(cache_low, cache_shape):
            for field in fields:
                assert registers[field, slot(p)] == (field, n, p)
        # All outgoing/final reads stay in the SAME register-owner thread.
        for p in owned:
            q = slot(p)
            assert (q % threads) + (q // threads) * threads == q
    return {"cache_zyx": cache_shape, "owned_zyx": tile, "threads": threads,
            "steps": steps, "slots_per_thread": slots_per_thread,
            "state_scalars_per_thread": 6 * slots_per_thread,
            "one_family_shared_bytes": 3 * padded * 4,
            "local_centers": local_centers, "register_reads": register_reads,
            "local_z": local_z,
            "shared_reads": shared_reads, "result": "pass"}


def main():
    modes = ((False, False), (True, False), (True, True))
    cases = [prove(steps=n, local_centers=c, local_z=z) for n in (2, 4, 8) for c, z in modes]
    cases += [prove((16, 16, 16), 256, 2, local_centers=c, local_z=z) for c, z in modes]
    cases += [prove((16, 16, 32), 512, 2, local_centers=c, local_z=z) for c, z in modes]
    negatives = []
    for label, kw in (("overwrite Eshared before H retires", {"overwrite_E_early": True}),
                      ("overwrite Hshared before E retires", {"overwrite_H_early": True}),
                      ("stale current-H import", {"stale_import_H": True}),
                      ("cross-thread register neighbor", {"register_neighbors": True}),
                      ("wrong register-Z direction", {"local_z": True, "wrong_z_direction": True})):
        try:
            prove(**kw)
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError(f"Negative control escaped: {label}")
    print(json.dumps({"scope": __doc__, "cases": cases,
                      "negative_controls_rejected": negatives}, indent=2))


if __name__ == "__main__":
    main()
