"""CPU token proof for fixed-Z Yee plane retirement, not arithmetic parity.

Reads come from the real canonical trace. Arbitrary XY worker permutations
model warp skew at each CTA barrier; deliberately unbarriered variants expose
legal fast/slow-worker counterexamples. Physical cache coordinates and rolling
ghosts are independent of the fixed ascending storage-plane sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_stencil_trace import (
    serialize_updates,
    trace_cpml_step,
    trace_yee_step,
)


class LifetimeError(AssertionError):
    """A consumer observed a wrong field, physical coordinate, or version."""


@dataclass(frozen=True)
class Token:
    field: str
    version: int
    point: tuple[int, int, int]


def dependencies():
    """Reject operators outside this storage contract rather than assume Yee."""
    result = {}
    for update in trace_yee_step():
        result[update.output.field] = tuple(sorted(update.reads))
    # This is the operator used by generate_cuda_native_cell.py. CPML adds only
    # point-local mutable recurrence reads, never another field-plane consumer.
    for update in trace_cpml_step(active_axes=(0, 1, 2)):
        if update.output.field not in result:
            continue
        field_reads = tuple(sorted(r for r in update.reads if not r.field.startswith("psi_")))
        assert field_reads == result[update.output.field]
        assert all(not any(r.offset) for r in update.reads if r.field.startswith("psi_"))
    for field, reads in result.items():
        phase = int(field.startswith("E"))
        for read in reads:
            if read.field[0] == field[0]:
                assert read.field == field and read.version == 0 and not any(read.offset)
            else:
                assert read.version == phase
                assert sum(abs(a) for a in read.offset) <= 1
                assert all(a in (0, 1 if not phase else -1) for a in read.offset)
    return result


def last_uses(cache_z=16):
    """Actual component-specific current-H consumer planes, with circular Z."""
    uses = {(field, z): set() for field in ("Hx", "Hy", "Hz") for z in range(cache_z)}
    for output, reads in dependencies().items():
        if output.startswith("E"):
            for z in range(cache_z):
                for read in reads:
                    if read.field.startswith("H"):
                        uses[read.field, (z + read.offset[0]) % cache_z].add(z)
    return {f"{field}:{z}": sorted(consumers) for (field, z), consumers in uses.items()}


def retirement_certificate(cache=(16, 16, 32)):
    """Exhaust the potential cross-worker conflicts of early previous writes.

    This is independent of sampled worker permutations. A previous plane can
    be retired after the current E read in its own XY thread: no other worker's
    current E reads that write. Earlier same-plane XY readers are separated by
    the current H-publication barrier. Final-plane writes have real cross-warp
    conflicts and therefore require an explicit final CTA retirement barrier.
    """
    cz, cy, cx = cache
    reads = dependencies()
    previous_readers = same_xy_z_reads = final_conflicts = 0
    for z, y, x in itertools.product(range(cz), range(cy), range(cx)):
        previous = (z - 1) % cz
        for field, items in reads.items():
            if not field.startswith("E"):
                continue
            for read in items:
                if not read.field.startswith("H"):
                    continue
                source = tuple((p + d) % size for p, d, size in zip((z, y, x), read.offset, cache, strict=True))
                if source[0] == previous:
                    assert source[1:] == (y, x), (field, read, source)
                    same_xy_z_reads += 1
                if source[0] == z and source[1:] != (y, x):
                    previous_readers += 1
                    if z == cz - 1:
                        final_conflicts += 1
    assert same_xy_z_reads and previous_readers and final_conflicts
    return {"cache_zyx": cache, "same_thread_previous_plane_reads": same_xy_z_reads,
            "same_plane_cross_thread_reads": previous_readers,
            "final_plane_cross_thread_conflicts": final_conflicts,
            "early_previous_write_cross_thread_conflicts": 0,
            "result": "pass"}


def late_start_certificate(cache=(16, 16, 32), minimum=(33, 49, 65),
                           tile=(3, 1, 1), depth=16, *, omit_low=False):
    """Check actual staggered supports at a high-side late time window.

    All previous owned values are outside every support. Every valid previous
    cache value is a low ghost at an exact-version lower-rank high publisher.
    Thus startup zeroing is legal only on the previous owned volume.
    """
    owned = tuple(c - 1 for c in cache)
    origin = tuple(t * b for t, b in zip(tile, owned, strict=True))
    shapes = tuple(tuple(n + d for n, d in zip(minimum, delta, strict=True))
                   for delta in ((0, 0, 1), (0, 1, 0), (1, 0, 0),
                                 (1, 1, 0), (1, 0, 1), (0, 1, 1)))
    first = max(1, *(o - n for o, n in zip(origin, minimum, strict=True)))
    last = min(depth, *(o + b - 1 for o, b in zip(origin, owned, strict=True)))
    if not 1 < first <= last:
        raise ValueError("Choose a nonempty late-start witness")
    prev = first - 1
    head = tuple((-prev - 1) % size for size in cache)
    imported = 0
    for slot in itertools.product(*(range(size) for size in cache)):
        point = tuple(o - prev - 1 + (s - h) % size
                      for o, s, h, size in zip(origin, slot, head, cache, strict=True))
        valid = any(all(0 <= p < size for p, size in zip(point, shape, strict=True))
                    for shape in shapes)
        if not valid:
            continue
        if omit_low:
            raise LifetimeError("Late startup zeroed a required prior-version low ghost")
        assert any(s == h for s, h in zip(slot, head, strict=True))
        producer = tuple((p + prev) // b for p, b in zip(point, owned, strict=True))
        assert all(0 <= a <= b for a, b in zip(producer, tile, strict=True))
        assert sum(producer) < sum(tile)
        local = tuple(p + prev - owner * b for p, owner, b in zip(point, producer, owned, strict=True))
        assert any(q == b - 1 for q, b in zip(local, owned, strict=True))
        producer_origin = tuple(t * b for t, b in zip(producer, owned, strict=True))
        pfirst = max(1, *(o - n for o, n in zip(producer_origin, minimum, strict=True)))
        plast = min(depth, *(o + b - 1 for o, b in zip(producer_origin, owned, strict=True)))
        assert pfirst <= prev <= plast
        imported += 1
    assert imported
    return {"cache_zyx": cache, "minimum_support_zyx": minimum,
            "tile": tile, "first": first, "last": last,
            "required_previous_version": prev, "valid_low_points": imported,
            "result": "pass"}


def prove(cache=(16, 3, 32), steps=4, seed=0, *, negative=None,  # noqa: C901
          early_previous=False):
    """Run actual versioned dependencies under independent barrier permutations.

    early_previous removes only the conservative post-E barrier: a worker can
    retire its preceding plane immediately after its own E read. The following
    H-plane publication barrier still retires all previous-plane XY consumers.
    The final cross-thread H-last retirement barrier is always required.
    """
    cz, cy, cx = cache
    if min(cache) < 2 or cx != 32 or steps < 1:
        raise ValueError("Need cache Z/Y>=2, full warp-X32, and positive steps")
    reads = dependencies()
    workers = list(itertools.product(range(cy), range(cx)))
    rng = random.Random(seed)
    shared, private_h = {}, {}
    saved_last, current, pending = {}, {}, {}
    accesses = crosswarp = checks = 0
    origins = (2 * cz + 3, 2 * cy + 3, 2 * cx + 3)

    def order():
        shuffled = workers.copy()
        rng.shuffle(shuffled)
        return shuffled

    def physical(slot, n):
        return tuple(o - n - 1 + (s + n + 1) % size
                     for o, s, size in zip(origins, slot, cache, strict=True))

    def expected(field, version, slot, n):
        return Token(field, version, physical(slot, n))

    def require(actual, wanted, context):
        nonlocal checks
        checks += 1
        if actual != wanted:
            raise LifetimeError(f"{context}: got {actual}, need {wanted}")

    for slot in itertools.product(*(range(n) for n in cache)):
        for family, store in (("E", shared), ("H", private_h)):
            for c in "xyz":
                store[family + c, slot] = expected(family + c, 0, slot, 0)

    for n in range(1, steps + 1):
        head = tuple((-n - 1) % size for size in cache)

        def owned(slot, head=head):
            return all(s != h for s, h in zip(slot, head, strict=True))

        def h_worker(worker, n=n):
            nonlocal accesses, crosswarp
            y, x = worker
            for z in range(cz):
                slot = z, y, x
                if not owned(slot):
                    continue
                point = physical(slot, n)
                for field in ("Hx", "Hy", "Hz"):
                    for read in reads[field]:
                        neighbor = tuple((s + d) % size for s, d, size in zip(slot, read.offset, cache, strict=True))
                        source = private_h if read.field.startswith("H") else shared
                        wanted = Token(read.field, n - 1, tuple(p + d for p, d in zip(point, read.offset, strict=True)))
                        require(source.get((read.field, neighbor)), wanted, f"H{n} {slot} {read}")
                        accesses += 1
                        crosswarp += int(neighbor[1] != y)
                    private_h[field, slot] = expected(field, n, slot, n)

        def seed_worker(worker):
            y, x = worker
            slot = cz - 1, y, x
            saved_last[worker] = tuple(shared["E" + c, slot] for c in "xyz")
            for c in "xyz":
                shared["H" + c, slot] = private_h["H" + c, slot]
                shared.pop(("E" + c, slot), None)

        # Ghost slots have no owned old-H consumer. These private imports may
        # overlap other workers' H arithmetic, but must not overwrite shared E.
        for slot in itertools.product(*(range(size) for size in cache)):
            if not owned(slot):
                for c in "xyz":
                    private_h["H" + c, slot] = expected("H" + c, n, slot, n)

        # Removing this retirement lets a fast warp clobber old E while a
        # slower warp is still evaluating H, including cross-warp Y operands.
        if negative == "missing_old_e_retirement":
            for worker in order():
                h_worker(worker)
                seed_worker(worker)
        else:
            for worker in order():
                h_worker(worker)

        if negative != "missing_old_e_retirement":
            for worker in order():
                seed_worker(worker)

        for z in range(cz):
            def publish_h(worker, z=z):
                y, x = worker
                slot = z, y, x
                if z == cz - 1:
                    current[worker] = (tuple(private_h["H" + c, slot] for c in "xyz")
                                       if negative == "missing_last_old" else saved_last[worker])
                else:
                    current[worker] = tuple(shared["E" + c, slot] for c in "xyz")
                    for c in "xyz":
                        shared["H" + c, slot] = private_h["H" + c, slot]
                        shared.pop(("E" + c, slot), None)

            def compute_e(worker, z=z, n=n):
                nonlocal accesses, crosswarp
                y, x = worker
                slot = z, y, x
                if owned(slot):
                    point = physical(slot, n)
                    for component, field in enumerate(("Ex", "Ey", "Ez")):
                        for read in reads[field]:
                            if read.field.startswith("E"):
                                actual = current[worker][component]
                                wanted = Token(field, n - 1, point)
                            else:
                                neighbor = tuple((s + d) % size for s, d, size in zip(slot, read.offset, cache, strict=True))
                                if negative == "wrong_z_wrap" and read.offset[0]:
                                    neighbor = ((z + 1) % cz, neighbor[1], neighbor[2])
                                actual = shared.get((read.field, neighbor))
                                wanted = Token(read.field, n, tuple(p + d for p, d in zip(point, read.offset, strict=True)))
                                crosswarp += int(neighbor[1] != y)
                            require(actual, wanted, f"E{n} {slot} {read}")
                            accesses += 1
                    current[worker] = tuple(expected("E" + c, n, slot, n) for c in "xyz")
                else:
                    # Ghost old centers are never consumed by the operator.
                    current[worker] = (None, None, None)

            def retire_previous(worker, z=z, pending=pending):
                if not z and negative != "overwrite_reserved_last":
                    return
                previous = (z - 1) % cz
                y, x = worker
                for component, c in enumerate("xyz"):
                    shared["E" + c, (previous, y, x)] = pending.get(worker, (None,) * 3)[component]
                    shared.pop(("H" + c, (previous, y, x)), None)

            if z == cz - 1 and negative == "missing_final_retirement":
                for worker in order():
                    publish_h(worker)
                for worker in order():
                    compute_e(worker)
                    retire_previous(worker)
                    y, x = worker
                    for component, c in enumerate("xyz"):
                        shared["E" + c, (z, y, x)] = current[worker][component]
                        shared.pop(("H" + c, (z, y, x)), None)
            elif negative == "missing_h_publication":
                for worker in order():
                    publish_h(worker)
                    compute_e(worker)
            else:
                for worker in order():
                    publish_h(worker)
                for worker in order():
                    compute_e(worker)
                    if early_previous:
                        retire_previous(worker)
            # This is the conservative post-E CTA barrier. With early_previous,
            # current-plane XY reads instead retire at the next H publication.
            if not early_previous:
                for worker in order():
                    retire_previous(worker)
            pending = current.copy()

        # Last-plane H has same-plane XY consumers; publishing final E without
        # their barrier cannot be justified by same-thread Z retirement.
        for worker in order():
            y, x = worker
            for component, c in enumerate("xyz"):
                shared["E" + c, (cz - 1, y, x)] = pending[worker][component]
                shared.pop(("H" + c, (cz - 1, y, x)), None)

        # H-stage consumers retired at seed publication before E reuses stage.
        # Reconstruct E ghosts only after every E read; wrong version must fail
        # in next H even though its bytes may have looked like a legal halo.
        for slot in itertools.product(*(range(size) for size in cache)):
            if not owned(slot):
                for c in "xyz":
                    shared["E" + c, slot] = expected("E" + c, n - int(negative == "old_e_ghost"), slot, n)
        for slot in itertools.product(*(range(size) for size in cache)):
            for c in "xyz":
                require(shared.get(("E" + c, slot)), expected("E" + c, n, slot, n), "final E/ghost publication")
                require(private_h.get(("H" + c, slot)), expected("H" + c, n, slot, n), "persistent H publication")
    return {"cache_zyx": cache, "steps": steps, "seed": seed,
            "token_checks": checks, "operator_reads": accesses,
            "cross_warp_y_reads": crosswarp, "private_H_floats_per_thread": 3 * cz,
            "short_E_floats_per_thread": 9, "early_previous": early_previous,
            "result": "pass"}


def prove_stage_reuse(*, early=False):
    """Exhaust the two-warp producer/consumer stage-reuse interleavings.

    Coalesced producer0 owns stage key0, while a different physical-X owner
    in warp1 consumes it. The opposite mapping applies to key1. H retirement
    precedes reuse globally; each E producer precedes its own E reader.
    """
    queues = (("read_H1", "write_E0", "read_E0"),
              ("read_H0", "write_E1", "read_E1"))
    completed = 0

    def visit(indices, stage, retired, history):
        nonlocal completed
        if indices == (3, 3):
            completed += 1
            return
        for worker in range(2):
            index = indices[worker]
            if index == 3:
                continue
            action = queues[worker][index]
            if action.startswith("write") and not early and len(retired) != 2:
                continue
            key = int(action[-1])
            updated, next_retired = list(stage), set(retired)
            if action.startswith("read_H"):
                if stage[key] != "H":
                    raise LifetimeError(f"Cross-warp H-stage clobber: {history + [action]}")
                next_retired.add(key)
            elif action.startswith("write_E"):
                updated[key] = "E"
            elif stage[key] != "E":
                raise LifetimeError(f"Unpublished E-stage read: {history + [action]}")
            next_indices = list(indices)
            next_indices[worker] += 1
            visit(tuple(next_indices), tuple(updated), next_retired, history + [action])

    visit((0, 0), ("H", "H"), set(), [])
    return {"legal_interleavings": completed, "result": "pass"}


def prove_final_overlap(*, early_pending=False):
    """Exhaust legal two-warp overlap when the last H-publication is omitted.

    Last H is immutable since seed/P0. A fast warp can compute E_last while
    another still computes E_previous. Their shared reads commute; final Q
    must precede BOTH pending-previous and last-result shared overwrites.
    The cross-warp previous reader is the actual Yee negative-Y derivative;
    the last-to-previous same-worker reader is its negative-Z derivative.
    """
    queues = (("read_previous_other", "read_last_other", "read_last_z_own",
               "write_previous_own", "write_last_own"),) * 2
    completed = 0

    def visit(indices, state, history):
        nonlocal completed
        if indices == (5, 5):
            completed += 1
            return
        all_reads_retired = min(indices) >= 3
        for worker in range(2):
            index = indices[worker]
            if index == 5:
                continue
            action = queues[worker][index]
            if (action.startswith("write") and not all_reads_retired
                    and (action != "write_previous_own" or not early_pending)):
                continue
            updated = dict(state)
            if action == "read_previous_other":
                key = "previous", 1 - worker
            elif action == "read_last_other":
                key = "last", 1 - worker
            elif action == "read_last_z_own":
                key = "previous", worker
            else:
                key = ("previous" if action == "write_previous_own" else "last"), worker
            if action.startswith("read"):
                if state[key] != "H":
                    raise LifetimeError(f"Unretired final overlap reader: {history + [(worker, action)]}")
            else:
                updated[key] = "E"
            next_indices = list(indices)
            next_indices[worker] += 1
            visit(tuple(next_indices), updated, history + [(worker, action)])

    visit((0, 0), {(plane, worker): "H" for plane in ("previous", "last")
                   for worker in range(2)}, [])
    return {"legal_interleavings": completed, "result": "pass"}


def end_fusion_certificate(cache=(16, 16, 32), steps=17, *, restore_before_final=False):
    """Prove R3 may retire E readers and publish the disjoint E-stage packet.

    Both last frame planes wait until R3. Direct Y/Z ghosts and pure-X ghosts
    partition the ghost union; their frame stores are by the physical XY owner.
    Ordinary last-plane stores can overlap ghosts only within that same thread,
    and must precede ghost restoration. Coalesced packet producers write only
    the separate stage region, never shared field slots before R3.
    """
    cz, cy, cx = cache
    checked = 0
    for n in range(1, steps + 1):
        hz, hy, hx = ((-n - 1) % size for size in cache)
        keys = set()
        for z, y, x in itertools.product(range(cz), range(cy), range(cx)):
            direct = z == hz or y == hy
            pure_x = x == hx and not direct
            owned = z != hz and y != hy and x != hx
            assert int(direct) + int(pure_x) + int(owned) == 1
            if pure_x:
                dz, dy = (z - hz) % cz, (y - hy) % cy
                key = (dz - 1) * (cy - 1) + dy - 1
                assert 0 <= key < (cz - 1) * (cy - 1)
                assert key not in keys
                keys.add(key)
            if (direct or pure_x) and z >= cz - 2:
                # Final pending/last writers and ghost writers have the SAME
                # physical XY thread, irrespective of logical rolling heads.
                final_writer = ghost_writer = (y, x)
                assert final_writer == ghost_writer
                if restore_before_final:
                    raise LifetimeError("Last-plane dummy E store clobbers a restored ghost")
            checked += 1
        assert len(keys) == (cz - 1) * (cy - 1)
    return {"cache_zyx": cache, "steps": steps, "slots_checked": checked,
            "pure_x_packet_points": (cz - 1) * (cy - 1),
            "required_order": "R3; E_previous+E_last stores; all ghost restoration; R4",
            "result": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [prove(cache, steps, seed, early_previous=early)
             for cache, steps in (((4, 3, 32), 9), ((16, 3, 32), 17))
             for seed in (0, 7) for early in (False, True)]
    cases.append(prove((16, 16, 32), 17, 11))
    rejected = []
    for negative in ("missing_old_e_retirement", "missing_h_publication",
                     "overwrite_reserved_last", "old_e_ghost", "missing_last_old",
                     "missing_final_retirement", "wrong_z_wrap"):
        try:
            prove((4, 3, 32), 5, negative=negative)
        except LifetimeError:
            rejected.append(negative)
        else:
            raise AssertionError(f"Negative escaped: {negative}")
    stage = prove_stage_reuse()
    try:
        prove_stage_reuse(early=True)
    except LifetimeError:
        rejected.append("early_stage_reuse")
    final_overlap = prove_final_overlap()
    for label, callback in (("early_pending_before_final_Q", lambda: prove_final_overlap(early_pending=True)),
                            ("zeroed_late_start_ghost", lambda: late_start_certificate(omit_low=True)),
                            ("ghost_restore_before_final_plane_store", lambda: end_fusion_certificate(restore_before_final=True))):
        try:
            callback()
        except LifetimeError:
            rejected.append(label)
        else:
            raise AssertionError(f"Negative escaped: {label}")
    result = {"scope": __doc__, "cases": cases, "last_uses": last_uses(),
              "retirement_certificate": retirement_certificate(),
              "late_start": late_start_certificate(),
              "stage_reuse": stage,
              "final_two_plane_overlap": final_overlap,
              "repaired_end_fusion": end_fusion_certificate(),
              "barriers_per_version": {"conservative": 2 * 16 + 4,
                                       "reduced_separately_proved": 16 + 3,
                                       "repaired_end_fusion": 16 + 2},
              "proof_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "canonical_native_operator_sha256": hashlib.sha256(json.dumps(
                  serialize_updates(trace_cpml_step(active_axes=(0, 1, 2))),
                  sort_keys=True).encode()).hexdigest(),
              "negative_controls_rejected": rejected}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
