"""Memory-region scheduling pass for a permanent two-family circular cache.

Imports/publications are moved across arithmetic only after affine-region
hazards are proved absent. Two independent phase clocks can replace a single
clock without adding a third CTA barrier. This is CPU schedule/codegen analysis;
async global-copy admission and CUDA visibility still need hardware validation.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.cuda_space_time_ir import (
        validate_operator,
        validate_stream_signature,
        yee_operator,
    )
except ModuleNotFoundError:
    from cuda_space_time_ir import (
        validate_operator,
        validate_stream_signature,
        yee_operator,
    )


@dataclass(frozen=True)
class Event:
    name: str
    reads: frozenset = frozenset()
    writes: frozenset = frozenset()
    barrier: bool = False
    async_completion: str | None = None


class ScheduleHazard(ValueError):
    pass


def check_schedule(events):
    """Reject conflicting interwarp accesses without an intervening CTA barrier.

    An event's own read/write operation is point-local and ordered internally.
    Separate events can overlap across warps and require disjoint footprints.
    Barriers do NOT complete cp.async: an explicit wait event is required.
    """
    pending, async_groups = [], set()
    checked = 0
    for event in events:
        if event.async_completion:
            if event.async_completion.startswith("issue:"):
                async_groups.add(event.async_completion[6:])
            elif event.async_completion.startswith("wait:"):
                async_groups.discard(event.async_completion[5:])
        if event.barrier:
            if async_groups:
                raise ScheduleHazard(f"CTA barrier does not complete asynchronous imports: {sorted(async_groups)}")
            pending.clear()
            continue
        for old in pending:
            conflict = old.writes.intersection(event.reads | event.writes) | event.writes.intersection(old.reads)
            if conflict:
                raise ScheduleHazard(f"{old.name} may overlap {event.name}: {next(iter(conflict))}")
            checked += 1
        pending.append(event)
    return {"events": len(events), "disjoint_event_pairs": checked,
            "cta_barriers": sum(e.barrier for e in events), "result": "pass"}


def regions(cache=(6, 16, 32), version=1):
    tile = tuple(n - 1 for n in cache)
    low = (-version,) * 3
    previous_low = (-version,) * 3
    current_low = (-version - 1,) * 3

    def points(start, size):
        return set(itertools.product(*(range(a, a + b) for a, b in zip(start, size, strict=True))))

    def slots(values):
        return frozenset(tuple(p % n for p, n in zip(v, cache, strict=True)) for v in values)

    owned = points(low, tile)
    previous = points(previous_low, cache)
    current = points(current_low, cache)
    high = tuple(a + b - 1 for a, b in zip(low, tile, strict=True))
    outgoing = {p for p in owned if any(a == b for a, b in zip(p, high, strict=True))}
    return {"owned": slots(owned), "previous": slots(previous),
            "current": slots(current), "incoming": slots(current - owned),
            "outgoing": slots(outgoing)}


def prove_two_phase_clocks(steps):
    """Visibility of each phase word is supplied by the opposite barrier."""
    visible = {"H": 1, "E": 1}
    pending = {}
    reads = []
    for n in range(1, steps + 1):
        assert visible["H"] == n
        reads.append(("H", n))
        # B1 publishes prior post-B2 E word and retires every H word reader.
        visible.update(pending)
        pending.clear()
        pending["H"] = n + 1
        assert visible["E"] == n
        reads.append(("E", n))
        # B2 publishes H word and retires every E word reader.
        visible.update(pending)
        pending.clear()
        pending["E"] = n + 1
    return {"phase_reads": reads, "H_publication_barrier": "B2",
            "E_publication_barrier": "next B1", "result": "pass"}


def compile_family_schedule(cache=(6, 16, 32), steps=4, *, publication="surface",
                            async_imports=False, omit_wait=False,
                            counter="private", late_h_publication=False,
                            early_e_import=False, operator=None):
    operator = yee_operator() if operator is None else operator
    semantic_proof = validate_operator(operator)
    validate_stream_signature(operator)
    if publication not in ("surface", "owned") or counter not in ("private", "two-phase", "single"):
        raise ValueError("Unsupported schedule/admission")
    events = []

    def memory(family, points):
        # Conservative whole-family slots include thread-private component y.
        return frozenset((family, p) for p in points)

    def emit(name, reads=(), writes=(), **kw):
        events.append(Event(name, frozenset(reads), frozenset(writes), **kw))

    for n in range(1, steps + 1):
        r = regions(cache, n)
        hclock = frozenset({("clock", "H" if counter == "two-phase" else "single")}) if counter != "private" else frozenset()
        eclock = frozenset({("clock", "E" if counter == "two-phase" else "single")}) if counter != "private" else frozenset()
        # Current H imports recycle old-H high slots, disjoint from own H old.
        emit(f"H{n} low import", writes=memory("H", r["incoming"]),
             async_completion=f"issue:H{n}" if async_imports else None)
        if early_e_import:
            emit(f"E{n} EARLY low import", writes=memory("E", r["incoming"]))
        emit(f"H{n} compute", reads=memory("H", r["owned"]) | memory("E", r["previous"]) | hclock,
             writes=memory("H", r["owned"]))
        if async_imports and not omit_wait:
            emit(f"wait H{n} imports", async_completion=f"wait:H{n}")
        emit(f"B1.{n}", barrier=True)
        if counter != "private":
            emit(f"publish next H clock {n+1}", writes=hclock)
        footprint = r["outgoing"] if publication == "surface" else r["owned"]
        if not late_h_publication:
            emit(f"H{n} publication", reads=memory("H", footprint))
        emit(f"E{n} low import", writes=memory("E", r["incoming"]),
             async_completion=f"issue:E{n}" if async_imports else None)
        emit(f"E{n} compute", reads=memory("E", r["owned"]) | memory("H", r["current"]) | eclock,
             writes=memory("E", r["owned"]))
        if async_imports and not omit_wait:
            emit(f"wait E{n} imports", async_completion=f"wait:E{n}")
        emit(f"B2.{n}", barrier=True)
        if counter != "private":
            emit(f"publish next E clock {n+1}", writes=eclock)
        if late_h_publication:
            emit(f"H{n} LATE publication", reads=memory("H", footprint))
        emit(f"E{n} publication", reads=memory("E", footprint))
    proof = check_schedule(events)
    return {"schema": "beamz.phase-family-schedule.v1", "cache_zyx": cache,
            "steps": steps, "counter": counter, "phase_clock_bytes": 8 if counter == "two-phase" else 0 if counter == "private" else 4, "publication": publication,
            "async_imports": async_imports,
            "events": [{"name": e.name, "read_slots": len(e.reads), "write_slots": len(e.writes),
                        "cta_barrier": e.barrier, "async_completion": e.async_completion} for e in events],
            "proof": proof, "affine_legality": semantic_proof,
            "phase_clocks": prove_two_phase_clocks(steps) if counter == "two-phase" else "private loop n",
            "requirements": ["point-local same-family old read only", "opposite-family curls",
                             "permanent distinct H/E shared storage", "one unique writer per physical field slot",
                             "all-lane shuffle inputs loaded before predicates", "async wait before each publishing barrier"]}


def main():
    cases = [compile_family_schedule(counter=c, async_imports=a, publication=p)
             for c in ("private", "two-phase") for a in (False, True) for p in ("surface", "owned")]
    negatives = []
    for name, kwargs in (("single counter races E reader", {"counter": "single"}),
                          ("late H publication races next H import", {"late_h_publication": True}),
                          ("early E import races old E readers", {"early_e_import": True}),
                          ("CTA barrier without async wait", {"async_imports": True, "omit_wait": True})):
        try:
            compile_family_schedule(**kwargs)
        except ScheduleHazard:
            negatives.append(name)
        else:
            raise AssertionError(f"Negative control escaped: {name}")
    try:
        from scripts.prove_cuda_circular_asymmetric import prove as prove_tokens
    except ModuleNotFoundError:
        from prove_cuda_circular_asymmetric import prove as prove_tokens
    tokens = [prove_tokens(steps=n, owner=o, family_reordered=True)
              for n in (2, 8, 16) for o in ((0, 0, 0), (1, 1, 1), (3, 2, 2))]
    tokens.append(prove_tokens(family_reordered=True, publish_all_owned=True))
    print(json.dumps({"scope": __doc__, "cases": cases, "field_token_cases": tokens,
                      "negative_controls_rejected": negatives}, indent=2))


if __name__ == "__main__":
    main()
