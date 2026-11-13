"""Read-only ownership and per-location proof for a proposed mixed psi load.

Does not implement the proposed CUDA transform or certify GPU execution.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

CACHE = (8, 16, 32)
OWNED = (7, 15, 31)


def location(slot, origin, n):
    delta = tuple((s-((-n-1) % c)) % c for s, c in zip(slot, CACHE))
    point = tuple(o-n-1+d for o, d in zip(origin, delta))
    return delta, point


def thread_slot(slot):
    z, y, x = slot
    return (z//4)*512+y*32+x, z % 4


def prove(depth, *, omit_first=False, omit_axis=None):
    reads = continued = fresh = 0
    # Origin translation cancels in equality and ownership predicates. Include
    # both signs to exercise the signed physical-coordinate representation.
    for origin in ((0, 0, 0), (21, 45, 93), (-7, -15, -31)):
        for first in range(1, depth+1):
            for slot in itertools.product(*(range(c) for c in CACHE)):
                last = None
                for n in range(first, depth+1):
                    delta, point = location(slot, origin, n)
                    if not all(delta):
                        continue
                    is_fresh = (n == first and not omit_first) or any(
                        d == 1 for a, d in enumerate(delta) if a != omit_axis)
                    reads += 1
                    if is_fresh:
                        fresh += 1
                    else:
                        previous_delta, previous_point = location(slot, origin, n-1)
                        assert n > first, "normal read on first owned update lacks an in-job store"
                        assert all(previous_delta), "normal read follows ghost point, not owned writer"
                        assert all(d == p+1 for d, p in zip(delta, previous_delta))
                        assert point == previous_point, "normal read follows a different physical point"
                        assert last == (n-1, point, thread_slot(slot)), "no same-thread immediately prior store"
                        continued += 1
                    # Canonical cell recurrence executes LoadLocal then
                    # StoreLocal under exactly the same Valid/LocalActive gate.
                    last = (n, point, thread_slot(slot))
    return {"depth": depth, "owned_reads": reads, "fresh_cg_reads": fresh,
            "continued_normal_reads_with_prior_store": continued,
            "all_first_versions": list(range(1, depth+1)), "cache_slots": 4096,
            "origin_translations": 3, "result": "pass"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = [args.source/"src"/n for n in (
        "native_persistent.cuh", "asymmetric_native.cuh", "asymmetric_native_cell_generated.cuh")]
    texts = [path.read_text() for path in paths]
    persistent, environment, cell = texts
    required = {
        "persistent_mapping": (persistent, ["const int first=Fast?1:window[0],prev=first-1;",
            "hz=(-n-1)&(CZ-1)", "zbase+j", "PersistentEnvironment env{p,flags};",
            'static_assert(CZ==8&&CY==16&&CX==32&&T==1024&&LocalZ==4&&YGroups==1,"Paired source mapping only");',
            "const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&(CY-1),zbase=(threadIdx.x/XY)*LocalZ;"]),
        "canonical_address_and_store": (environment, [
            "const int size=p.physical_n[axis]+family;",
            "return coord<12?coord:24+coord-size;",
            "p.phase[slot/6].psi[slot%6].output[PsiOffset(slot,z,y,x)]=value;",
            "return p.phase[slot/6].psi[slot%6].input[PsiOffset(slot,z,y,x)];"]),
    }
    for label, (text, tokens) in required.items():
        for token in tokens:
            assert token in text, (label, token)
    for slot in range(12):
        assert cell.count(f"env.LoadLocal({slot},z,y,x)") == 1
        assert cell.count(f"env.StoreLocal({slot},z,y,x,next_psi);") == 1
        start = cell.index(f"env.LoadLocal({slot},z,y,x)")
        finish = cell.index(f"env.StoreLocal({slot},z,y,x,next_psi);", start)
        assert finish > start
    negative = []
    for name, kwargs in [("omit-first-guard", {"omit_first": True})] + [
        (f"omit-delta-one-axis-{a}", {"omit_axis": a}) for a in range(3)]:
        try:
            prove(4, **kwargs)
        except AssertionError as e:
            negative.append({"mutation": name, "rejected": True, "witness": str(e)})
        else:
            raise AssertionError(f"Unsafe mutation accepted: {name}")
    result = {
        "schema": "beamz.persistent-mixed-psi-independent-review.v1",
        "result": "conditionally-supported-not-implemented-not-GPU-validated",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [prove(d) for d in (4, 16)], "negative_controls": negative,
        "formula": "fresh = n==first || dz==1 || dy==1 || dx==1, only at owned points; continued delta>=2 implies previous delta=delta-1 and identical physical p, cache slot, thread, local j",
        "phase_and_address": "H/E and all six terms have independent allocations; each phase's current recurrence is preceded by that phase's same-thread store on the previous version. Valid and LocalActive depend only on the fixed physical point/term. Canonical packed index is injective over 24 CPML planes with phase-specific N+phase extent; tangent coordinates are unchanged. Input/output must alias the same term, never different terms.",
        "handoff": "Existing physical owner interval proof persistent-psi-handoff-review.json proves no previous/future CTA writes this scalar during the continued interval. A new physical point or first tile version always takes coherent CG, after the existing scheduler acquire and CTA barrier.",
        "memory_model": "Same-thread scalar F32 StoreLocal -> subsequent normal LoadLocal uses the same generic proxy, full overlap, program order. PTX morally-strong and causality/SC-per-location rules prohibit reading a coherence-older write. Cache operators are hints and do not weaken this rule. Fresh CG need not update/invalidate L1: its following same-thread store is the necessary bridge. Neighbor-word stores are disjoint, including when sharing a cache line. No read-only/noncoherent loads or virtual aliases are admitted.",
        "sources": [
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#morally-strong-operations",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#causality",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#sequential-consistency-per-location"],
        "implementation_gates": ["per-point env flags preserve low7 and set bit128 from this point's fresh predicate", "whole-byte flag equality must mask fresh bit", "no __ldg/ld.global.nc for mutable psi", "verify ordinary branch coherent scalar PTX and global store alias identity", "execute stress probe and all30 exact first/replay plus sanitizers before promotion"],
        "gpu_executed": False,
    }
    probe = Path(__file__).with_name("cuda_persistent_mixed_psi_stress.cu")
    result["stress_probe"] = {
        "path": str(probe), "sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        "compiled": False, "executed": False,
        "coverage": "Different-SM acquire/release handoff after L1 primer, fresh CG read, scalar own WB-store to CA-load rounds, concurrent adjacent-word false sharing. Requires positive neighbor-store/different-SM evidence. Acquire may invalidate the primer; does not prove stale-L1 survival.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"passed": True, "output": str(args.output), "continued_checks": sum(c["continued_normal_reads_with_prior_store"] for c in result["cases"]), "negative_controls": len(negative)}))


if __name__ == "__main__":
    main()
