"""Independent support-aware proof for later FULL-step nearest3 readiness.

The initial clipped-first wait remains ALL7. Only later n>first waits reduce
to immediate negative-axis faces. Completion/WAR remains all7 consumers.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

B = (7, 15, 31)
MASKS = list(itertools.product((0, 1), repeat=3))[1:]
FACES = [d for d in MASKS if sum(d) == 1]


def geometry(extent, depth):
    def window(q):
        origin = tuple(x*b for x, b in zip(q, B))
        return max(1, *(o-n for o, n in zip(origin, extent))), min(depth, *(o+b-1 for o, b in zip(origin, B)))

    def parents(q, nearest):
        return [tuple(x-d for x, d in zip(q, delta)) for delta in (FACES if nearest else MASKS)
                if all(x >= d for x, d in zip(q, delta))]

    @lru_cache(None)
    def includes(q, requested, target, version):
        first, last = window(q)
        if first > last:
            return False  # Empty completion carries no imaginary physical work.
        t = min(last, max(first, requested))
        if q == target:
            return first <= version <= t
        if any(x < y for x, y in zip(q, target)):
            return False
        for step in range(first, t+1):
            for p in parents(q, step > first):
                if includes(p, step, target, version):
                    return True
        return False

    def start_covers(q, n, target, version):
        first, _ = window(q)
        return any(includes(p, n, target, version) for p in parents(q, n > first))

    # All inclusive0..N coordinates intentionally overapproximate the union of
    # six canonical staggered supports, whose real sizes are N or N+1. Passing
    # this larger cube cannot accidentally assume every max index is N-1.
    mutable = prior_completion_routes = late_start = early_completion_faces = inclusive_N = 0
    for point in itertools.product(*(range(n+1) for n in extent)):
        for n in range(2, depth+1):
            q = tuple((p+n)//b for p, b in zip(point, B))
            first, last = window(q)
            assert first <= n <= last
            if n <= first:
                continue
            target = tuple((p+n-1)//b for p, b in zip(point, B))
            mask = tuple(x-y for x, y in zip(q, target))
            if sum(mask) < 2:
                continue
            pf, pl = window(target)
            assert pf <= n-1 <= pl
            assert first <= n-1 <= last
            # The new owned point was a supported ghost of THIS tile in its
            # immediately previous active iteration. That iteration already
            # acquired the old owner's full publication, including local state.
            assert start_covers(q, n-1, target, n-1)
            for axis, bit in enumerate(mask):
                if bit:assert point[axis] == q[axis]*B[axis]-(n-1)-1
            assert start_covers(q, n, target, n-1), (extent, depth, point, n, q, target)
            mutable += 1
            changed = [a for a, bit in enumerate(mask) if bit]
            if all(point[a] == 0 for a in changed):
                assert first <= n-1
                prior_completion_routes += 1
            if any(point[a] == 0 for a in changed):
                early_completion_faces += 1
                for axis in changed:
                    if point[axis] == 0:
                        face = list(q);face[axis] -= 1
                        lo, hi = window(tuple(face))
                        assert lo <= n-1 == hi
            if any(p == size for p, size in zip(point, extent)):
                inclusive_N += 1
            if first > 1:
                late_start += 1
    counts = tuple((n+depth+b-1)//b+1 for n, b in zip(extent, B))
    ghosts = inactive_omitted = 0
    for q in itertools.product(*(range(c) for c in counts)):
        first, last = window(q)
        for n in range(max(first+1, 2), last+1):
            origin = tuple(x*b for x, b in zip(q, B))
            for mask in MASKS:
                if sum(mask) < 2 or any(x < bit for x, bit in zip(q, mask)):
                    continue
                supported = all(0 <= o-n-1 <= size if bit else max(0, o-n) <= min(size, o+b-1-n)
                                for bit, o, size, b in zip(mask, origin, extent, B))
                if not supported:
                    inactive_omitted += 1
                    continue
                target = tuple(x-bit for x, bit in zip(q, mask))
                assert start_covers(q, n, target, n), (extent, depth, q, n, target)
                # Every intermediary along ANY subset path is active at n.
                axes = [a for a, bit in enumerate(mask) if bit]
                for subset in itertools.product((0, 1), repeat=len(axes)):
                    intermediary = list(q)
                    for axis, bit in zip(axes, subset):intermediary[axis] -= bit
                    lo, hi = window(tuple(intermediary))
                    assert lo <= n <= hi
                ghosts += 1
    return {"extent_N_zyx": extent, "depth": depth, "inclusive_physical_points": (extent[0]+1)*(extent[1]+1)*(extent[2]+1),
            "mutable_diagonal_n_minus1_handoffs": mutable,
            "all_changed_axes_p0_early_completion_routes": prior_completion_routes,
            "handoffs_with_changed_face_completed_at_n_minus1": early_completion_faces,
            "handoffs_touching_inclusive_N_coordinate": inclusive_N,
            "late_first_mutable_handoffs": late_start,
            "supported_diagonal_current_ghost_requests": ghosts,
            "unsupported_diagonal_current_requests_masked_out": inactive_omitted,
            "memoized_cumulative_history_queries": includes.cache_info().currsize, "result": "pass"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    new_path, old_path = a.source/"src/native_persistent.cuh", a.parent/"src/native_persistent.cuh"
    new, old = new_path.read_text(), old_path.read_text()
    assert hashlib.sha256(old.encode()).hexdigest() == "f76604eede410c12d37ae0a5e0759be72f3c2c05cae6910cb9716922483bc742"
    # Exact names are checked after the implementing agent publishes source;
    # no silently weakened fallback is accepted by this independent review.
    assert new.count("WaitUpstream<false>(") == 3 and new.count("WaitUpstream<true>(step+1)") == 1
    inverse = new.replace("// TRACE_NEAREST3_LATER_ONLY: fullstep publication; startup always ALL7.\ntemplate<bool NearestOnly=false>\n", "")
    inverse = inverse.replace("for(int neighbor=1;neighbor<(NearestOnly?4:8);++neighbor){const int bits=NearestOnly?(1<<(neighbor-1)):neighbor;const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;", "for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;")
    inverse = inverse.replace("WaitUpstream<true>(step+1);", "WaitUpstream(step+1);").replace("WaitUpstream<false>(", "WaitUpstream(")
    assert inverse == old, "Full consumer inverse differs beyond admitted waits"
    for name in ("native_extension.cc", "native_persistent.h"):
        assert (a.source/"src"/name).read_bytes() == (a.parent/"src"/name).read_bytes()
    def section(text, start, end):return text[text.index(start):text.index(end)]
    for start, end in (("__device__ __forceinline__ bool WaitCompleted", "__device__ __forceinline__ void Claim"),
                       ("__device__ __forceinline__ void Claim", "__device__ __forceinline__ uint32_t WaitProducer")):
        assert section(new, start, end) == section(old, start, end)
    cases = [geometry(extent, depth) for extent, depth in (((25, 31, 47), 4), ((25, 31, 47), 16), ((27, 35, 63), 16))]
    def startup_window(q):
        origin = tuple(x*b for x, b in zip(q, B))
        return max(1, *(o-n for o, n in zip(origin, (25, 31, 47)))), min(16, *(o+b-1 for o, b in zip(origin, B)))
    assert startup_window((2, 1, 2)) == (15, 16)
    assert startup_window((1, 1, 2)) == (15, 13)
    assert startup_window((2, 0, 2)) == (15, 14)
    assert startup_window((1, 0, 2)) == (15, 13)
    result = {"schema": "beamz.persistent-later-nearest3-independent-review.v1",
        "result": "support-aware-fullstep-later-only-transitive-readiness-proved-under-original-ownership-contract",
        "sha256": {str(q): hashlib.sha256(q.read_bytes()).hexdigest() for q in (old_path, new_path, Path(__file__))},
        "cases": cases,
        "entire_consumer_inverse_byte_exact": True,
        "host_and_ABI_byte_identical": True,
        "startup_nearest_negative": {"N_zyx": [25, 31, 47], "depth": 16,
            "current_q": [2, 1, 2], "current_window": [15, 16],
            "Z_face_q_and_empty_window": [[1, 1, 2], [15, 13]],
            "Y_face_q_and_empty_window": [[2, 0, 2], [15, 14]],
            "diagonal_q_and_empty_window": [[1, 0, 2], [15, 13]],
            "rejected": True,
            "reason": "Empty Z/Y faces can claim and complete without acquiring diagonal D's tag. X-face ancestors have X<=1, while D.X=2. Thus startup-nearest need not establish the unconditional all7 base tags, even though D has no supported current payload. Retaining all7 startup is necessary for this implementation contract."},
        "current_field_proof": "Needed diagonal current ghost on changed axis a has p_a=o_a-n-1>=0, hence n<=o_a-1. Current active n and monotone origins imply every reduced-subset intermediary first<=n and last>=n. Each nearest parent must therefore execute stepn and acquire its own ancestors at n before releasing current ready. Composing those releases covers each valid diagonal trail load. Unsupported negative coordinates are masked before loading.",
        "mutable_state_proof": "Fresh physical psi/observer handoff needs diagonal writer D at n-1. Because reduced n>first, this same current tile already executed n-1. Its new-owned physical point p at n was a valid low ghost at n-1, whose supported field packet required D's full publication n-1. The already-proved current-field transitive chain therefore acquired D's psi/observer write in the previous local iteration. Prior CTA/RE and program order carry that visibility into current local-state load; contiguous owner intervals prevent D rewriting p afterward. This argument does not require an intermediary to execute current n. Corroboration: a changed face with p_a=0 completes at n-1 and exports prior-state history; a positive changed coordinate stays active at n. Never infer fictitious n work from promoted ReadyDepth.",
        "startup": "All7 full-step first wait remains exactly all7, including late first>1. It acquires tags before bases validation and previous-version packet imports. Nearest3 is only in PublishAndWait's later n+1 path. This proof does not admit startup reduction or phase-ready2n/H-ready semantics.",
        "scope_cumulativity": "Ancestor GPU release/acquire -> intermediary leader program order/CTA join -> intermediary GPU release/acquire exports ancestor generic payload causality, even when intermediary does not itself read every ancestor word. Existing producer/consumer cumulative-join PTX proof applies. Batched publication and acquired floors may strengthen the observed version; promotion still represents actual last active history, never fictitious future work.",
        "lifetime": "Claim and WaitCompleted must remain byte-identical, waiting old holder plus ALL7 direct consumers. The current tile still directly reads diagonal trail storage; nearest3 readiness does not retire or remove those consumers. Captured all7 startup tags remain stable through current completion. Counter resets and owner-tag epochs unchanged.",
        "canonical_support": "Oracle overapproximates actual canonical six Yee shapes with inclusive0..N cube; explicit N+1 endpoints included. H/E/term addresses remain separate and canonical packed psi mapping is unchanged.",
        "limits": "CPU support/history proof, not numerical simulation or hardware timing. Requires actual source inverse, admission tests, GPU all30 exact/replay, root-owned compiler/resource and sanitizer screens. Does not certify phase-ready or startup nearest3.",
        "gpu_executed": False}
    a.output.parent.mkdir(parents=True, exist_ok=True);a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"passed": True, "output": str(a.output), "mutable_handoffs": sum(c["mutable_diagonal_n_minus1_handoffs"] for c in cases), "current_ghosts": sum(c["supported_diagonal_current_ghost_requests"] for c in cases)}))


if __name__ == "__main__":main()
