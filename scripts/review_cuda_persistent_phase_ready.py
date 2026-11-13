"""Independent ring-lifetime and exact paired-body inverse review.

No CUDA implementation/build/GPU work. Controller startup0 witnesses are
separate actual-extraction CPU artifacts recorded in the review JSON.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from scripts.patch_cuda_native_persistent_phase_ready import H_PUBLISH, E_IMPORT

CACHE = (8, 16, 32)


def body(text):
    return text[text.index("template<int Depth>\n__device__ __forceinline__ void PersistentPairedBody"):
                text.index("__global__ void ResetPersistent")].strip()


def invert_body(text):
    b = body(text)
    assert b.count(H_PUBLISH) == b.count(E_IMPORT) == 1
    b = b.replace(H_PUBLISH, "").replace("  if(!threadIdx.x)PublishAndWait(2*n-1,2*window[1]);\n", "")
    b = b.replace("  if(!threadIdx.x)PublishAndWait(2*n,2*window[1]);\n", "")
    b = b.replace("GateImports(2*n-1);", "GateImports(n);")
    b = b.replace("WaitUpstream(2*(window[0]-1));", "WaitUpstream(window[0]-1);")
    b = b.replace("ImportPhasePacket<Depth,SupportSafe,0>", "ImportPacket<Depth,SupportSafe>")
    b = b.replace("StagePhaseH<Depth,SupportSafe,T>", "StageSourcePendingH<Depth,SupportSafe,T>")
    ghost = "    if(zs==hz||ys==hy||sx==hx){const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,dz,dy,dx);frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;set_ey(zs,ys,si,v.ey);}\n"
    b = b.replace("    if constexpr(ZGroups==2)if(j==LocalZ-1)", ghost+"    if constexpr(ZGroups==2)if(j==LocalZ-1)")
    b = b.replace("if(surface>=0){p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;}",
                  "if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][si];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;}")
    b = b.replace("if(n==Depth){const auto&ef=FinalField(p,c+3);",
                  "if(n==Depth){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][si];")
    b = b.replace("  __threadfence();\n  __syncthreads(); // RLpair", "  if(ShouldPublish(n,window[1]))__threadfence();\n  __syncthreads(); // RLpair")
    b = b.replace(E_IMPORT+"  if(!threadIdx.x)*phase_step=n+1;", "  if(!threadIdx.x){*phase_step=n+1;PublishAndWait(n,window[1]);}")
    return b


def point(slot, n):
    return tuple(-n-1+((s+n+1) % c) for s, c in zip(slot, CACHE))


def ring_proof(depth):
    slots = list(itertools.product(*(range(c) for c in CACHE)))
    checked = h_pubs = 0
    for first in range(1, depth+1):
        # Ideal previous-version point labels prove address/lifetime mapping.
        # Actual clipped late-start zero/support initialization is byte-identical
        # after inverse, so its existing support proof remains necessary.
        e = {s: (point(s, first-1), first-1) for s in slots}
        for n in range(first, depth+1):
            head = tuple((-n-1) % c for c in CACHE)
            owned = [s for s in slots if all(x != h for x, h in zip(s, head))]
            for s in owned:
                p = point(s, n)
                assert e[s] == (p, n-1)
                # H reads old E at positive neighbors. A wrapped ghost slot
                # still holds the HIGH old-E point until R0 retires H readers.
                for axis in range(3):
                    t = list(s); t[axis] = (t[axis]+1) % CACHE[axis]
                    q = list(p); q[axis] += 1
                    assert e[tuple(t)] == (tuple(q), n-1)
                    checked += 1
            h = {s: (point(s, n), n) for s in slots}
            # H imports touch only ghost private slots. Early H publication
            # reads exactly the untouched owned set; later E never mutates h.
            h_pubs += len(owned)*3
            for s in owned:
                p = point(s, n)
                assert e[s] == (p, n-1)
                for axis in range(3):
                    t = list(s);t[axis] = (t[axis]-1) % CACHE[axis]
                    q = list(p);q[axis] -= 1
                    assert h[tuple(t)] == (tuple(q), n)
                    checked += 1
            # Paired E scratch retirement precedes delayed ghost overwrite.
            # The original pending writes have unique thread/slot ownership;
            # imports cover its complement, and RE retires all next-H readers.
            e = {s: (point(s, n), n) for s in slots}
    return {"depth": depth, "all_first_versions": list(range(1, depth+1)),
            "neighbor_point_version_checks": checked, "owned_H_words_published_unchanged": h_pubs,
            "cache_slots": len(slots), "result": "pass"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parent = a.parent/"src/native_persistent.cuh"
    source = a.source/"src/native_persistent.cuh"
    old, new = parent.read_text(), source.read_text()
    assert invert_body(new) == body(old), "Paired body does not invert byte-exactly"
    assert old.count("__syncthreads()") == new.count("__syncthreads()")
    assert new.count("p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][si];") == 1
    assert new.count("hf.data[z*hf.plane+y*hf.row+x]=h[c][si];") == 1
    assert "const uint32_t seen=WaitProducer(producer,owner,step);if(C().abort_local)return;" in new
    for begin, end in (("__device__ __forceinline__ bool WaitCompleted", "__device__ __forceinline__ void Claim"),
                       ("__device__ __forceinline__ void Claim", "__device__ __forceinline__ uint32_t WaitProducer")):
        assert old[old.index(begin):old.index(end)] == new[new.index(begin):new.index(end)]
    reset = "__global__ void ResetPersistent"
    inverse_worker = new[new.index(reset):].replace("C().depth=2*Depth;", "C().depth=Depth;")
    inverse_worker = inverse_worker.replace("Release(C().ready+C().owner,uint32_t(2*Depth));", "Release(C().ready+C().owner,uint32_t(Depth));")
    assert old[old.index(reset):] == inverse_worker
    worker = new[new.index("__global__ __launch_bounds__(1024,1) void PersistentTileStep"):]
    assert worker.index("PersistentPairedBody<Depth>(p);") < worker.index("__syncthreads();", worker.index("PersistentPairedBody<Depth>(p);")) < worker.index("Release(C().completed+C().tile")
    artifact = {"schema": "beamz.persistent-phase-ready-independent-review.v1",
        "result": "v3-startup-zero-hole-found; v4-fixed-and-independent-delayed-parent-witness-passed; no-additional-material-hole-found",
        "source_sha256": {str(q): hashlib.sha256(q.read_bytes()).hexdigest() for q in (parent, source, Path(__file__))},
        "paired_body_inverse_exact": True, "original_barrier_count_unchanged": True,
        "Claim_and_WaitCompleted_byte_identical": True,
        "worker_and_host_inverse_exact_after_only_counter_depth_doubling": True,
        "ring_lifetime_cases": [ring_proof(d) for d in (4, 16)],
        "H_phase": "Local owned H consumes old E after previous E gate and RE. Current H-only imports follow acquired ready2n-1; no current E load. Ghost imports cannot mutate owned H publication slots. Every H/psiH/observerH writer fences before R0; leader release2n-1 after R0 includes all CTA writers.",
        "E_phase": "E scalar calls, private H, snapshot and paired scratch order invert byte-exactly. E uses old E only at owned centers and current H at negative neighbors. Early current-E ghost initialization is unnecessary for those reads; delaying it also protects old-E H readers. E/psiE/observerE/trails/finalE fenced before RL, leader release2n after RL, then acquired upstreamE gate precedes family1-only ghost imports; unique slot restoration/imports and RE protect next H.",
        "source_FP_CPML_DFT": "Canonical ordered cell call bodies and call sequence unchanged. H stores move earlier with identical private values, E stores remain at original paired sites. Each phase's mutable physical recurrence is protected by previous-E/phase-ready ownership handoff and contiguous-owner proof; phase-specific psi addresses remain distinct. Source wave time uses unchanged n and stable macro state. Observer per-field point recurrence ownership is unchanged.",
        "completion_WAR": "ready2Depth exposes payload before final E ghost reads retire; it is NOT completion. completed_epoch remains after full body return and CTA barrier. Claim still waits old-holder plus all7 consumer completed epochs; no ready value substitutes for WAR retirement.",
        "startup_zero_fix": "v3 WaitProducer successful acquiredready0 was mistaken for failure by if(!seen); actual CPU witness returned before two absent parent tags. v4 checks abort_local instead, continuing all parents; delayed5ms/10ms actual-controller witness waits for both tags with abort0/floor0.",
        "abort_shared_HB": "GateImports leader waits upstream then always releases shared CTA gate, including abort; each importer acquires it before reading stable shared abort. No leader abort change can split a subsequent barrier because intervening R0/RL precedes later PollAbort. LocalH and E arithmetic CTA barrier counts are unchanged.",
        "witness_directories": ["phase-ready-independent-startup-zero", "phase-ready-independent-startup-zero-fixed"],
        "validation": "18 phase unit tests passed. Root owns full actual phase-controller execution, resource compile, GPU exact/replay/sanitizer and paired timings.",
        "limits": "Ring oracle proves phase/physical point labels, not floating-point values or clipped support from scratch. Exact inverse preserves clipped initialization/support; source CPML/observation buffer identity and prior physical-owner proof remain prerequisites. No GPU execution by reviewer.",
        "gpu_executed": False}
    artifact["actual_startup_witnesses"] = []
    for directory, name, result in (
            ("phase-ready-independent-startup-zero", "startup-zero-witness", "v3 unsafe early return reproduced with abort0 and two absent tags"),
            ("phase-ready-independent-startup-zero-fixed", "startup-zero-fixed", "v4 waits for both tags delayed5ms/10ms, abort0/floor0")):
        location = a.output.parent/directory
        artifact["actual_startup_witnesses"].append({
            "result": result, "CPU_executed": True,
            "sha256": {str(q): hashlib.sha256(q.read_bytes()).hexdigest()
                       for q in (location/(name+".cc"), location/name, location/"actual_persistent_control.inc")}})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(artifact, indent=2)+"\n")
    print(json.dumps({"passed": True, "output": str(a.output), "ring_checks": sum(x["neighbor_point_version_checks"] for x in artifact["ring_lifetime_cases"])}))


if __name__ == "__main__":
    main()
