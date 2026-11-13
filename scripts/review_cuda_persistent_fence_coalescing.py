"""Independent cumulative-HB proof for coalescing GPU publication fences.

This models documented synchronization relations, not GPU timing/hardware.
Removing a fence alone is not the proposed transform: the writer join and
GPU-scope release/acquire plus consumer join remain mandatory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path


def histories(order, *, warp_barrier=True, chain=True, producer_cta=False,
              ready_release=True, consumer_join=True):
    all_writers = set(range(1024))
    if producer_cta:
        leader = all_writers
    else:
        previous = set()
        for warp in order:
            local = set(range(warp*32, warp*32+32)) if warp_barrier else {warp*32}
            previous = previous | local if chain else local
        leader = previous
    if not ready_release:
        leader = set()
    # Thread0 GPU acquire imports leader's cumulative history. A subsequent
    # CTA/shared release-acquire gate carries it to each matching payload reader.
    readers = [leader if consumer_join or i == 0 else set() for i in range(1024)]
    return sum(i in readers[i] for i in range(1024))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    orders = [list(range(32)), list(reversed(range(32)))]
    rng = random.Random(826513)
    for _ in range(128):
        order = list(range(32));rng.shuffle(order);orders.append(order)
    for order in orders:
        assert histories(order) == 1024
    assert histories(orders[0], producer_cta=True) == 1024
    negatives = []
    for name, kwargs, expected in (
            ("omit-full-warp-memory-barrier", {"warp_barrier": False}, 32),
            ("relax-shared-RMW-breaks-cumulative-chain", {"chain": False}, 32),
            ("omit-GPU-release-pattern", {"ready_release": False}, 0),
            ("omit-consumer-join", {"consumer_join": False}, 1)):
        result = histories(orders[0], **kwargs)
        assert result == expected and result < 1024
        negatives.append({"mutation": name, "ordered_matching_writer_reader_pairs": result,
                          "rejected": True, "interpretation": "Missing HB proof; not a claim that every hardware execution fails."})
    here = Path(__file__).resolve().parent
    files = (Path(__file__), here/"cuda_persistent_fence_coalescing_stress.cu",
             here/"patch_cuda_native_persistent_h_publication_join.py")
    result = {
        "schema": "beamz.persistent-publication-fence-coalescing-independent-review.v1",
        "result": "sound-under-listed-synchronization-contract; no-GPU-execution-by-reviewer",
        "sha256": {str(q): hashlib.sha256(q.read_bytes()).hexdigest() for q in files},
        "primary_NVIDIA_sources": {
            "release_acquire_patterns_and_explicit_foreign_thread_cumulativity_footnote": "https://docs.nvidia.com/cuda/parallel-thread-execution/#release-and-acquire-patterns",
            "bar_sync_synchronizes_with_matching_bar_sync": "https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-synchronization",
            "base_causality_transitive_cumulative": "https://docs.nvidia.com/cuda/parallel-thread-execution/#causality-order",
            "payload_read_cannot_observe_coherence_older_write": "https://docs.nvidia.com/cuda/parallel-thread-execution/#causality",
            "full_mask_warp_barrier_memory_ordering": "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-warp-sync",
            "CUTLASS_independent_implementation": "https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/barrier.h"},
        "formal_CTA_chain": "For EVERY writer i: generic scalar global W_i ->po matching bar.sync_i ->sw bar.sync_leader ->po st.release.gpu.global Ready ->sw observing ld.acquire.gpu.global Ready in consumer leader ->po st.release.cta.shared Gate ->sw ld.acquire.cta.shared Gate in reader i ->po generic scalar global R_i. Program-order and synchronizes edges compose in base causality. W_i/R_i use identical address and generic proxy, so the chain preserves causality; R_i cannot observe a write older than W_i. No per-writer fence is needed.",
        "formal_warp_RMW_chain": "Every writer W_i -> full-member __syncwarp memory ordering -> that warp leader atom.acq_rel.cta.shared.add counter. All32 same-word naturally aligned RMWs serialize; each acquire side observes the preceding RMW release side. Inductively the final RMW carries all32 warp histories (1024 writers). Thread0's ld.acquire.cta.shared counter==32 observes that final release, then its GPU Ready release/acquire and consumer gate export/import the complete history. No per-writer fence is needed.",
        "scope_and_proxy": "Each warp join covers exactly its participating writer/leader pair; shared RMW/acquire scope CTA covers every warp leader and thread0; GPU Ready scope covers producer/consumer leaders; consumer CTA gate covers its readers. There is no requirement that every synchronization operation have GPU scope. Payload endpoints use the same generic address/proxy. Texture/readonly loads, virtual aliases and asynchronous proxies are excluded.",
        "CUTLASS_interpretation": "GenericBarrier::arrive_inc calls Sync::sync(), then only thread0 invokes red_release. red_release uses a leader GPU acq_rel fence followed by relaxed reduction and explicitly accounts for other threads' weak writes before the block barrier. Its leader fence is a release-pattern component because red.relaxed alone has no release; BeamZ already uses a standalone st.release.gpu.global release pattern. This corroborates cumulative joining without per-writer fences; CUTLASS itself is not the formal proof.",
        "assumptions": [
            "Every participating writer's ordinary global field/psi/trail/observer store is before its matched producer join; no writer omitted by divergent predicates",
            "Full CTA barrier remains before publication OR full-mask warp barrier and all32 acq_rel shared RMW arrivals remain before publication",
            "Counter reset is exclusive after all prior arrivals/readers retire and before the existing RE; no mixed-generation counter32",
            "GPU Ready release/acquire remain strong, aligned and same epoch/tag; relaxed speculative polling alone never admits reads",
            "Consumer leader distributes acquired history through a real CTA barrier or shared release/acquire gate before payload reads",
            "Existing full completion/WAR retention prevents later overwrites until all payload readers retire",
            "No async copy, readonly/noncoherent payload load, virtual alias or same-word unrelated writer"],
        "HB_algebra_checks": {"warp_arrival_orders": len(orders), "ordered_pairs_per_order": 1024,
                              "total_positive_matching_pairs": len(orders)*1024+1024,
                              "negative_controls": negatives},
        "old_oracle_correction": "An omitted-writer-fence negative that assumes individual GPU fences are mandatory does not establish unsoundness of the cumulative-join alternative. Preserve negatives for missing joins/releases/reader retirement, and admit documented cumulative HB paths.",
        "stress_probe": {"source": str(here/"cuda_persistent_fence_coalescing_stress.cu"),
            "compiled": False, "gpu_executed": False,
            "modes": {"0": "fence-free CTA join", "1": "fence-free warp/RMW join", "2": "CTA plus leader-only fence reference", "3": "all-writer fence reference"},
            "coverage": "All1024 writers/readers, distinct-SM check, two-slot epoch reuse, adjacent-word false sharing, acquired tag-like epoch and reader acknowledgments, join reset, fail-closed watchdog. Positive checks validate actual code, not a proof of necessity or worst-case scheduling."},
        "not_changed": "No backend implementation edits, CUDA builds or GPU jobs by reviewer. Root owns resource/PTX/SASS inspection, real stress, all30 exact/replay/sanitizer and paired timing gates."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"passed": True, "output": str(a.output), "positive_pairs": result["HB_algebra_checks"]["total_positive_matching_pairs"], "negative_controls": len(negatives)}))


if __name__ == "__main__":
    main()
