"""Source and endpoint-storage proofs for the perturbing CTA diagnostic."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_cuda_cta_intervals import ring_dependencies, tile_geometry
from scripts.cuda_persistent_wavefront_dag import build, rank_slots
from scripts.patch_cuda_cta_timing_diagnostic import PARENT, TIMERS, sha, transformed


def braced(text, begin):
    brace = text.index("{", begin)
    level, end = 1, brace + 1
    while level:
        level += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[brace:end]


def iteration(text, function):
    begin = text.index("void " + function + "(")
    body = braced(text, begin)
    return braced(body, body.index("for(int n="))


def protocol(counts=(4, 5, 3), macros=3, *, negative=None):
    rng = random.Random(8293)
    coordinates, ranks = tile_geometry(counts)
    count = ranks.size
    starts, ends, smids = np.zeros(count, np.int64), np.zeros(count, np.int64), np.full(count, -1, np.int32)
    timer, writes, barrier_cases, empty = 0, 0, 0, 0
    for macro in range(macros):
        start_writes, end_writes, sm_writes = np.zeros(count, int), np.zeros(count, int), np.zeros(count, int)
        for rank in range(int(ranks.max()) + 1):
            # Simulate policy/compact permutations without changing tile identity.
            queues = [list(np.flatnonzero((ranks == rank) & (np.arange(count) % 6 == policy))) for policy in range(6)]
            for queue in queues:
                rng.shuffle(queue)
                for block, tile in enumerate(queue):
                    tile = int(tile)
                    index = block if negative == "block-index-instead-of-tile" else tile
                    timer += 1
                    starts[index], smids[index] = timer, tile % 7
                    start_writes[index] += 1;sm_writes[index] += 1
                    threads = 512 if tile % 2 else 1024
                    participants = set(range(threads))
                    is_empty = tile % 5 == 0
                    empty += int(is_empty)
                    if negative == "nonuniform-terminal-return":
                        participants.remove(threads - 1)
                    if len(participants) != threads:
                        raise AssertionError("Terminal barrier misses a CTA participant")
                    barrier_cases += 1
                    timer += threads + (0 if is_empty else 100)
                    ends[index] = timer;end_writes[index] += 1
        if any(np.any(x != 1) for x in (start_writes, end_writes, sm_writes)):
            raise AssertionError("Diagnostic arrays need exactly one actual-tile writer per macro")
        writes += int(start_writes.sum() + end_writes.sum() + sm_writes.sum())
        if np.any(ends < starts) or np.any(smids < 0):
            raise AssertionError("Incomplete last macro overwrite")
    # Public reset and a new tail-only invocation must not expose old intervals.
    if negative != "omitted-reset":
        starts.fill(0);ends.fill(0);smids.fill(-1)
    if np.any(starts) or np.any(ends) or np.any(smids != -1):
        raise AssertionError("A reset/tail-only run exposes stale prior-macro diagnostics")
    return {"counts": counts, "macros": macros, "endpoint_writes": writes,
            "converged_terminal_barriers": barrier_cases, "uniform_empty_exits": empty,
            "last_macro_single_writer_and_reset": True}


def crosscheck_DAG():
    rows = []
    for counts in ((2, 3, 4), (4, 4, 4), (5, 3, 6), (7, 4, 3)):
        slots, frontier = rank_slots(counts)
        _, ranks = tile_geometry(counts)
        rng = random.Random(312)
        for permute in (False, True):
            selected = np.asarray(slots, np.int32).copy()
            if permute:
                for rank in range(int(ranks.max()) + 1):
                    ids = np.flatnonzero(ranks == rank)
                    values = list(range(ids.size));rng.shuffle(values)
                    selected[ids] = values
            own, summary = ring_dependencies(counts, selected, frontier)
            parent = build(counts, selected, frontier)
            if own != parent["predecessors"]:
                raise AssertionError("Independent CTA analyzer and persistent DAG disagree")
            rows.append({"counts": counts, "permuted_actual_style_slots": permute,
                         "exact_predecessor_sets_agree": True, "total_edges": sum(map(len, own)),
                         "RAW_and_old_reader_WAR_and_old_writer_WAW": True})
    return rows


def prove(parent, source, output):
    parent, source, output = Path(parent), Path(source), Path(output)
    before = {n: (parent / "src" / n).read_bytes() for n in PARENT}
    expected, structure = transformed(before)
    actual = {n: (source / "src" / n).read_bytes() for n in PARENT}
    if actual != expected:
        raise ValueError("Actual diagnostic source differs from current exact-inverse producer")
    old, new = before["asymmetric_native.cuh"].decode(), actual["asymmetric_native.cuh"].decode()
    loops = []
    for name in ("UniformBody", "SourcePendingPairedBody"):
        original, diagnostic = iteration(old, name), iteration(new, name)
        if original != diagnostic or "TileInterval" in diagnostic:
            raise AssertionError("Diagnostic changed an iteration or inserted loop timing/barriers")
        loops.append({"body": name, "iteration_sha256": sha(original.encode()),
                      "all_original_loop_bytes_and_barrier_sites_identical": True})
    if (new.count("BeginTileInterval(p,tile);") != 2 or new.count("EndTileInterval(p,tiles);") != 4
            or new.count(TIMERS) != 1 or "static_cast<const volatile int32_t*>(tiles)[blockIdx.x]" not in new):
        raise AssertionError("Entry/exit/late-ID instrumentation differs")
    positives = [protocol(counts, macros) for counts, macros in itertools.product(((2, 3, 4), (4, 5, 3)), (1, 3))]
    negatives = []
    for mutation in ("block-index-instead-of-tile", "nonuniform-terminal-return", "omitted-reset"):
        try:
            protocol(negative=mutation)
        except AssertionError as error:
            negatives.append({"mutation": mutation, "rejected": True, "witness": str(error)})
        else:
            raise AssertionError("Unsafe diagnostic protocol escaped:" + mutation)
    result = {"schema": "beamz.CTA-diagnostic-source-protocol-proof.v1", "diagnostic_only": True,
              "actual_source_hashes": {n: sha(v) for n, v in actual.items()},
              "source_structure": structure, "original_iteration_bindings": loops,
              "positive_endpoint_cases": positives, "negative_endpoint_cases": negatives,
              "independent_ring_DAG_crosschecks": crosscheck_DAG(),
              "scope": "Source/unique-writer/converged-exit/reset protocol; mocked scheduling, not CUDA intrinsic timing or resource proof",
              "producer_sha256": sha(Path(__file__).read_bytes()), "GPU_executed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = prove(a.parent, a.source, a.output)
    print(json.dumps({"output": str(a.output), "original_iterations": len(result["original_iteration_bindings"]),
                      "positive_endpoint_cases": len(result["positive_endpoint_cases"]),
                      "negative_endpoint_cases": len(result["negative_endpoint_cases"]),
                      "independent_DAG_crosschecks": len(result["independent_ring_DAG_crosschecks"])}))


if __name__ == "__main__":
    main()
