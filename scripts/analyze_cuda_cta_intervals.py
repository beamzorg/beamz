"""Analyze instrumented CTA presence and exact conservative ring dependencies.

Intervals come from a perturbing diagnostic build. Presence is not instruction
utilization, occupancy, or a speed prediction. SM attribution uses startSMID.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def tile_geometry(counts):
    counts = tuple(int(n) for n in counts)
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError("Positive3D tile geometry required")
    ids = np.arange(np.prod(counts), dtype=np.int64)
    z = ids // (counts[1] * counts[2])
    y = ids // counts[2] % counts[1]
    x = ids % counts[2]
    return np.stack((z, y, x), axis=1), z + y + x


def ring_dependencies(counts, slots, frontier, epochs=4):
    """RAW neighbors plus old-consumer→new-owner WAR and writer→writer edges."""
    coordinates, ranks = tile_geometry(counts)
    slots = np.asarray(slots)
    if epochs != 4 or slots.dtype != np.int32 or slots.shape != ranks.shape or frontier < 1:
        raise ValueError("Actual four-epoch slot map required")
    for rank in range(int(ranks.max()) + 1):
        values = slots[ranks == rank]
        if sorted(map(int, values)) != list(range(values.size)) or values.size > frontier:
            raise ValueError("Actual rank slots must be a unique contiguous permutation")
    count = ranks.size
    predecessors = [set() for _ in range(count)]
    consumers = [[] for _ in range(count)]
    raw_edges = 0
    nz, ny, nx = map(int, counts)
    for tile, (z, y, x) in enumerate(coordinates):
        for dz, dy, dx in itertools.product((0, 1), repeat=3):
            if not (dz or dy or dx) or z < dz or y < dy or x < dx:
                continue
            producer = int(((z - dz) * ny + y - dy) * nx + x - dx)
            predecessors[tile].add(producer)
            consumers[producer].append(tile)
            raw_edges += 1
    holders = {}
    war_candidates = war_added = waw_added = 0
    for tile in np.argsort(ranks, kind="stable"):
        tile = int(tile)
        key = (int(ranks[tile]) % epochs) * frontier + int(slots[tile])
        if key in holders:
            old = holders[key]
            if int(ranks[tile] - ranks[old]) < epochs:
                raise ValueError("Trail slot is reused before four ranks elapse")
            if old not in predecessors[tile]:
                predecessors[tile].add(old)
                waw_added += 1
            for consumer in consumers[old]:
                war_candidates += 1
                if ranks[consumer] >= ranks[tile]:
                    raise ValueError("Ring anti-dependency is not forward in rank")
                if consumer not in predecessors[tile]:
                    predecessors[tile].add(consumer)
                    war_added += 1
        holders[key] = tile
    for tile, before in enumerate(predecessors):
        if any(ranks[p] >= ranks[tile] for p in before):
            raise ValueError("Ready DAG must remain acyclic and strictly forward in rank")
    return predecessors, {"raw_neighbor_edges": raw_edges, "old_consumer_WAR_candidates": war_candidates,
                          "new_direct_WAR_edges": war_added, "new_direct_WAW_edges": waw_added,
                          "acyclic_by_strict_rank": True, "physical_trail_epochs": epochs,
                          "scope": "Conservative all7geometric neighbors; includes slot-reuse hazards; does not remove exact input/final field or mutable-state anti-dependencies for future alias modes"}


def union_length(starts, ends, low, high):
    pairs = sorted((max(int(s), low), min(int(e), high)) for s, e in zip(starts, ends, strict=True) if int(e) > low and int(s) < high)
    total, tail = 0, low
    for start, end in pairs:
        if end <= start:
            continue
        if start > tail:
            total += end - start
        elif end > tail:
            total += end - tail
        tail = max(tail, end)
    return total


def policy(flags):
    value = int(flags) & 63
    if value == 0:
        return 0
    if not value & 17:
        return 1
    return {1: 2, 16: 3, 17: 4}.get(value, 5)


def valid_component_work(counts, physical_shape, depth):
    coordinates, ranks = tile_geometry(counts)
    origin = coordinates * np.asarray((7, 15, 31))
    owned = np.asarray((7, 15, 31))
    support = np.asarray(((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)))
    result = np.zeros(ranks.size, np.int64)
    for version in range(1, int(depth) + 1):
        low, high = origin - version, origin - version + owned - 1
        for delta in support:
            extent = np.asarray(physical_shape) + delta
            widths = np.maximum(0, np.minimum(high, extent - 1) - np.maximum(low, 0) + 1)
            result += np.prod(widths, axis=1)
    return result


def summarize(arrays, metadata, physical_shape=None):
    starts = np.asarray(arrays["start_ns"])
    ends = np.asarray(arrays["end_ns"])
    smids = np.asarray(arrays["smid"])
    flags = np.asarray(arrays["tile_flags"])
    counts = tuple(metadata["counts_zyx"])
    coordinates, ranks = tile_geometry(counts)
    count = ranks.size
    if any(a.shape != (count,) for a in (starts, ends, smids, flags)):
        raise ValueError("One start/end/startSMID/flag value per actual tile required")
    if starts.dtype != np.uint64 or ends.dtype != np.uint64 or smids.dtype != np.int32:
        raise ValueError("Exact uint64 globaltimer and int32 SMID arrays required")
    if np.any(starts == 0) or np.any(ends < starts) or np.any(smids < 0):
        raise ValueError("Missing, stale or reversed last-macro CTA intervals")
    ids = np.asarray(arrays["tile_ids"])
    if sorted(map(int, ids)) != list(range(count)):
        raise ValueError("Actual launch plan must cover every tile exactly once")
    offsets = np.asarray(arrays["rank_offsets"])
    sizes = np.bincount(ranks)
    expected_offsets = np.concatenate(([0], np.cumsum(sizes)))
    if (offsets.shape != expected_offsets.shape or not np.array_equal(offsets, expected_offsets)
            or not np.array_equal(ranks[ids], np.repeat(np.arange(sizes.size), sizes))):
        raise ValueError("Actual tile IDs and rank ranges do not match diagnostic geometry")
    offset = int(starts.min())
    relative_starts = (starts - np.uint64(offset)).astype(np.int64)
    relative_ends = (ends - np.uint64(offset)).astype(np.int64)
    durations = relative_ends - relative_starts
    span = int(relative_ends.max())
    if span < 1:
        raise ValueError("Positive interval envelope required")
    sms = sorted(map(int, np.unique(smids)))
    physical_sms = int(metadata["physical_SM_count"])
    if physical_sms < len(sms) or physical_sms < 1:
        raise ValueError("Observed SMID cardinality exceeds physical SM count")
    per_sm = []
    central_low, central_high = span // 10, span - span // 10
    central_total = 0
    for sm in sms:
        selected = smids == sm
        present = union_length(relative_starts[selected], relative_ends[selected], 0, span)
        central = union_length(relative_starts[selected], relative_ends[selected], central_low, central_high)
        central_total += central
        per_sm.append({"smid": sm, "CTA_count": int(selected.sum()), "recorded_CTA_presence_ns": present,
                       "fraction_envelope_with_recorded_CTA": present / span})
    per_rank = []
    rank_violations = 0
    previous_end = 0
    for rank in range(int(ranks.max()) + 1):
        selected = ranks == rank
        begin, end = int(relative_starts[selected].min()), int(relative_ends[selected].max())
        gap = begin - previous_end if rank else 0
        rank_violations += int(gap < 0)
        per_rank.append({"rank": rank, "tile_count": int(selected.sum()), "first_start_relative_ns": begin,
                         "last_end_relative_ns": end, "gap_after_previous_rank_ns": gap})
        previous_end = end
    policies = np.asarray([policy(f) for f in flags])
    per_policy = []
    work = valid_component_work(counts, physical_shape, metadata["depth"]) if physical_shape is not None else None
    for p in range(6):
        selected = policies == p
        values = durations[selected]
        if not values.size:
            continue
        row = {"policy": p, "CTA_count": int(values.size), "median_CTA_duration_ns": float(np.median(values)),
               "p95_CTA_duration_ns": float(np.percentile(values, 95)), "max_CTA_duration_ns": int(values.max())}
        if work is not None:
            row["valid_Yee_component_updates"] = int(work[selected].sum())
            row["zero_component_work_CTA_count"] = int(np.count_nonzero(work[selected] == 0))
        per_policy.append(row)
    before, dag = ring_dependencies(counts, arrays["slots"], int(metadata["max_frontier"]), int(metadata["epochs"]))
    finish = np.zeros(count, np.int64)
    dependency_violations = 0
    for tile in np.argsort(ranks, kind="stable"):
        preds = before[int(tile)]
        dependency_violations += int(bool(preds) and max(int(relative_ends[p]) for p in preds) > int(relative_starts[tile]))
        finish[tile] = int(durations[tile]) + max((int(finish[p]) for p in preds), default=0)
    result = {"schema": "beamz.perturbing-CTA-interval-analysis.v1", "diagnostic_only": True,
              "no_performance_candidate_or_speedup_claim": True,
              "tile_count": count, "physical_SM_count": physical_sms, "observed_start_SMIDs": sms,
              "SM_count_without_recorded_start": physical_sms - len(sms),
              "last_macro_CTA_envelope_ns": span, "recorded_CTA_presence_SM_ns": sum(r["recorded_CTA_presence_ns"] for r in per_sm),
              "fraction_SM_envelope_with_recorded_CTA_present": sum(r["recorded_CTA_presence_ns"] for r in per_sm) / (physical_sms * span),
              "central10to90pct_CTA_presence_fraction": central_total / (physical_sms * (central_high - central_low)),
              "rank_order_violations": rank_violations, "conservative_dependency_timestamp_violations": dependency_violations,
              "per_SM": per_sm, "per_rank": per_rank, "per_policy": per_policy, "conservative_ready_DAG": dag,
              "fixed_duration_unlimited_workers_DAG_critical_path_ns": int(finish.max()),
              "DAG_duration_model_is_not_a_hardware_bound": True,
              "limitations": ["Timerstores, extraCTA-endbarriers and parameter pointers alter execution/resources",
                              "Interval presence is not SM instruction utilization or occupancy; memory/barrier stalls remain inside intervals",
                              "SMID recorded only atstart; migration/preemption cannot be reconstructed",
                              "One last moving macro; noDFT/tail/clock coverage",
                              "Fixed measured tile durations depend on original contention; DAG critical path is an illustrative scheduling model",
                              "Persistent execution additionally needs resident-worker deadlock avoidance, release/acquire publication, epoch tags and strict observation ordering"]}
    if work is not None:
        result["productive_CTA_count"] = int(np.count_nonzero(work))
        result["zero_component_work_CTA_count"] = int(np.count_nonzero(work == 0))
        result["valid_Yee_component_updates"] = int(work.sum())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arrays", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--physical-shape", type=int, nargs=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.arrays, allow_pickle=False) as arrays:
        result = summarize(arrays, json.loads(a.metadata.read_text()), a.physical_shape)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("tile_count", "last_macro_CTA_envelope_ns", "fraction_SM_envelope_with_recorded_CTA_present", "rank_order_violations")}))


if __name__ == "__main__":
    main()
