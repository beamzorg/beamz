"""Analyze diagnostic per-CTA moving-box records; cycles include profiler overhead.

Barriers are existing program barriers. Thread0 clocks measure its path and wait;
publication by slower warps can appear in the following H wait. Setup isolates
common metadata only; update totals still include per-cell address calculations.
Global timestamps delimit a thread0 residency proxy, not exact warp occupancy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
from collections import defaultdict
from pathlib import Path

PHASES = (
    "init",
    "h_setup",
    "h_update_and_address",
    "h_wait",
    "imports_and_address",
    "import_wait",
    "e_setup",
    "e_update_and_address",
    "e_wait",
    "publication_and_address",
    "loop",
)


def summarize(path, size, steps, tile):
    bz, by, bx = tile
    nx = (size + steps + bx) // bx
    ny = (size + steps + by) // by
    nz = (size + steps + bz) // bz
    rows = path.read_bytes()
    record = struct.Struct("<19Q")
    if len(rows) != nx * ny * nz * record.size:
        raise ValueError("Record file length differs from exact geometry")
    sums = defaultdict(lambda: [0] * len(PHASES))
    counts = defaultdict(int)
    intervals = defaultdict(list)
    rank_span = {}
    seen = set()
    unaccounted = []
    total_cycles = []
    for row in record.iter_unpack(rows):
        cb, ce, gb, ge, sm, rank, index, observed_steps, *phases = row
        if index in seen or observed_steps != steps or ce < cb or ge < gb:
            raise ValueError("Invalid/duplicate record")
        seen.add(index)
        z = index // (nx * ny)
        y = (index // nx) % ny
        x = index % nx
        if rank != z + y + x:
            raise ValueError("Recorded rank differs from geometry")
        origin = (z * bz, y * by, x * bx)
        interior = all(
            o >= steps + 1 and o + b <= size for o, b in zip(origin, tile, strict=True)
        )
        for group in ("all", "interior" if interior else "boundary"):
            counts[group] += 1
            for i, v in enumerate(phases):
                sums[group][i] += v
        intervals[sm].append((gb, ge))
        rank_span[rank] = (
            min(gb, rank_span.get(rank, (gb, ge))[0]),
            max(ge, rank_span.get(rank, (gb, ge))[1]),
        )
        unaccounted.append(ce - cb - sum(phases))
        total_cycles.append(ce - cb)
    first = min(a for v in intervals.values() for a, _ in v)
    last = max(b for v in intervals.values() for _, b in v)

    def union(v):
        total = 0
        lo = hi = None
        for a, b in sorted(v):
            if lo is None:
                lo, hi = a, b
            elif a > hi:
                total += hi - lo
                lo, hi = a, b
            else:
                hi = max(hi, b)
        return total + (hi - lo if hi is not None else 0)

    rank_intervals = sorted(rank_span.values())
    rank_gaps = [
        max(0, b[0] - a[1])
        for a, b in zip(rank_intervals, rank_intervals[1:], strict=False)
    ]
    return {
        "scope": __doc__,
        "records": len(seen),
        "tile_zyx": tile,
        "size": size,
        "steps": steps,
        "groups": {
            g: {
                "boxes": counts[g],
                "phase_cycle_sums": dict(zip(PHASES, v, strict=True)),
                "phase_fractions": dict(
                    zip(PHASES, [x / sum(v) for x in v], strict=True)
                ),
            }
            for g, v in sums.items()
        },
        "median_cta_cycles": statistics.median(total_cycles),
        "median_unassigned_final_cycles": statistics.median(unaccounted),
        "thread0_global_span_ms": (last - first) / 1e6,
        "rank_gap_ms": sum(rank_gaps) / 1e6,
        "rank_gap_fraction_of_span": sum(rank_gaps) / (last - first),
        "active_recorded_sms": len(intervals),
        "sm_thread0_window_coverage_mean": statistics.mean(
            union(v) / (last - first) for v in intervals.values()
        ),
        "timings_are_instrumented": True,
        "raw": str(path.resolve()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--tile", nargs=3, type=int, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    d = summarize(a.records, a.size, a.steps, a.tile)
    s = json.dumps(d, indent=2)
    if a.output:
        a.output.write_text(s + "\n")
    else:
        print(s)


if __name__ == "__main__":
    main()
