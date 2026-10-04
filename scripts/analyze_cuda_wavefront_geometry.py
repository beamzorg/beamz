"""Geometry-only comparison of overlapping CUDA tiles and a staircase DAG.

This checks coverage and counts potential work slots; it does not simulate
Maxwell updates, validate memory ordering, or predict GPU throughput.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def analyze():
    checks = 0
    for n in (1, 7, 12, 13, 16, 17, 31, 32, 33, 63, 64, 65, 97, 256, 1024):
        for size in (2, 4, 8, 12, 16):
            for step in range(size):
                counts = [0] * n
                for block in range(math.ceil(n / size) + 1):
                    for cell in range(
                        max(0, block * size - step),
                        min(n, (block + 1) * size - step),
                    ):
                        counts[cell] += 1
                assert counts == [1] * n, (n, size, step)
                checks += 1
    halos = []
    for x, y, z in ((16, 8, 16), (15, 16, 16), (32, 4, 16), (16, 8, 4)):
        stages = [(x + 3 - k) * (y + 3 - k) * (z + 3 - k) for k in range(4)]
        halos.append(
            dict(
                tile_xyz=[x, y, z],
                stage_cell_slots=stages,
                owned_four_phase_cell_updates=4 * x * y * z,
                issued_slot_ratio=sum(stages) / (4 * x * y * z),
            )
        )
    widths = []
    tile = (16, 8, 16)
    for shape in ((128, 256, 512), (1024, 256, 64), (97, 289, 593), (64, 256, 1024)):
        grid = tuple(math.ceil(n / t) for n, t in zip(shape, tile, strict=True))
        spatial = Counter(
            a + b + c
            for a in range(grid[0])
            for b in range(grid[1])
            for c in range(grid[2])
        )
        # Spatial edges increment x+y+z by one. The temporal edge moves
        # (-1,-1,-1,+1 pass), so x+y+z+4*pass increases by one as well.
        pipelined = Counter()
        for time_pass in range(8):
            for rank, count in spatial.items():
                pipelined[rank + 4 * time_pass] += count
        widths.append(
            dict(
                logical_zyx=shape,
                tile_zyx=tile,
                block_grid_zyx=grid,
                single_pass_max_rank_width=max(spatial.values()),
                eight_pass_max_rank_width=max(pipelined.values()),
            )
        )
    return dict(
        note=(
            "Analytical geometry model, not a CUDA benchmark or scheduler "
            "correctness proof. Rank widths exclude extra halo blocks and "
            "runtime clipping. CUDA slot ratios are unclipped tile envelopes, "
            "not instruction counts."
        ),
        cpu_window_coverage_checks=checks,
        halo_envelopes=halos,
        wavefront_rank_widths=widths,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(), indent=2) + "\n")
