"""Prove sparse low-X/high-X private staging reuse with a real warp fence.

The low-X source owner and high-X writer share physical Y/Z and GPU warp.
__shfl_sync is NOT a memory-ordering fence; __syncwarp after all low private
loads orders both Hy and Ey stage reads before another lane's high writes.
B3 publishes high packets and next B1 retires their reads before reuse.
Only the original three-CTA-barrier circular schedule is admitted here.
Source: https://docs.nvidia.com/cuda/archive/13.2.2/cuda-programming-guide/05-appendices/cpp-language-extensions.html
"""

from __future__ import annotations

import json


def prove(steps=16, *, warp_fence=True, omit_ghost_lane=False,
          compact_high_key=False, high_read_before_b3=False,
          next_import_before_b1=False):
    bz, by, bx, cz, cy, cx = 5, 15, 31, 6, 16, 32
    low_count, high_count = bz * by, (bz - 1) * (by - 1)
    checks = 0
    for n in range(1, steps + 1):
        hz, hy, hx = (-n - 1) % cz, (-n - 1) % cy, (-n - 1) % cx
        low = {}
        for k in range(low_count):
            lz, ly = divmod(k, by)
            j, sy = (hz + lz + 1) % cz, (hy + ly + 1) % cy
            reader_thread = sy * cx + hx
            low[k] = (j, sy, reader_thread)
        written = set()
        for t in range(high_count):
            lz, ly = divmod(t, by - 1)
            j, sy = (hz + lz + 1) % cz, (hy + ly + 1) % cy
            writer_thread = sy * cx + (hx + bx) % cx
            k = lz * (by - 1 if compact_high_key else by) + ly
            assert k not in written
            written.add(k)
            source_j, source_y, reader = low[k]
            assert (source_j, source_y) == (j, sy), "Sparse high key changed physical YZ"
            assert reader // 32 == writer_thread // 32, "Staging reuse crosses GPU warps"
            participants = set(range(32)) - ({hx} if omit_ghost_lane else set())
            assert reader % 32 in participants and writer_thread % 32 in participants, "Warp fence excludes staging reader/writer"
            # Both shared Hy/Ey loads retire. Hy shuffle dependence alone does
            # not order Ey, which is unused until the following H phase.
            assert warp_fence, "Ey stage load has no shared-memory read->write fence"
            assert not high_read_before_b3, "Coalesced high reader precedes its CTA publication barrier"
            if n < steps:
                assert not next_import_before_b1, "Next import overwrites staging before prior high readers retire"
            checks += 2
    assert low_count == 75 and high_count == 56
    return {"steps": steps, "low_points": low_count, "high_points": high_count,
            "stage_bytes": 2 * low_count * 4, "ordered_component_reuses": checks,
            "required_warp_fence": "full-mask __syncwarp after ALL low Hy/Ey loads, outside predicates",
            "required_CTA_order": "B2 publishes low; B3 publishes high; next B1 retires high readers",
            "result": "pass"}


def main():
    cases = [prove(n) for n in (2, 4, 8, 16, 32)]
    negatives = []
    for name, options in (("Hy shuffle alone", {"warp_fence": False}),
                          ("ghost lane excluded", {"omit_ghost_lane": True}),
                          ("compact rather than sparse key", {"compact_high_key": True}),
                          ("high read before B3", {"high_read_before_b3": True}),
                          ("next import before B1", {"next_import_before_b1": True})):
        try:
            prove(**options)
        except AssertionError:
            negatives.append(name)
        else:
            raise AssertionError(f"Negative control escaped: {name}")
    print(json.dumps({"scope": __doc__, "cases": cases, "negative_controls_rejected": negatives}, indent=2))


if __name__ == "__main__":
    main()
