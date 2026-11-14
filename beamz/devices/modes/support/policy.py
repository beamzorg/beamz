"""Dimension-independent port bounds and candidate ordering policy."""

from __future__ import annotations


def solve_interval(start, stop, limit, padding=0, min_cells=2):
    """Pad outward-snapped cell bounds, clip, and retain minimum solver support."""
    start = max(0, min(int(start) - int(padding), int(limit)))
    stop = max(start, min(int(stop) + int(padding), int(limit)))
    need = max(1, int(min_cells))
    while stop - start < need and (start > 0 or stop < int(limit)):
        if start > 0:
            start -= 1
        if stop - start < need and stop < int(limit):
            stop += 1
    return start, stop


def candidate_rank(
    neff, *, polarization_matches=True, aperture_fraction=1.0, minimum=None
):
    """Prefer polarization, then optional confinement, then decreasing real neff."""
    return (
        bool(polarization_matches),
        minimum is None or aperture_fraction >= minimum,
        float(complex(neff).real),
    )
