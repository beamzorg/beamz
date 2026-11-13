"""Independent footprint proof for optional traced compact queue tightening."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cuda_mstep_dependencies import intermediate_offsets, tighten_first_queue
from tests.unit.test_cuda_mstep_plan import _plan


def _coordinates(row, shape):
    z, rest = divmod(int(row), shape[1] * shape[2])
    y, x = divmod(rest, shape[2])
    return z, y, x


@pytest.mark.parametrize("shape", [(7, 17, 33), (17, 33, 49), (31, 41, 57)])
@pytest.mark.parametrize("source", [False, True])
def test_traced_first_queue_matches_independent_yee_read_closure(shape, source):
    group = SimpleNamespace(
        coeffs=np.ones((1, 1, 1, 1)), starts=np.array([[5, 16, 32]])
    )
    original, fields = _plan(shape, groups=(group if source else None,) + (None,) * 8)
    candidate = tighten_first_queue(original, fields)
    # Independently compose forward E1 support and backward H1 dependencies.
    axes = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    required = set()
    for row in original.edge_cells:
        coordinate = _coordinates(row, original.shape)
        for forward in axes:
            for backward in axes:
                point = tuple(
                    p + a - b
                    for p, a, b in zip(coordinate, forward, backward, strict=True)
                )
                if all(0 <= p < n for p, n in zip(point, original.shape, strict=True)):
                    required.add(
                        (point[0] * original.shape[1] + point[1]) * original.shape[2]
                        + point[2]
                    )
    assert set(candidate.first_cells.tolist()) == required
    assert required <= set(original.first_cells.tolist())
    assert set(original.edge_cells.tolist()) <= required
    assert not candidate.first_cells.flags.writeable
    assert candidate.slots is original.slots
    assert candidate.edge_cells is original.edge_cells
    assert candidate.scratch_cells == original.scratch_cells


def test_traced_intermediate_envelope_avoids_chebyshev_corner_expansion():
    offsets = set(intermediate_offsets())
    assert len(offsets) == 13
    assert (0, 1, 1) not in offsets
    assert (0, -1, -1) not in offsets
    assert (1, -1, 0) in offsets
    assert (1, 1, 1) not in offsets


def test_monitor_interpolation_and_backward_dependencies_survive_tightening():
    base, fields = _plan((31, 41, 57))
    indices = np.full((1, 6, 1, 2), -1, np.int32)
    indices[0, 0, 0, 0] = (16 * fields[3][1] + 24) * fields[3][2] + 32
    indices[0, 5, 0, 1] = (17 * fields[2][1] + 25) * fields[2][2] + 33
    original, _ = _plan((31, 41, 57), indices=indices)
    candidate = tighten_first_queue(original, fields, indices)
    rows = set(candidate.first_cells.tolist())
    ny, nx = base.shape[1:]
    for center in ((16, 24, 32), (17, 25, 33)):
        for delta in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
            z, y, x = (a + b for a, b in zip(center, delta, strict=True))
            assert (z * ny + y) * nx + x in rows
    assert rows <= set(original.first_cells.tolist())
