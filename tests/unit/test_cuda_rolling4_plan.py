"""Independent dependency-envelope checks for the four-step interior planner."""

from __future__ import annotations

import numpy as np

from scripts.cuda_rolling4_plan import build_rolling4_plan
from scripts.prove_cuda_rolling4_plan import closure, prove


def shapes(n):
    return [
        (n, n, n + 1),
        (n, n + 1, n),
        (n + 1, n, n),
        (n + 1, n + 1, n),
        (n + 1, n, n + 1),
        (n, n + 1, n + 1),
    ]


def expand3(mask, radius):
    result = mask.copy()
    for axis in range(3):
        expanded = np.zeros_like(result)
        for shift in range(-radius, radius + 1):
            source = [slice(None)] * 3
            target = [slice(None)] * 3
            if shift < 0:
                source[axis] = slice(-shift, None)
                target[axis] = slice(None, shift)
            elif shift > 0:
                source[axis] = slice(None, -shift)
                target[axis] = slice(shift, None)
            expanded[tuple(target)] |= result[tuple(source)]
        result = expanded
    return result


def test_four_step_q_envelope_matches_independent_3d_dilation():
    plan = build_rolling4_plan(shapes(64), 12, [])
    q = np.zeros(plan.shape, bool)
    q.reshape(-1)[plan.edge_cells] = True
    s = np.zeros(plan.shape, bool)
    s.reshape(-1)[plan.first_cells] = True
    assert np.array_equal(s, expand3(q, 3))
    assert not np.array_equal(s, expand3(q, 2))
    assert plan.core_cells == q.size - np.count_nonzero(q)
    assert np.all(np.diff(plan.first_cells) > 0) and np.all(
        np.diff(plan.edge_cells) > 0
    )


def test_observation_envelope_contains_every_early_yee_dependency():
    dims = shapes(96)
    point = (48, 48, 48)
    absent = build_rolling4_plan(dims, 12, [])
    assert np.ravel_multi_index(point, absent.shape) not in absent.first_cells
    indices = np.full((1, 6, 1, 1), -1, np.int32)
    indices[0, 0, 0, 0] = np.ravel_multi_index(point, dims[3])
    plan = build_rolling4_plan(dims, 12, [], indices)
    s = np.zeros(plan.shape, bool)
    s.reshape(-1)[plan.first_cells] = True
    for version in [1, 2, 3]:
        for family, stage, component, relative in closure(1, version, 0, (0, 0, 0)):
            if stage:
                physical = tuple(point[a] + relative[a] for a in range(3))
                assert s[physical], (family, stage, component, physical)


def test_source_halos_and_cpml_are_absent_from_bulk_update_support():
    dims = shapes(96)
    start = np.array([[40, 40, 40]], np.int32)
    extent = (1, 2, 3, 1)
    plan = build_rolling4_plan(dims, 12, [(extent, start)])
    required = set().union(
        *(closure(f, 4, c, (0, 0, 0)) for f in [0, 1] for c in range(3))
    )
    updates = [node for node in required if node[1] > 0]
    for ty, tx in np.argwhere(plan.core_masks):
        mask = plan.core_masks[ty, tx]
        for bit in range(4):
            if not (mask & (1 << bit)):
                continue
            lo = np.array(
                [plan.bounds[0], ty * 16 + (bit // 2) * 8, tx * 32 + (bit % 2) * 16]
            )
            hi = lo + np.array([plan.bounds[1] - plan.bounds[0] - 1, 7, 15])
            for family, _stage, component, offset in updates:
                minimum = lo + offset
                maximum = hi + offset
                assert np.all(minimum >= 12)
                assert np.all(maximum < np.array(dims[3 * family + component]) - 12)
                end = start[0] + np.array(extent[1:])
                assert not (np.all(minimum < end) and np.all(maximum >= start[0]))


def test_symbolic_proof_has_a_radius_two_counterexample():
    result = prove()
    assert result["passed"]
    assert result["core_updated_support_min"] == [-4] * 3
    assert result["negative_radius2_missing_node"]
