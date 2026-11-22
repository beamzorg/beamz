"""Independent state-version proof for full-bank in-place first edge updates.

This oracle checks dependency closure, publication, source/observation chronology,
auxiliary recurrence clocks, odd tails and split calls on actual Yee supports.
It is not floating-point validation or an XLA input-aliasing proof; those need
the full-state GPU tests. Private mutable banks here represent FFI output banks.
"""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from beamz.lattice import component_shapes
from beamz.simulation.cuda.mstep import build_mstep_plan

FIELDS = ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")
TERMS = (
    ((5, 1), (4, 0)),
    ((3, 0), (5, 2)),
    ((4, 2), (3, 1)),
    ((2, 1), (1, 0)),
    ((0, 0), (2, 2)),
    ((1, 2), (0, 1)),
)


def _shift(value, target_shape, offset, fill):
    result = np.full(target_shape, fill, dtype=value.dtype)
    starts = [max(0, -d) for d in offset]
    ends = [min(n, value.shape[a] - offset[a]) for a, n in enumerate(target_shape)]
    if all(a < b for a, b in zip(starts, ends, strict=True)):
        dst = tuple(slice(a, b) for a, b in zip(starts, ends, strict=True))
        src = tuple(
            slice(a + d, b + d) for a, b, d in zip(starts, ends, offset, strict=True)
        )
        result[dst] = value[src]
    return result


def _fixture(pml):
    shape = (37, 65, 97) if pml else (17, 41, 73)
    supports = component_shapes(shape)
    shapes = tuple(supports[name] for name in FIELDS)
    terms = (
        tuple(
            SimpleNamespace(
                axis=axis, component=name, slab=SimpleNamespace(low=pml, high=pml)
            )
            for name in FIELDS
            for axis in {"x": (1, 0), "y": (0, 2), "z": (2, 1)}[name[1]]
        )
        if pml
        else ()
    )
    cpml = SimpleNamespace(
        h_terms=terms[: len(terms) // 2], e_terms=terms[len(terms) // 2 :]
    )
    baseline = build_mstep_plan(shapes, cpml, (None,) * 9)
    core = np.ones(baseline.shape, bool)
    core.reshape(-1)[baseline.edge_cells] = False
    source_point = tuple(np.argwhere(core)[len(np.argwhere(core)) // 2])
    group = SimpleNamespace(
        coeffs=np.ones((1, 1, 1, 1)), starts=np.asarray([source_point])
    )
    groups = (group,) + (None,) * 8
    sourced = build_mstep_plan(shapes, cpml, groups)
    core = np.ones(sourced.shape, bool)
    core.reshape(-1)[sourced.edge_cells] = False
    points = np.argwhere(core)
    assert len(points)
    # Observe core cells as well as the physical origin. Negative interpolation
    # requests must be published even when the core around them stays stale.
    monitor_points = [tuple(points[len(points) // 2]), tuple(points[-1]), (0, 0, 0)]
    indices = np.full((1, 6, 1, len(monitor_points)), -1, np.int32)
    for c, dims in enumerate((*shapes[3:], *shapes[:3])):
        for i, (z, y, x) in enumerate(monitor_points):
            indices[0, c, 0, i] = (z * dims[1] + y) * dims[2] + x
    plan = build_mstep_plan(shapes, cpml, groups, indices)
    first = np.zeros(plan.shape, bool)
    first.reshape(-1)[plan.first_cells] = True
    core = np.ones(plan.shape, bool)
    core.reshape(-1)[plan.edge_cells] = False
    allocated = np.empty(plan.shape, bool)
    spare = plan.scratch_cells // 128 - 1
    for z in range(plan.shape[0]):
        allocated[z] = np.repeat(
            np.repeat(plan.slots[z] < spare, 8, axis=0), 16, axis=1
        )[: plan.shape[1], : plan.shape[2]]
    assert np.all(allocated[first & core])
    z, y, x = source_point
    assert not core[
        max(0, z - 2) : z + 3, max(0, y - 2) : y + 3, max(0, x - 2) : x + 3
    ].any()
    masks, publications, scratch, first_masks = [], [], [], []
    auxiliary_masks = {}
    monitor_dependencies = []
    for c, dims in enumerate(shapes):
        region = tuple(slice(0, n) for n in dims)
        masks.append(~core[region])
        first_masks.append(first[region])
        publications.append((first & core)[region])
        scratch.append((allocated & core)[region])
        coordinates = np.indices(dims)
        for t, (_, axis) in enumerate(TERMS[c]):
            active = (
                (coordinates[axis] < pml) | (coordinates[axis] >= dims[axis] - pml)
                if pml
                else np.zeros(dims, bool)
            )
            assert not (active & core[region]).any()
            auxiliary_masks[c, t] = active
        for point in monitor_points:
            for offset in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
                index = tuple(a + b for a, b in zip(point, offset, strict=True))
                if all(0 <= index[a] < dims[a] for a in range(3)):
                    assert first[index]
                    monitor_dependencies.append((c, index))
    return SimpleNamespace(
        shapes=shapes,
        q=masks,
        first=first_masks,
        p=publications,
        scratch=scratch,
        auxiliary_masks=auxiliary_masks,
        source_point=source_point,
        monitors=monitor_dependencies,
    )


@dataclass
class _State:
    fields: list
    auxiliary: dict
    step: int
    sources: list
    observations: list
    bank: int = 0


def _initial(fixture, step=7):
    return _State(
        [np.full(shape, step, np.int16) for shape in fixture.shapes],
        {
            key: np.where(mask, step, 0).astype(np.int16)
            for key, mask in fixture.auxiliary_masks.items()
        },
        step,
        [],
        [],
    )


def _source(state, fixture, step, timing):
    field = 0 if timing == 1 else 3
    expected = step if timing == 0 else step + 1
    assert state.fields[field][fixture.source_point] == expected
    state.sources.append((step, timing, field, fixture.source_point))


def _phase(own, opposite, output, fixture, phase, step, masks, *, allow_taint=False):
    for c in range(3 * phase, 3 * phase + 3):
        mask, dims = masks[c], fixture.shapes[c]
        valid = own.fields[c] == step
        if not allow_taint:
            assert np.all(valid[mask]), ("own", c, step)
        for field, axis in TERMS[c]:
            required = step if phase == 0 else step + 1
            for direction in (0, 1 if phase == 0 else -1):
                offset = tuple(direction if a == axis else 0 for a in range(3))
                sampled = _shift(opposite.fields[field], dims, offset, required)
                valid &= sampled == required
                if not allow_taint:
                    assert np.all(sampled[mask] == required), (
                        "opposite",
                        c,
                        field,
                        axis,
                        direction,
                        step,
                    )
        for term in range(2):
            active = fixture.auxiliary_masks[c, term] & mask
            assert np.all(valid[active]), (
                "Invalid discarded field cannot update active CPML recurrence"
            )
            assert np.all(output.auxiliary[c, term][active] == step)
            output.auxiliary[c, term][active] = step + 1
        output.fields[c][mask] = np.where(valid[mask], step + 1, -1)


def _observe(state, fixture, step):
    assert all(state.fields[c][index] == step + 1 for c, index in fixture.monitors), (
        "missing H/E core publication at observation"
    )
    state.observations.append((step + 1, len(fixture.monitors)))


def _reference(state, fixture, steps):
    state = deepcopy(state)
    all_cells = [np.ones(shape, bool) for shape in fixture.shapes]
    for step in range(state.step, state.step + steps):
        _source(state, fixture, step, 0)
        _phase(state, state, state, fixture, 0, step, all_cells)
        _source(state, fixture, step, 1)
        _phase(state, state, state, fixture, 1, step, all_cells)
        _source(state, fixture, step, 2)
        _observe(state, fixture, step)
    state.step += steps
    return state


def _candidate(
    state, fixture, steps, *, publish_h=True, publish_e=True, second_e_from_old=False
):
    state = deepcopy(state)
    for local in range(0, steps - 1, 2):
        step = state.step + local
        _source(state, fixture, step, 0)
        # Bare core computes exact H1/E1 and H2/E2. Earlier native-core token
        # proofs establish that its owned cells and first-step values are valid.
        scratch = [
            np.where(mask, step + 1, -1).astype(np.int16) for mask in fixture.scratch
        ]
        dest = _State(
            [np.where(~mask, step + 2, -1).astype(np.int16) for mask in fixture.q],
            state.auxiliary,
            state.step,
            state.sources,
            state.observations,
            state.bank ^ 1,
        )
        for phase, publish in ((0, publish_h), (1, publish_e)):
            for c in range(3 * phase, 3 * phase + 3):
                if publish:
                    state.fields[c][fixture.p[c]] = scratch[c][fixture.p[c]]
            _phase(state, state, state, fixture, phase, step, fixture.q)
            _source(state, fixture, step, phase + 1)
        _observe(state, fixture, step)
        _source(state, fixture, step + 1, 0)
        _phase(state, state, dest, fixture, 0, step + 1, fixture.q)
        _source(dest, fixture, step + 1, 1)
        _phase(
            state,
            state if second_e_from_old else dest,
            dest,
            fixture,
            1,
            step + 1,
            fixture.q,
        )
        _source(dest, fixture, step + 1, 2)
        _observe(dest, fixture, step + 1)
        state = dest
    if steps % 2:
        step = state.step + steps - 1
        all_cells = [np.ones(shape, bool) for shape in fixture.shapes]
        _source(state, fixture, step, 0)
        _phase(state, state, state, fixture, 0, step, all_cells)
        _source(state, fixture, step, 1)
        _phase(state, state, state, fixture, 1, step, all_cells)
        _source(state, fixture, step, 2)
        _observe(state, fixture, step)
    state.step += steps
    return state


def _equal(actual, expected):
    assert actual.step == expected.step
    assert actual.sources == expected.sources
    assert actual.observations == expected.observations
    for a, b in zip(actual.fields, expected.fields, strict=True):
        np.testing.assert_array_equal(a, b)
    for key, value in actual.auxiliary.items():
        np.testing.assert_array_equal(value, expected.auxiliary[key])


@pytest.mark.parametrize("pml", [0, 12])
@pytest.mark.parametrize("steps", [2, 3, 6, 7])
def test_inplace_first_edges_publish_every_required_time_version(pml, steps):
    fixture = _fixture(pml)
    initial = _initial(fixture)
    actual = _candidate(initial, fixture, steps)
    _equal(actual, _reference(initial, fixture, steps))
    assert actual.bank == (steps // 2) % 2


@pytest.mark.parametrize("pml", [0, 12])
def test_nonzero_clock_split_calls_preserve_recurrence_and_observation_order(pml):
    fixture = _fixture(pml)
    initial = _initial(fixture)
    actual = _candidate(_candidate(initial, fixture, 3), fixture, 4)
    _equal(actual, _reference(initial, fixture, 7))


@pytest.mark.parametrize(
    "fault", [{"publish_h": False}, {"publish_e": False}, {"second_e_from_old": True}]
)
def test_missing_core_publication_or_wrong_second_bank_is_rejected(fault):
    fixture = _fixture(12)
    with pytest.raises(AssertionError):
        _candidate(_initial(fixture), fixture, 2, **fault)
