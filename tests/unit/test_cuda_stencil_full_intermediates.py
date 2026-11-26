"""Taint proof for full-bank first phases without temporal-core publication.

Not every computed E1 halo cell is valid or needed. This independently infers
the required E1 region from Q's actual H2 stencils, E2's own values, and DFT
requests, then permits taint only outside that region and inactive CPML state.
No native implementation is imported; this proves the proposed schedule.
"""

from copy import deepcopy

import numpy as np
import pytest

from tests.unit.test_cuda_stencil_inplace_edges import (
    TERMS,
    _equal,
    _fixture,
    _initial,
    _observe,
    _phase,
    _reference,
    _shift,
    _source,
    _State,
)


def _required_e1(fixture):
    required = [np.zeros(shape, bool) for shape in fixture.shapes]
    for c in range(3, 6):
        required[c] |= fixture.q[c]
    for c in range(3):
        for field, axis in TERMS[c]:
            for direction in (0, 1):
                # Scatter required Q output positions into source E supports.
                offset = tuple(-direction if a == axis else 0 for a in range(3))
                required[field] |= _shift(
                    fixture.q[c], fixture.shapes[field], offset, False
                )
    for c, point in fixture.monitors:
        if c >= 3:
            required[c][point] = True
    return required


def _prove_closure(fixture):
    required = _required_e1(fixture)
    for c in range(3, 6):
        assert np.all(fixture.first[c][required[c]]), (
            "Required E1 missing from first queue"
        )
        for field, axis in TERMS[c]:
            for direction in (0, -1):
                offset = tuple(direction if a == axis else 0 for a in range(3))
                supported = _shift(
                    fixture.first[field], fixture.shapes[c], offset, True
                )
                assert np.all(supported[required[c]]), "Required E1 reads uncomputed H1"
    for c, point in fixture.monitors:
        assert fixture.first[c][point]
    return required


def _candidate_full_first(
    state, fixture, steps, *, drop_halo_h=False, wrong_second_bank=False
):
    state = deepcopy(state)
    required = _prove_closure(fixture)
    tainted_cells = []
    for local in range(0, steps - 1, 2):
        step = state.step + local
        _source(state, fixture, step, 0)
        dest = _State(
            [np.where(~mask, step + 2, -1).astype(np.int16) for mask in fixture.q],
            state.auxiliary,
            state.step,
            state.sources,
            state.observations,
            state.bank ^ 1,
        )
        first_h = fixture.first
        if drop_halo_h:
            first_h = [mask & fixture.q[c] for c, mask in enumerate(fixture.first)]
        _phase(state, state, state, fixture, 0, step, first_h)
        _source(state, fixture, step, 1)
        _phase(state, state, state, fixture, 1, step, fixture.first, allow_taint=True)
        _source(state, fixture, step, 2)
        invalid = sum(
            int(np.count_nonzero((state.fields[c] == -1) & fixture.first[c]))
            for c in range(3, 6)
        )
        tainted_cells.append(invalid)
        for c in range(3, 6):
            assert np.all(state.fields[c][required[c]] == step + 1), (
                "Invalid E1 reached required region"
            )
        _observe(state, fixture, step)
        _source(state, fixture, step + 1, 0)
        _phase(state, state, dest, fixture, 0, step + 1, fixture.q)
        _source(dest, fixture, step + 1, 1)
        _phase(
            state,
            state if wrong_second_bank else dest,
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
    return state, tainted_cells


@pytest.mark.parametrize("pml", [0, 12])
@pytest.mark.parametrize("steps", [2, 3, 6, 7])
def test_full_first_queue_no_publication_preserves_required_versions(pml, steps):
    fixture = _fixture(pml)
    initial = _initial(fixture)
    actual, discarded = _candidate_full_first(initial, fixture, steps)
    _equal(actual, _reference(initial, fixture, steps))
    assert actual.bank == (steps // 2) % 2
    assert all(count > 0 for count in discarded), (
        "Fixture must exercise invalid discarded E1"
    )


@pytest.mark.parametrize("pml", [0, 12])
def test_full_first_queue_odd_split_and_nonzero_origin(pml):
    fixture = _fixture(pml)
    initial = _initial(fixture, step=13)
    first, _ = _candidate_full_first(initial, fixture, 3)
    split, _ = _candidate_full_first(first, fixture, 4)
    _equal(split, _reference(initial, fixture, 7))


@pytest.mark.parametrize("fault", [{"drop_halo_h": True}, {"wrong_second_bank": True}])
def test_missing_required_h1_or_wrong_second_h2_bank_is_rejected(fault):
    fixture = _fixture(12)
    with pytest.raises(AssertionError):
        _candidate_full_first(_initial(fixture), fixture, 2, **fault)


def test_source_and_monitor_requests_are_essential_to_closure():
    fixture = _fixture(12)
    # A deep observed E1 sample can require an H1 neighbour inside the core.
    required = _prove_closure(fixture)
    changed = deepcopy(fixture)
    for c in range(3, 6):
        for field, axis in TERMS[c]:
            offset = tuple(1 if a == axis else 0 for a in range(3))
            needed_h = _shift(required[c], fixture.shapes[field], offset, False)
            vulnerable = needed_h & fixture.first[field] & ~fixture.q[field]
            locations = np.argwhere(vulnerable)
            if len(locations):
                changed.first[field][tuple(locations[0])] = False
                with pytest.raises(AssertionError, match="uncomputed H1"):
                    _prove_closure(changed)
                return
    raise AssertionError("Fixture lacks a core H1 monitor dependency")
