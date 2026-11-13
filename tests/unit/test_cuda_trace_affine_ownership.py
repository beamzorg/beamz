"""Illegal retiming, missing publications and lattice parity must be rejected."""
import copy

import pytest

from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.cuda_trace_affine_ownership import (
    BOX_BASIS, DIAMOND_XY_BASIS, prove, solve_retiming, synthesize, verify_transform,
)


def test_existing_box_schedule_is_rediscovered():
    plan = synthesize()
    assert plan['time_skew'] == [1, 1, 1]
    assert set(plan['tensor_retiming'].values()) == {(0, 0, 0)}


def test_rotated_xy_schedule_needs_component_retiming():
    plan = synthesize(DIAMOND_XY_BASIS)
    assert plan['determinant'] == 2
    assert plan['time_skew'] == [1, 1, 1]
    assert len(set(plan['tensor_retiming'].values())) > 1
    operator = yee_operator(cpml_axes=(0, 1, 2))
    bad = {name:(0, 0, 0) for name in plan['tensor_retiming']}
    with pytest.raises(IllegalSchedule, match='Future-owner'):
        verify_transform(operator, DIAMOND_XY_BASIS, plan['time_skew'], bad)


def test_zero_time_skew_exposes_a_dependency_cycle():
    with pytest.raises(IllegalSchedule, match='cycle'):
        solve_retiming(yee_operator(cpml_axes=(0, 1, 2)), (0, -1, 1), 0)


@pytest.mark.parametrize('basis,tile', [(BOX_BASIS, (2, 3, 4)),
                                      (DIAMOND_XY_BASIS, (2, 3, 8))])
@pytest.mark.parametrize('steps', [1, 4, 8])
def test_every_physical_yee_and_cpml_update_has_one_legal_owner(basis, tile, steps):
    plan = synthesize(basis)
    result = prove(plan, shape=(2, 3, 4), tile=tile, steps=steps)
    assert result['unique_updates'] > 0
    assert result['executed_over_useful_updates'] == 1
    assert result['result'] == 'pass'


def test_missing_boundary_publication_is_detected():
    with pytest.raises(IllegalSchedule, match='publication'):
        prove(synthesize(DIAMOND_XY_BASIS), shape=(2, 3, 4),
              tile=(2, 3, 4), steps=4, drop_publication=True)


def test_psi_cannot_be_detached_from_its_field_owner():
    plan = synthesize(DIAMOND_XY_BASIS)
    bad = copy.deepcopy(plan['tensor_retiming'])
    bad['psi_H0'] = tuple(v+1 for v in bad['psi_H0'])
    with pytest.raises(IllegalSchedule):
        verify_transform(yee_operator(cpml_axes=(0, 1, 2)), plan['spatial_basis'],
                         plan['time_skew'], bad)


@pytest.mark.parametrize('basis', [((1, 0, 0),)*3, ((1, 0), (0, 1)),
                                   ((0, 0, 0),)*3])
def test_singular_or_incomplete_spatial_basis_is_rejected(basis):
    with pytest.raises(IllegalSchedule):
        synthesize(basis)


def test_fractional_retiming_cannot_enter_an_integer_owner_map():
    plan=synthesize()
    bad=dict(plan['tensor_retiming']);bad['Hx']=(0.5,0,0)
    with pytest.raises(IllegalSchedule):
        verify_transform(yee_operator(cpml_axes=(0,1,2)),plan['spatial_basis'],plan['time_skew'],bad)
