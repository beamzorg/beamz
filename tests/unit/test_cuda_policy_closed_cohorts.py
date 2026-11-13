"""Typed cohort closure and physical-slot lifetime regressions."""
from scripts.cuda_policy_closed_cohorts import compile_cohorts,hazards,policy,schedule_first_owner_plan
import pytest


def plan(counts=(7,5,4)):
    nz,ny,nx=counts;ids=sorted(range(nz*ny*nx),key=lambda t:(t//(ny*nx)+(t//nx)%ny+t%nx,t));slots=[0]*len(ids);rank_sizes={}
    for t in ids:
        rank=t//(ny*nx)+(t//nx)%ny+t%nx;slots[t]=rank_sizes.get(rank,0);rank_sizes[rank]=slots[t]+1
    flags=[(0,2,1,16,17,3)[(t*1103515245+12345)%6] for t in range(len(ids))]
    return dict(counts=counts,tile_ids=ids,slots=slots,tile_flags=flags,max_frontier=max(rank_sizes.values()),depth=16)


def test_uniform_certificate_does_not_change_policy():
    for flags in range(64):assert policy(flags)==policy(flags|64)


def test_all_typed_hazards_close_inside_or_before_kernel():
    p=plan();start,finish,_=hazards(p);r=compile_cohorts(p,4,'earliest');mapping=r['cohort_of']
    assert sum(bool(v) for v in finish)>0
    for t in range(len(start)):
        for parent in start[t]|finish[t]:assert mapping[parent]<=mapping[t]
    assert sorted(t for c in r['cohorts'] for t in c['tiles'])==list(range(len(start)))
    for c in r['cohorts']:assert {policy(p['tile_flags'][t]) for t in c['tiles']}=={c['policy']}


def test_omitting_retirement_really_changes_legal_plan():
    p=plan();a,b,_=hazards(p);_,missing,_=hazards(p,omit_war=True)
    assert any(b[t]-a[t] for t in range(len(a)))
    assert all(not d for d in missing)


def test_homogeneous_region_is_one_kernel():
    p=plan();p['tile_flags']=[0]*len(p['tile_ids']);r=compile_cohorts(p,4)
    assert r['cohort_count']==1 and r['cohorts'][0]['initial_roots']==1


def test_schedule_first_allocation_preserves_physics_closures():
    p=plan();physical=compile_cohorts(p,4,'earliest',omit_war=True);allocated,result=schedule_first_owner_plan(p)
    assert result['cohort_of']==physical['cohort_of']
    start,finish,_=hazards(allocated)
    for t in range(len(start)):
        for parent in finish[t]:assert result['cohort_of'][parent]<=result['cohort_of'][t]
    assert len(allocated['owner_slots'])==len(p['tile_ids'])


def test_clipped_corner_startup_requires_diagonal_readiness():
    from scripts.prove_cuda_partial_plane_pipeline import simulate
    case=dict(minimum=(4,4,11),owned=(3,3,7),depth=4,workers=82,seed=13,
              closed_epochs=4,closed_allocation='no-reuse',queue_order='lifo',
              closed_flags='homogeneous',schedule='consumer-fast')
    # Two tied late starts publish skipped step1 before their diagonal parent
    # has produced Ex(4,4,5). Extra storage-retirement edges must not mask this.
    with pytest.raises(AssertionError,match='ghost version'):
        simulate(**case,ready_mode='nearest3')
    result=simulate(**case,ready_mode='startup7')
    assert result['field_reads']==25772 and result['partial_producer_reads']>0
