"""Bounds, ownership and independent observation geometry for CUDA searches."""
import pytest
from scripts.cuda_trace_phase_cycle import PARENT,block,prove_schedule,stage
from scripts.cuda_trace_box_search import verify_mapping,stage as stage_box
from scripts.prove_cuda_shared_box import rebound


@pytest.mark.parametrize('cycle',[1,2,4,8])
def test_short_phase_cycles_cover_late_empty_and_short_windows(cycle):
    assert prove_schedule(cycle)['window_cases']==314


@pytest.mark.parametrize('cache,threads',[((8,8,32),512),((16,16,32),1024),((4,8,32),256)])
def test_cache_owner_bijection_and_coalesced_low_packet_coverage(cache,threads):
    result=verify_mapping(cache,threads)
    assert result['owner_checks']==16*cache[0]*cache[1]*cache[2]


def test_reject_thread_mapping_that_splits_an_xy_plane():
    with pytest.raises(ValueError,match='Whole XY planes'):
        verify_mapping((8,16,32),256)


def test_observation_sparse_geometry_does_not_follow_cache_shape(tmp_path):
    output=tmp_path/'candidate/source'
    stage_box(PARENT,output,(8,8,32),512,min_blocks=2)
    _,_,before=block((PARENT/'src/asymmetric_native.cuh').read_text(),'void Observe(')
    _,_,after=block((output/'src/asymmetric_native.cuh').read_text(),'void Observe(')
    assert before==after
    assert '(x&15)' in after and '(y&7)' in after


@pytest.mark.parametrize('mutation',['alias-H-E-banks','early-E-ghost-publication','missing-PF1','missing-PF2','single-shared-clock'])
def test_new_shared_owner_map_rejects_visibility_hazards(mutation):
    simulate,_=rebound()
    with pytest.raises(AssertionError):simulate(first=1,steps=2,seed=7,negative=mutation)


def test_new_shared_owner_map_retains_versions_across_ring_wrap():
    simulate,_=rebound()
    result=simulate(first=13,steps=8,seed=29)
    assert result['actual_field_and_clock_checks']>100000


def test_trace_selected_profile_staging_keeps_every_independent_term():
    from scripts.cuda_trace_profile_staging import helper
    from scripts.cuda_native_geometry_ir import derive_geometry
    for vector in (False,True):
        _,slots=helper(vector)
        for axis in range(3):
            expected={6*t['phase']+int(t['term'][-1]) for t in derive_geometry()['cpml_terms'] if t['axis_zyx']==axis}
            assert set(slots[axis])==expected and len(expected)==4


def test_fused_cohorts_keep_exactly_one_owner_and_disjoint_mutable_histories():
    from scripts.cuda_trace_fused_cohorts import prove
    result=prove()
    assert result['random_rank_partitions']==2000
    assert result['equal_rank_physical_trajectory_pairs_checked']>0


def test_native_shape_masks_cover_all_six_distinct_yee_supports():
    from scripts.cuda_trace_shape_edges import prove
    assert prove()['component_wall_and_stride_checks']==9216


def test_affine_coordinates_retain_repeated_cache_wraps():
    from scripts.cuda_trace_affine_coordinates import prove
    assert prove()['periodic_coordinate_checks']==7224


def test_software_sass_analysis_rejects_drops_and_unmapped_pcs(tmp_path):
    import json
    from scripts.analyze_cuda_sass_range import analyze
    sass=tmp_path/'kernel.sass';sass.write_text(' /*0010*/ @P0 FADD R0, R1, R2 ;\n')
    data=tmp_path/'metrics.json'
    base={'dropped_records':0,'records':1,'rows':[{'function':'kernel','pc':16,'metrics':{'smsp__sass_inst_executed':3,'smsp__sass_thread_inst_executed_pred_on':48}}]}
    data.write_text(json.dumps(base));result=analyze(data,sass)
    assert result['active_predicated_lane_fraction']==0.5
    assert result['dynamic_warp_instruction_categories'][0]['category']=='floating point'
    base['dropped_records']=1;data.write_text(json.dumps(base))
    with pytest.raises(ValueError,match='zero-drop'):analyze(data,sass)
    base['dropped_records']=0;base['rows'][0]['pc']=32;data.write_text(json.dumps(base))
    with pytest.raises(ValueError,match='absent'):analyze(data,sass)


def test_terminal_store_schedule_preserves_the_source_pipeline(tmp_path):
    from scripts.cuda_trace_grouped_publication import stage as grouped
    from scripts.cuda_trace_terminal_stores import stage as terminal
    from scripts.cuda_trace_phase_cycle import sha
    parent=tmp_path/'grouped/source';output=tmp_path/'terminal/source'
    grouped(PARENT,parent)
    before=(parent/'src/asymmetric_native.cuh').read_text()
    record=terminal(parent,output,sha(before.encode()))
    after=(output/'src/asymmetric_native.cuh').read_text()
    assert block(before,'void SourcePendingPairedBody(')[2]==block(after,'void SourcePendingPairedBody(')[2]
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        old=block(before,signature)[2];new=block(after,signature)[2]
        assert old.count('__syncthreads()')==new.count('__syncthreads()')
        assert 'if(n==Depth)' not in new
        assert 'if(window[1]<Depth)return;' in new
    assert record['terminal_window_checks']==292


def test_effect_grouping_preserves_memory_bits_for_both_guard_outcomes():
    from scripts.cuda_trace_effect_schedule import prove_native_publication
    result=prove_native_publication()
    assert result['guard_groups_before']==6 and result['guard_groups_after']==2
    assert result['finite_memory_bitpattern_checks']==23040
    assert result['component_support_masks_checked']==64
    assert result['inner_component_predicates_retained']


def test_effect_grouping_rejects_possible_trail_field_alias():
    from scripts.cuda_trace_effect_schedule import native_publication,group_guards
    effects=native_publication()
    alias={effects[2].buffer:'trails'}
    with pytest.raises(ValueError,match='overlapping stores'):group_guards(effects,aliases=alias)


def test_effect_grouping_rejects_new_value_dependency():
    from scripts.cuda_trace_effect_schedule import native_publication,group_guards
    effects=native_publication()
    with pytest.raises(ValueError,match='dependency'):group_guards(effects,dependencies=[(effects[2].name,effects[4].name)])


def test_effect_grouping_cannot_cross_barriers_or_ordered_observations():
    from dataclasses import replace
    from scripts.cuda_trace_effect_schedule import native_publication,group_guards
    effects=list(native_publication());effects[-1]=replace(effects[-1],region='after-next-barrier')
    with pytest.raises(ValueError,match='regions'):group_guards(effects)
    effects=list(native_publication());effects[0]=replace(effects[0],ordered=True)
    with pytest.raises(ValueError,match='source/DFT'):group_guards(effects)


def test_effect_grouping_rejects_invalid_alias_ranges():
    from dataclasses import replace
    from scripts.cuda_trace_effect_schedule import native_publication,group_guards
    effects=list(native_publication());effects[0]=replace(effects[0],word_interval=(735,0))
    with pytest.raises(ValueError,match='intervals'):group_guards(effects)
