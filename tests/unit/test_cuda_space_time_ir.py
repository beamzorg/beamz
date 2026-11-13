"""Compiler legality/storage regressions, including actual reused-frame tokens."""

from dataclasses import replace

import pytest

from scripts.cuda_phase_schedule_ir import ScheduleHazard, compile_family_schedule
from scripts.cuda_space_time_ir import (
    IllegalSchedule,
    Operator,
    StreamShape,
    compile_stream,
    validate_operator,
    yee_operator,
)
from scripts.cuda_stencil_trace import Field, Update
from scripts.prove_cuda_circular_asymmetric import prove as prove_asymmetric
from scripts.prove_cuda_streamed_ir import prove


def changed(operator, name, expression):
    return Operator(tuple(Update(u.output, expression) if u.output.field == name else u
                          for u in operator.updates), operator.tensors)


@pytest.mark.parametrize("depth,persistent", ((1, 2), (2, 8), (4, 20), (8, 44)))
def test_frontier_is_derived_from_last_cross_plane_consumers(depth, persistent):
    plan = compile_stream(yee_operator(), StreamShape((64, 17 - depth, 33 - depth), depth, 512))
    assert plan["resources"]["persistent_field_scalars_per_thread"] == persistent
    assert ("Hz", depth) not in {(f["tensor"], f["version"]) for f in plan["frontier"]}
    assert plan["shared_components"]["H"] == [0, 2]
    assert plan["shared_components"]["E"] == ([] if depth == 1 else [0, 2])
    assert all(a["full_warp"] for a in plan["accesses"] if a["route"] == "warp-shuffle")


def test_stationary_coefficients_and_local_cpml_do_not_enter_field_frontier():
    plain = compile_stream(yee_operator())
    cpml = compile_stream(yee_operator(cpml_axes=(0, 1, 2), precision="bfloat16"))
    assert cpml["frontier"] == plain["frontier"]
    assert len(cpml["legality"]["local_state"]) == 12
    assert all(c["stationary"] for c in cpml["legality"]["coefficients"])
    assert any(c["scope"] == "profile" for c in cpml["legality"]["coefficients"])
    assert any(c["scope"] == "material" for c in cpml["legality"]["coefficients"])
    assert cpml["semantic_sha256"] != plain["semantic_sha256"]


def test_future_owner_and_phase_cycles_fail_closed():
    operator = yee_operator()
    with pytest.raises(IllegalSchedule, match="future owner"):
        compile_stream(changed(operator, "Hx", Field("Ex", 0)[(0, 0, 2)]))
    with pytest.raises(IllegalSchedule, match="producer"):
        compile_stream(changed(operator, "Hx", Field("Ex", 1)[(0, 0, 0)]))
    with pytest.raises(IllegalSchedule, match="previous Z frontier"):
        compile_stream(changed(operator, "Hx", Field("Ex", 0)[(-1, 0, 0)]))


def test_affine_legality_does_not_admit_an_unsupported_custom_lowering():
    operator = changed(yee_operator(), "Hx", Field("Hy", 0)[(0, 0, 0)])
    validate_operator(operator)
    with pytest.raises(IllegalSchedule, match="old own component"):
        compile_stream(operator)


def test_local_recurrence_spatial_read_is_rejected():
    operator = changed(yee_operator(cpml_axes=(0, 1, 2)), "psi_H0", Field("psi_H0", 0)[(1, 0, 0)])
    with pytest.raises(IllegalSchedule, match="same-point"):
        compile_stream(operator)


def test_thread_union_needs_no_computed_apron_but_reports_external_apron():
    shape = StreamShape()
    plan = compile_stream(yee_operator(), shape)
    assert plan["resources"]["exchange_shared_bytes"] == 8192
    providers = plan["provider_materialization"]
    assert providers["counts"]["extra_apron_scalar_points"] > 0
    for plane in ("first", "interior"):
        assert providers["counts"]["unique_external_scalar_fetches"][plane] < providers["counts"]["raw_external_read_edges"][plane]
    with pytest.raises(IllegalSchedule, match="physical X row"):
        compile_stream(yee_operator(), replace(shape, depth=5))


@pytest.mark.parametrize("depth,cpml", ((1, False), (2, True), (4, False)))
def test_two_actual_exchange_frames_preserve_versions_without_redundant_updates(depth, cpml):
    result = prove(shape=(3, 4, 5), tile=(2, 2, 3), depth=depth, cpml=cpml)
    assert result["executed_over_useful"] == 1.0
    assert result["shared_bytes"] <= 4 * 2 * 2 * 32 * (2 + depth - 1)


@pytest.mark.parametrize("option", ("current_e_frame", "stale_h_frame", "omit_barrier_a", "wrong_old_h"))
def test_bad_frame_or_register_lifetime_is_detected(option):
    with pytest.raises(AssertionError):
        prove(shape=(3, 4, 5), tile=(2, 2, 3), **{option: True})


def test_local_psi_store_must_follow_all_reads_of_the_old_value():
    with pytest.raises(AssertionError):
        prove(shape=(3, 4, 5), tile=(2, 2, 3), cpml=True, early_psi_store=True)


@pytest.mark.parametrize("counter", ("private", "two-phase"))
@pytest.mark.parametrize("async_imports", (False, True))
def test_family_import_motion_and_two_barriers_have_no_interwarp_hazards(counter, async_imports):
    result = compile_family_schedule(steps=3, counter=counter, async_imports=async_imports)
    assert result["proof"]["cta_barriers"] == 6
    # Whole-owned publication is also safe when separated by field family.
    compile_family_schedule(steps=3, counter=counter, publication="owned")


@pytest.mark.parametrize("options", ({"counter": "single"}, {"late_h_publication": True},
                                     {"early_e_import": True}, {"async_imports": True, "omit_wait": True}))
def test_unsynchronized_metadata_or_cache_reuse_is_rejected(options):
    with pytest.raises(ScheduleHazard):
        compile_family_schedule(**options)


def test_family_reordered_cache_tokens_cover_current_h_before_e_import():
    result = prove_asymmetric(steps=2, family_reordered=True, publish_all_owned=True)
    assert result["result"] == "pass"
    assert result["phase_metadata"]["cta_barriers"] == 4
