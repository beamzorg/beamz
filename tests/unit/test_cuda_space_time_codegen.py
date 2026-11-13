"""Validate actual generated control flow, callback coordinates and mutable SSA."""

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.cuda_codegen_audit import assess, sass_costs
from scripts.cuda_space_time_codegen import generate, mutable_groups
from scripts.cuda_space_time_ir import (
    IllegalSchedule,
    Operator,
    StreamShape,
    yee_operator,
)
from scripts.cuda_stencil_trace import Coefficient, Expr, Update
from scripts.cuda_trace_cpml_lower import lower_group

ROOT = Path(__file__).resolve().parents[2]


def test_codegen_deterministic_fixed_frontier_and_full_warp_inputs():
    first = generate()
    assert first == generate()
    code = first["source"]
    assert code.count("float history_") == 20
    assert "history_Hz_4" not in code
    assert "old_e_previous_" not in code
    assert "old_h_previous_" not in code
    assert code.count("__syncthreads()") == 8
    assert "__shfl_sync(0xffffffffu" in code
    assert all(group["cell_ssa_liveness"]["peak_live_scalars"] > 0 for group in first["groups"])


def test_relative_lowering_preserves_semantics_and_removes_absolute_read_reconstruction():
    direct = generate()
    relative = generate(relative_callbacks=True, materialized=True)
    assert direct["plan"]["semantic_sha256"] == relative["plan"]["semantic_sha256"]
    assert direct["plan"]["frontier"] == relative["plan"]["frontier"]
    assert "env.Read(" not in relative["source"]
    assert "provider_z=" not in relative["source"]
    assert "env.ReadPacketRelative(" in relative["source"]
    assert relative["source"].count("env.PublishPacketRelative(") == 4
    assert all(p["one_cell_lower_neighbor_admitted"] for p in relative["plan"]["provider_materialization"]["providers"])


@pytest.mark.parametrize("precision", ("float32", "bfloat16"))
def test_native_term_lowering_guards_inactive_profiles_and_keeps_unrounded_correction(precision):
    operator = yee_operator(cpml_axes=(0, 1, 2), precision=precision)
    for phase in (0, 1):
        for field, group in mutable_groups(operator, phase):
            terms, value = lower_group(field, group, operator.specs)
            assert len(terms) == 2
            assert all(t.state is not None and t.precision == precision for t in terms)
            assert value.op == "fma_rn"
    code = generate(operator, native_boundaries=True)["source"]
    assert code.count("if(env.LocalActive(") == 4 * 12
    assert "=__fmul_rn(-1.0f,derivative_1);" in code
    assert "__fmaf_rn(derivative_1," in code
    assert ("__float2bfloat16_rn(next_psi)" in code) == (precision == "bfloat16")


def test_term_lowering_rejects_shifted_profile_or_rounded_field_correction():
    operator = yee_operator(cpml_axes=(0, 1, 2), precision="bfloat16")
    field, group = mutable_groups(operator, 0)[0]
    primary = next(u for u in group if u.output.field == field)
    first, second = primary.value.args[1].args
    derivative, kappa, recurrence = first.args[1].args
    wrong = Expr("mul_rn", (first.args[0], Expr("fma_rn", (derivative, kappa, Expr("round_bf16", (recurrence,))))))
    root = Expr("fma_rn", (primary.value.args[0], Expr("add", (wrong, second)), primary.value.args[2]))
    bad = tuple(Update(u.output, root) if u.output.field == field else u for u in group)
    with pytest.raises(IllegalSchedule, match="recurrence"):
        lower_group(field, bad, operator.specs)
    a, old, product = recurrence.args
    # Shift the inv-kappa profile while keeping all other graph nodes intact.
    profile = kappa.args[0]
    axis = profile.axes[0]
    offset = tuple(int(i == axis) for i in range(3))
    shifted = Coefficient(profile.name, profile.axes)[offset]
    wrong = Expr("mul_rn", (first.args[0], Expr("fma_rn", (derivative, shifted, recurrence))))
    root = Expr("fma_rn", (primary.value.args[0], Expr("add", (wrong, second)), primary.value.args[2]))
    bad = tuple(Update(u.output, root) if u.output.field == field else u for u in group)
    with pytest.raises(IllegalSchedule, match="unshifted"):
        lower_group(field, bad, operator.specs)


def test_unsupported_precision_and_cross_group_mutability_fail_closed():
    operator = yee_operator()
    bad = Operator(operator.updates, (replace(operator.tensors[0], precision="bfloat16"), *operator.tensors[1:]))
    with pytest.raises(IllegalSchedule, match="Field storage"):
        generate(bad)
    with pytest.raises(IllegalSchedule):
        generate(shape=StreamShape((64, 14, 29), 4, 512))


@pytest.mark.parametrize("options", (
    ("--materialize-providers", "--relative-callbacks", "--native-boundaries"),
    ("--materialize-providers", "--relative-callbacks", "--native-boundaries", "--cpml", "--bf16", "--pec-mask", "63"),
    ("--materialize-providers", "--relative-callbacks", "--native-boundaries", "--cpml", "--bf16", "--inactive-cpml", "--zero-fields"),
    ("--loop-stages", "--relative-callbacks", "--native-boundaries", "--cpml", "--bf16", "--pec-mask", "63", "--depth", "4"),
))
def test_actual_generated_body_matches_independent_cpu_simt_oracle(tmp_path, options):
    if shutil.which("c++") is None:
        pytest.skip("C++20 compiler needed for the CPU SIMT oracle")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_cuda_streamed_host.py"),
                             "--output-dir", str(tmp_path), *options],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"errors":0' in result.stdout


def test_compact_stage_loop_keeps_named_queues_and_two_static_phase_bodies():
    unrolled = generate(relative_callbacks=True)
    compact = generate(relative_callbacks=True, loop_stages=True)
    assert compact["plan"]["semantic_sha256"] == unrolled["plan"]["semantic_sha256"]
    assert compact["stage_control_lowering"]["persistent_scalars_per_thread"] == 20
    assert compact["source"].count("float queue_") == 20
    assert "#pragma unroll 1" in compact["source"]
    assert "queue_Hx[" not in compact["source"]
    assert compact["static_source_costs"]["field_body_copies"] == 6
    assert unrolled["static_source_costs"]["field_body_copies"] == 24
    assert compact["static_source_costs"]["packet_publication_sites"] == 1
    assert compact["source"].count("__syncthreads()") == 2
    with pytest.raises(IllegalSchedule):
        generate(loop_stages=True, materialized=True)


def test_sass_budget_counts_static_sites_and_fails_closed_on_empty_selection():
    text = """Function : test_kernel
 /*0000*/ MOV R1,R0;
 /*0010*/ @P0 BRA 0x40;
 /*0020*/ SHFL.IDX R0,R1,0,0;
 /*0030*/ EXIT;
Function : other
 /*0000*/ NOP;
"""
    selected = sass_costs(text, "test_kernel")
    assert selected[0]["static_instructions"] == 4
    assert selected[0]["branch_sites"] == 2
    assert selected[0]["instruction_footprint_bytes"] == 64
    assert not assess(selected, max_instructions=3)["passes_static_review"]
    with pytest.raises(ValueError):
        sass_costs(text, "missing")
