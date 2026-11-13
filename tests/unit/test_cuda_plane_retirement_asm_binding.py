"""PTX selector source/bit semantics, independent of compiler resource success."""

import pytest

from scripts.cuda_plane_retirement_asm_binding import (
    bit_certificate,
    inspect_blocks,
    operand_snapshot_certificate,
)


def blocks():
    one = '''   asm volatile("{ .reg .pred selected; .reg .f32 ax,ay,az; "
    "mov.f32 ax,%3; mov.f32 ay,%4; mov.f32 az,%5; "
    "setp.eq.s32 selected,%6,K; "
    "selp.f32 %0,ax,%0,selected; selp.f32 %1,ay,%1,selected; selp.f32 %2,az,%2,selected; }"
    : "+f"(hp[0]),"+f"(hp[1]),"+f"(hp[2])
    : "f"(h[0][K]),"f"(h[1][K]),"f"(h[2][K]),"r"(j));'''
    return "\n".join(one.replace("K", str(k)) for k in range(16))


def test_literal_predicate_and_all_three_component_indices_match():
    _, constants = inspect_blocks(blocks())
    assert constants == list(range(16))


@pytest.mark.parametrize("mutation", ["wrong_component_slot", "missing_snapshot", "write_only_output", "missing_block"])
def test_changed_selector_instruction_or_constraint_is_rejected(mutation):
    source = blocks()
    if mutation == "wrong_component_slot":
        source = source.replace('"f"(h[1][0])', '"f"(h[1][1])', 1)
    elif mutation == "missing_snapshot":
        source = source.replace("mov.f32 ay,%4; ", "", 1)
    elif mutation == "write_only_output":
        source = source.replace('"+f"(hp[0])', '"=f"(hp[0])', 1)
    else:
        source = source[:source.rindex("   asm volatile")]
    with pytest.raises(ValueError):
        inspect_blocks(source)


def test_selection_preserves_special_fp32_bits_without_arithmetic():
    result = bit_certificate()
    assert "80000000" in result["fp32_bit_patterns"]
    assert "7f800001" in result["fp32_bit_patterns"]
    with pytest.raises(ValueError, match="changed FP32 source bits"):
        bit_certificate(wrong_component=True)


def test_all_input_snapshots_precede_outputs_under_adversarial_coalescing():
    assert operand_snapshot_certificate()["result"] == "pass"
    with pytest.raises(ValueError, match="clobbered by an earlier output"):
        operand_snapshot_certificate(omit_snapshot=True)
