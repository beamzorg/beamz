"""Exact-source and bit-selection admission for literal PTX H-plane selection."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_plane_retirement_binding import digest
from scripts.cuda_plane_retirement_rolled_binding import admit as admit_rolled

SELECTOR = """   #pragma unroll
   for(int k=0;k<Slots;++k)if(j==k){hp[0]=h[0][k];hp[1]=h[1][k];hp[2]=h[2][k];}"""
ASM_SHA = "e41d80e721987e5e28fbc5586ad721b60a85a70ee0a926bd5943f75e5bbadb13"


def inspect_blocks(source):
    blocks = list(re.finditer(r"   asm volatile\(.*?\);", source, re.DOTALL))
    if len(blocks) != 16:
        raise ValueError("Expected exactly sixteen literal asm selector blocks")
    constants = []
    for index, match in enumerate(blocks):
        parts = match.group().split("\n    :")
        if len(parts) != 3:
            raise ValueError("Changed inline-asm operand constraints")
        text = "".join(re.findall(r'"([^"\\]*)"', parts[0])).strip()
        if not (text.startswith("{") and text.endswith("}")):
            raise ValueError("PTX temporaries need per-block local scope")
        instructions = [" ".join(i.split()) for i in text[1:-1].split(";") if i.strip()]
        expected = [".reg .pred selected", ".reg .f32 ax,ay,az",
                    "mov.f32 ax,%3", "mov.f32 ay,%4", "mov.f32 az,%5",
                    f"setp.eq.s32 selected,%6,{index}",
                    "selp.f32 %0,ax,%0,selected", "selp.f32 %1,ay,%1,selected",
                    "selp.f32 %2,az,%2,selected"]
        if instructions != expected:
            raise ValueError("PTX snapshot/predicate/component instruction order changed")
        outputs = re.sub(r"\s+", "", parts[1])
        inputs = re.sub(r"\s+", "", parts[2])
        if outputs != '"+f"(hp[0]),"+f"(hp[1]),"+f"(hp[2])':
            raise ValueError("Selector must read and write all three old hp values")
        expected_inputs = f'"f"(h[0][{index}]),"f"(h[1][{index}]),"f"(h[2][{index}]),"r"(j));'
        if inputs != expected_inputs:
            raise ValueError("Predicate or literal same-component H operands changed")
        constants.append(index)
    if any(source[a.end():b.start()].strip() for a, b in zip(blocks, blocks[1:], strict=False)):
        raise ValueError("Unexpected operations between selector blocks")
    return blocks, constants


def bit_certificate(*, wrong_component=False):
    # Includes both zero signs, subnormals, infinities and distinct q/sNaN bits.
    values = (0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
              0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
              0x7fc00001, 0x7fc12345, 0x7f800001, 0xffc54321, 0x7f7fffff, 0xff7fffff)
    checked = 0
    for rotation in range(16):
        h = tuple(tuple(values[(c * 5 + k + rotation) % 16] for k in range(16)) for c in range(3))
        for j in range(16):
            hp = [0, 0, 0]
            for k in range(16):
                snapshot = tuple(h[(c + int(wrong_component)) % 3][k] for c in range(3))
                if j == k:
                    hp[:] = snapshot
            if hp != [h[c][j] for c in range(3)]:
                raise ValueError("PTX selector changed FP32 source bits")
            checked += 3
    return {"fp32_bit_patterns": [f"{value:08x}" for value in values],
            "selected_component_bits_checked": checked, "result": "pass"}


def operand_snapshot_certificate(*, omit_snapshot=False):
    # Adversarial allowed operand coalescing: old hp0 and candidate Hy share A.
    # MOV all candidates first prevents hp0's first SELP from clobbering Hy.
    registers = {"A": 0, "B": 0, "C": 0, "D": 0x3f800000, "E": 0x80000000}
    sources, outputs = ("D", "A", "E"), ("A", "B", "C")
    snapshot = [registers[name] for name in sources]
    for c in range(3):
        registers[outputs[c]] = registers[sources[c]] if omit_snapshot else snapshot[c]
    if [registers[name] for name in outputs] != snapshot:
        raise ValueError("Uncaptured asm input was clobbered by an earlier output")
    return {"adversarial_operand_coalescing": "old hp0 aliases candidate Hy", "result": "pass"}


def admit(source, cell, proof):
    blocks, constants = inspect_blocks(source)
    reconstructed = source[:blocks[0].start()] + SELECTOR + source[blocks[-1].end():]
    reconstructed = reconstructed.replace("AsymmetricRetireLoopAsmE", "AsymmetricRetireLoopE")
    parent = admit_rolled(reconstructed, cell, proof)
    if digest(source) != ASM_SHA:
        raise ValueError("Asm consumer differs from independently reviewed source")
    return {"schema": "beamz.cuda-plane-retirement-asm-admission.v1",
            "admitted_source": True, "consumer_sha256": digest(source),
            "source_binding_sha256": digest(Path(__file__).read_bytes()),
            "literal_selector_constants": constants, "parent_source_admission": parent,
            "bit_selection": bit_certificate(), "operand_snapshot": operand_snapshot_certificate(),
            "ptx_isa_reference": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp",
            "barriers_per_step": 18,
            "scope": "Exact-source selector/lifetime derivation only; resource, arithmetic and sanitizer gates required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.consumer, args.cell, args.proof)}:
        parser.error("Admission output must differ from source/proof inputs")
    result = admit(args.consumer.read_bytes().decode(), args.cell.read_bytes().decode(), json.loads(args.proof.read_text()))
    result["proof_file_sha256"] = digest(args.proof.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "parent_source_admission"}))


if __name__ == "__main__":
    main()
