"""Static code-size/control-flow review for generated CUDA lowerings.

Counts are static SASS sites, not executed instructions, cache misses, measured
DRAM, or runtime. Function selection must be explicit for multi-kernel probes.
SM70+ instruction bytes are16; source call sites are only a precompile warning.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BRANCH = frozenset({"BRA", "BRX", "BRXU", "JMP", "JMX", "CALL", "JCAL", "RET", "EXIT"})
CONTROL = BRANCH | {"BSSY", "BSYNC", "SSY", "SYNC", "BREAK", "CONT", "YIELD"}


def source_costs(source):
    return {"source_bytes": len(source.encode()),
            "field_body_copies": source.count("=env.Finish("),
            "scalar_external_read_sites": len(re.findall(r"env\.Read(?:Relative)?\(", source)),
            "packet_external_read_sites": len(re.findall(r"env\.ReadPacket(?:Relative)?\(", source)),
            "scalar_publication_sites": source.count("env.Publish("),
            "packet_publication_sites": source.count("env.PublishPacketRelative("),
            "cta_barrier_sites": source.count("__syncthreads()"),
            "full_warp_shuffle_sites": source.count("__shfl_sync(0xffffffffu"),
            "scope": "static emitted source sites; not dynamic operations or compiler SASS"}


def sass_costs(source, function_pattern):
    pattern = re.compile(function_pattern)
    functions = []
    current = None
    for line in source.splitlines():
        match = re.search(r"Function\s*:\s*(\S+)", line)
        if match:
            current = {"name": match[1], "instructions": []}
            functions.append(current)
            continue
        match = re.match(r"\s*/\*([0-9a-fA-F]{4,8})\*/\s+(.+?);", line)
        if match and current is not None:
            current["instructions"].append((int(match[1], 16), match[2].strip()))
    selected = []
    for function in functions:
        if not pattern.search(function["name"]):
            continue
        instructions = function["instructions"]
        counts = Counter()
        for _, instruction in instructions:
            pieces = instruction.split()
            opcode = pieces[1] if pieces[0].startswith("@") else pieces[0]
            counts[opcode.split(".")[0]] += 1
        selected.append({"name": function["name"], "static_instructions": len(instructions),
                         "instruction_footprint_bytes": max((a for a, _ in instructions), default=-16) + 16,
                         "branch_sites": sum(counts[c] for c in BRANCH),
                         "control_flow_sites": sum(counts[c] for c in CONTROL),
                         "global_load_sites": counts["LDG"], "global_store_sites": counts["STG"],
                         "local_load_sites": counts["LDL"], "local_store_sites": counts["STL"],
                         "shared_load_sites": counts["LDS"], "shared_store_sites": counts["STS"],
                         "shuffle_sites": counts["SHFL"], "opcodes": dict(sorted(counts.items()))})
    if not selected or any(f["static_instructions"] == 0 for f in selected):
        raise ValueError("SASS function selection is absent or empty; refusing a zero-cost result")
    return selected


def assess(functions, *, max_instructions=2048, max_branches=128):
    return {"scope": "static review thresholds; passing is not correctness/performance admission",
            "limits": {"static_instructions": max_instructions, "branch_sites": max_branches},
            "functions": [{"name": f["name"],
                           "instruction_budget": f["static_instructions"] <= max_instructions,
                           "branch_budget": f["branch_sites"] <= max_branches} for f in functions],
            "passes_static_review": all(f["static_instructions"] <= max_instructions and f["branch_sites"] <= max_branches for f in functions)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sass", type=Path, required=True)
    p.add_argument("--function-regex", required=True)
    p.add_argument("--max-instructions", type=int, default=2048)
    p.add_argument("--max-branches", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    functions = sass_costs(a.sass.read_text(), a.function_regex)
    result = {"functions": functions, "review": assess(functions, max_instructions=a.max_instructions, max_branches=a.max_branches)}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["review"], sort_keys=True))


if __name__ == "__main__":
    main()
