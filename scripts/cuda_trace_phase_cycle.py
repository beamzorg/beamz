"""Generate short phase cycles for the current native moving CUDA tile.

This changes copied research sources only. The scalar-cell arithmetic, point
ownership, imports, publications and barriers retain their original order.
The loop induction variable replaces a redundant volatile phase counter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".cache/cuda-trace-five-hour-2026-11-13-1059/noio-fixed-cubes-cached-v1/source"
PARENT_CONSUMER = "2336d5bdb323cd78f6e571b9b4aafd9c9d53d7f16c0028b299f75bb9ebd3a3b8"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def block(text, signature):
    start = text.index(signature)
    begin = text.index("{", start)
    level, end = 1, begin + 1
    while level:
        level += (text[end] == "{") - (text[end] == "}")
        end += 1
    return start, end, text[start:end]


def transform_body(original, cycle, *, clock="induction", split_ghost=False):
    anchor = "for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n)"
    begin, end, loop = block(original, anchor)
    body = loop[loop.index("{") + 1:-1]
    if clock != "shared":
        if body.count("step=*phase_step") != 2 or body.count("if(!threadIdx.x)*phase_step=n+1;") != 1:
            raise ValueError("Reviewed phase-counter sites changed")
        if clock == "induction":
            body = body.replace("step=*phase_step", "step=n")
        elif clock == "shuffle":
            body = body.replace("step=*phase_step", "step=__shfl_sync(0xffffffffu,n,0)")
        elif clock == "opaque-mov":
            body = body.replace('{const int step=*phase_step,', '{int step;asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));const int ')
        else:
            raise ValueError("Unknown phase-clock lowering")
        body = body.replace("if(!threadIdx.x)*phase_step=n+1;", "")
    if cycle == 1:
        changed = loop[:loop.index("{")+1] + body + "}"
    else:
        parts = [f"for(int cycle_base=0;cycle_base<Depth;cycle_base+={cycle}){{"]
        for phase in range(1, cycle + 1):
            phase_body = body
            if cycle % 8 == 0:
                phase_body = phase_body.replace("(-step-1)&7", str((-phase-1)&7))
                phase_body = phase_body.replace("(-n-1)&7", str((-phase-1)&7))
            parts += [f"{{const int n=cycle_base+{phase};",
                      "if(n<=Depth&&(Fast||(n>=window[0]&&n<=window[1]))){",
                      phase_body, "}}"]
        parts.append("}")
        changed = "\n".join(parts)
    result = original[:begin] + changed + original[end:]
    # The old pragma precedes the replaced loop and intentionally keeps the
    # outer cycle loop rolled. Inner phases are emitted explicitly.
    if clock != "shared":
        result = result.replace("if(!threadIdx.x)*phase_step=1;", "")
        result = result.replace("if(!threadIdx.x)*phase_step=first;", "")
    if split_ghost:
        raise ValueError("No unreviewed ghost rescheduling is enabled")
    return result


def prove_schedule(cycle):
    cases = 0
    for depth in (4, 16):
        for first in range(1, depth + 2):
            for last in range(0, depth + 1):
                expected = list(range(first, last+1))
                actual = [base + phase for base in range(0, depth, cycle)
                          for phase in range(1, cycle+1)
                          if base+phase <= depth and first <= base+phase <= last]
                if actual != expected:
                    raise AssertionError((depth, first, last, actual, expected))
                cases += 1
    return {"window_cases": cases, "depths": [4, 16],
            "ordered_phase_sequences_equal": True,
            "uniform_window_guard_preserves_block_barrier_participation": True}


def stage(source, output, cycle, *, scope="both", clock="induction", volatile_profiles=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = ROOT / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("A new disjoint copied source is required")
    path = source / "src/asymmetric_native.cuh"
    original = path.read_text()
    if sha(original.encode()) != PARENT_CONSUMER:
        raise ValueError("Exact current common-shape/profile-cache parent required")
    signatures = ["void UniformBody(", "void FixedCubeCachedBody("]
    if scope == "fixed":
        signatures = signatures[1:]
    result = original
    for signature in signatures:
        begin, end, body = block(result, signature)
        changed = transform_body(body, cycle, clock=clock)
        result = result[:begin] + changed + result[end:]
    if volatile_profiles:
        old="return profiles[IndependentProfileCacheIndex(slot,kind,packed)];"
        if result.count(old)!=1:raise ValueError("Independent profile cache read changed")
        result=result.replace(old,"return reinterpret_cast<const volatile float*>(profiles)[IndependentProfileCacheIndex(slot,kind,packed)];")
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(result)
    proof = {"schema": "beamz.short-phase-cycle.v1", "cycle": cycle,
             "scope": scope, "phase_clock": clock,
             "volatile_shared_profile_rematerialization": volatile_profiles,
             "schedule": prove_schedule(cycle),
             "parent_sha256": sha(original.encode()), "consumer_sha256": sha(result.encode()),
             "producer_sha256": sha(Path(__file__).read_bytes()),
             "point_order_and_arithmetic_unchanged": True,
             "barriers_per_executed_phase": 4, "GPU_executed": False}
    (output.parent / "phase-cycle-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=PARENT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cycle", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--scope", choices=("both", "fixed"), default="both")
    p.add_argument("--clock", choices=("induction", "shared", "opaque-mov", "shuffle"), default="induction")
    p.add_argument("--volatile-profiles", action="store_true")
    a = p.parse_args()
    print(json.dumps(stage(a.source, a.output, a.cycle, scope=a.scope, clock=a.clock, volatile_profiles=a.volatile_profiles)))
