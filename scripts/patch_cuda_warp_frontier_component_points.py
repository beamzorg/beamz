"""Trace-schedule peer exchanges immediately before each component update.

All full-mask shuffles remain outside owned/support divergence. The canonical
read graph supplies the minimal peer set for each component; H0/H1/E0/E1 need
one XY peer, H2/E2 need two. Component FP order remains0,1,2. This reduces the
source-level peer frontier without claiming that ptxas honors that lifetime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES, once
from scripts.patch_cuda_warp_frontier_onehot import transformed as onehot
from scripts.prove_cuda_frontier_schedule import actual_reads
from scripts.cuda_space_time_ir import FIELDS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def closing_brace(text, opening):
    level, end = 1, opening+1
    while level:
        if text[end] == "{":
            level += 1
        elif text[end] == "}":
            level -= 1
        end += 1
    return end


def peers(phase, component):
    needed = set()
    for read in actual_reads()[phase*3+component]:
        if read.field not in FIELDS or read.field[0] == ("H" if phase == 0 else "E"):
            continue
        if read.offset[1]:
            needed.add(read.field.lower()+"_y")
        if read.offset[2]:
            needed.add(read.field.lower()+"_x")
    return needed


def lower_phase(body, phase, *, warp_fence=False):
    target = "h" if phase == 0 else "e"
    source = "e" if phase == 0 else "h"
    marker = "    // Every lane participates, including inactive physical boundary/ghost lanes.\n" if phase == 0 else "    const float hx_y=__shfl_sync"
    start = body.index(marker)
    pred = "    if(zs!=hz&&sy!=hy&&sx!=hx){" if phase == 0 else "    if(zs!=ehz&&sy!=ehy&&sx!=ehx){"
    pred_start = body.index(pred, start)
    pred_open = body.index("{", pred_start)
    pred_end = closing_brace(body, pred_open)
    shuffle_text = body[start:pred_start]
    shuffle_lines = [line.strip() for line in shuffle_text.splitlines() if "const float " in line]
    shuffle_map = {line.split("const float ", 1)[1].split("=", 1)[0]: line for line in shuffle_lines}
    if len(shuffle_map) != 4:
        raise ValueError("Expected four original unconditional peer shuffles")
    read_start = body.index("     const auto read=", pred_open)
    read_open = body.index("{", read_start)
    read_end = closing_brace(body, read_open)+1  # include semicolon
    read_lambda = body[read_start:read_end]
    result = []
    for c in range(3):
        needed = peers(phase, c)
        if needed - set(shuffle_map):
            raise ValueError("Canonical component requires an unmodeled peer")
        result.append("    {")
        result.append(f"     constexpr int c={c};")
        for name, line in shuffle_map.items():
            result.append("     "+(line if name in needed else f"constexpr float {name}=0.f;"))
        result.append(pred)
        result.append(read_lambda)
        result.append(f"     float value=0.f;if(env.Valid(c{'+3' if phase else ''},z,y,x))value=beamz::cuda::trace_native::AdvanceNativeCell<{phase},{c},false>(env,read,n,z,y,x);")
        result.append(f"     {target}[c][j]=value;"+("WarpFrontierStore(frame,c,zs,sy,sx,value);" if phase == 0 else ""))
        result.append("    }")
        if warp_fence and c != 2:
            result.append("     __syncwarp(0xffffffffu); // optional measured compiler/lifetime boundary")
        result.append("    }")
    return body[:start]+"\n".join(result)+body[pred_end:]


def literalize(body, phase):
    coord = "low+((zs-hz)&7)" if phase == 0 else "elow+((zs-ehz)&7)"
    marker = "   #pragma unroll\n   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+"+coord+";"
    if body.count(marker) != 1:
        raise ValueError("Expected one complete phase point loop")
    start = body.index(marker)
    opening = body.index("{", start)
    end = closing_brace(body, opening)
    contents = body[opening+1:end-1]
    replacement = "\n".join("   {constexpr int j="+str(j)+";"+contents+"   }" for j in range(8))
    return body[:start]+replacement+body[end:]


def transformed(original, *, warp_y=4, literal_points=True, warp_fence=False):
    full, header, proof = onehot(original, warp_y=warp_y)
    begin = full.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void PermanentFamilyBody")
    end = full.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", begin)
    body = full[begin:end]
    for phase in (0, 1):
        body = lower_phase(body, phase, warp_fence=warp_fence)
    if literal_points:
        for phase in (0, 1):
            body = literalize(body, phase)
    calls = 8 if literal_points else 1
    for phase in (0, 1):
        for c in range(3):
            if body.count(f"AdvanceNativeCell<{phase},{c},false>") != calls:
                raise ValueError("Component arithmetic replication does not match point schedule")
    changed = full[:begin]+body+full[end:]
    proof.update({"component_scheduled_fullmask_shuffles": True,
                  "literal_Z_points": 8 if literal_points else "pragma-unrolled8",
                  "component_peer_sets": {f"{phase}{c}": sorted(peers(phase, c)) for phase in (0, 1) for c in range(3)},
                  "peak_peer_floats_source_level": max(len(peers(p, c)) for p in (0, 1) for c in range(3)),
                  "ptxas_resource_fit_unproven": True,
                  "added_warp_barriers_per_point": 2 if warp_fence else 0,
                  "all_shuffles_fullmask_outside_owned_and_Valid": True,
                  "body_sha256": sha(body.encode())})
    return changed, header, proof


def patch_copy(source, output, *, warp_y=4, literal_points=True, warp_fence=False):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    for name, expected in PARENT_HASHES.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 parent required: {name}")
    code, header, proof = transformed((source / "src/asymmetric_native.cuh").read_text(),
                                     warp_y=warp_y, literal_points=literal_points, warp_fence=warp_fence)
    proof.update({"parent_hashes": PARENT_HASHES, "patcher_sha256": sha(Path(__file__).read_bytes()),
                  "result_source_sha256": sha(code.encode()), "gpu_executed": False})
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(code)
    (output / "src/warp_frontier_layout_generated.cuh").write_text(header)
    (output.parent / "warp-frontier-component-points-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--warp-y", type=int, choices=(2, 4, 8), default=4)
    p.add_argument("--rolled-points", action="store_true")
    p.add_argument("--warp-fence", action="store_true")
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, warp_y=a.warp_y,
                               literal_points=not a.rolled_points, warp_fence=a.warp_fence)))


if __name__ == "__main__":
    main()
