"""Trace-generated point boundaries for the register-owned warp frontier.

Each literal Z point reloads its phase's stable opaque clock and rematerializes
its physical coordinates. This deliberately limits cross-point CSE/liveness.
An optional empty inline-PTX read/write boundary pins the48 persistent field
values between points. Neither form promises a resource win: ptxas is the gate.
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
from scripts.patch_cuda_warp_frontier_512 import transformed as frontier


def sha(data):
    return hashlib.sha256(data).hexdigest()


def expand_point_loop(body, clock, *, fence=False):
    anchor = f"   #pragma unroll\n   for(int j=0;j<Slots;++j){{\n    const int step=*{clock}"
    if body.count(anchor) != 1:
        raise ValueError(f"Expected one rematerialized{clock} point loop")
    start = body.index(anchor)
    opening = body.index("{", start)
    level, closing = 1, opening+1
    while level:
        if body[closing] == "{":
            level += 1
        elif body[closing] == "}":
            level -= 1
        closing += 1
    contents = body[opening+1:closing-1]
    state_fence = ""
    if fence:
        operands = [f'"+f"({f}[{c}][{j}])' for f in ("h", "e") for c in range(3) for j in range(8)]
        state_fence = '\n    asm volatile("" : '+", ".join(operands)+");\n"
    literal = "\n".join("   {constexpr int j="+str(j)+";"+contents+state_fence+"   }" for j in range(8))
    return body[:start]+literal+body[closing:]


def transformed(original, *, warp_y=4, fence=False):
    full, header, proof = frontier(original, warp_y=warp_y)
    begin = full.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void PermanentFamilyBody")
    end = full.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", begin)
    body = full[begin:end]
    h_prefix = """  {const int step=*h_step,hz=(-step-1)&7,hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;
   const int dy=(sy-hy)&15,dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);"""
    h_new = """  {
   #pragma unroll
   for(int j=0;j<Slots;++j){
    const int step=*h_step,hz=(-step-1)&7,hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;
    const int dy=(sy-hy)&15,dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;
    const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);"""
    body = once(body, h_prefix, h_new)
    e_prefix = """  {const int step=*e_step,ehz=(-step-1)&7,ehy=(-step-1)&15,ehx=(-step-1)&31,elow=-step-1;
   const int y=oy+elow+((sy-ehy)&15),x=ox+elow+((sx-ehx)&31);
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);"""
    e_new = """  {
   #pragma unroll
   for(int j=0;j<Slots;++j){
    const int step=*e_step,ehz=(-step-1)&7,ehy=(-step-1)&15,ehx=(-step-1)&31,elow=-step-1;
    const int y=oy+elow+((sy-ehy)&15),x=ox+elow+((sx-ehx)&31);
    const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);"""
    body = once(body, e_prefix, e_new)
    for clock in ("h_step", "e_step"):
        body = expand_point_loop(body, clock, fence=fence)
    # Every point executes exactly the prior ordered scalar graph; the clock
    # value is stable until its reader-retirement barrier, independently of
    # the warp schedule and of this greater number of ordinary shared loads.
    if body.count("const int step=*h_step") != 8 or body.count("const int step=*e_step") != 8:
        raise ValueError("Expected eight opaque SSA intervals per phase")
    changed = full[:begin]+body+full[end:]
    proof.update({"point_phase_clock_rematerialization": True, "literal_Z_points": 8,
                  "phase_clock_loads_per_thread_step": 16,
                  "parent_phase_clock_loads_per_thread_step": 2,
                  "empty_state_readwrite_fence": fence,
                  "fence_is_resource_hypothesis_not_ptxas_result": True,
                  "clock_stability": "h_step cannotchangeuntilPF1 afterallHpoints; e_step cannotchangeuntilPF2 afterallEpoints",
                  "resource_gate": "zero stack/local/spills required before GPU", "body_sha256": sha(body.encode())})
    return changed, header, proof


def patch_copy(source, output, *, warp_y=4, fence=False):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    for name, expected in PARENT_HASHES.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 parent required: {name}")
    changed, header, proof = transformed((source / "src/asymmetric_native.cuh").read_text(), warp_y=warp_y, fence=fence)
    proof.update({"parent_hashes": PARENT_HASHES, "patcher_sha256": sha(Path(__file__).read_bytes()),
                  "result_source_sha256": sha(changed.encode()), "gpu_executed": False})
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(changed)
    (output / "src/warp_frontier_layout_generated.cuh").write_text(header)
    (output.parent / "warp-frontier-phase-points-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--warp-y", type=int, choices=(2, 4, 8), default=4)
    p.add_argument("--fence", action="store_true")
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, warp_y=a.warp_y, fence=a.fence)))


if __name__ == "__main__":
    main()
