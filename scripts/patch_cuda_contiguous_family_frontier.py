"""Trace-derived two-barrier four-bank owner mapping on selected52865.

Each1024-thread owner receives four contiguous Z planes. Ey/Hy travel through
warp-X shuffles or same-thread Z registers; only two chunk-boundary snapshots
per component remain shared. Ex/Ez/Hx/Hz retain permanent shared banks.
Z/Y private E imports follow the complete own-warp H loop; all cross-warp
consumers use immutable chunk snapshots. A conservative mode defers themPF1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES, once, transformed as permanent
except ModuleNotFoundError:
    from patch_cuda_permanent_family_frontier import PARENT_HASHES, once, transformed as permanent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(original, *, scope=7, defer_zy_e=False):
    full, proof = permanent(original, scope=scope)
    begin = full.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void PermanentFamilyBody")
    end = full.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", begin)
    body = full[begin:end]
    body = once(body, "Slots=PAD/T,ZStep=T/512;", "Slots=PAD/T,ZStep=1;")
    body = once(body, "frame+6*PAD);", "frame+4*PAD+2048);")
    body = once(body,
        " extern __shared__ float frame[];size_t*bases=",
        " extern __shared__ float frame[];\n float*ey_snapshot=frame+4*PAD;float*hy_snapshot=ey_snapshot+1024;\n size_t*bases=")
    body = once(body, "float*low_stage=frame+6*PAD+8*sizeof(size_t)/sizeof(float)+4;",
        "float*low_stage=frame+4*PAD+2048+8*sizeof(size_t)/sizeof(float)+4;")
    body = once(body, "zbase=threadIdx.x/512;", "zbase=(threadIdx.x/512)*4;")
    body = once(body,
        "for(int c=0;c<3;++c){h[c][j]=Initial(p,c,z,y,x);e[c][j]=Initial(p,c+3,z,y,x);frame[(c+3)*PAD+Slot(zs,sy,sx)]=e[c][j];}",
        "for(int c=0;c<3;++c){h[c][j]=Initial(p,c,z,y,x);e[c][j]=Initial(p,c+3,z,y,x);if(c!=1)frame[(c==0?2:3)*PAD+Slot(zs,sy,sx)]=e[c][j];}\n  if(j==0)ey_snapshot[(zbase/4)*512+sy*32+sx]=e[1][j];")
    body = once(body,
        "   frame[3*PAD+Slot(zs,sy,sx)]=v.ex;frame[4*PAD+Slot(zs,sy,sx)]=v.ey;frame[5*PAD+Slot(zs,sy,sx)]=v.ez;",
        "   frame[2*PAD+Slot(zs,sy,sx)]=v.ex;frame[3*PAD+Slot(zs,sy,sx)]=v.ez;\n   if(j==0)ey_snapshot[(zbase/4)*512+sy*32+sx]=v.ey;")
    body = once(body,
        "for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);\n    if(zs!=hz&&sy!=hy&&sx!=hx){",
        "for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);\n    const float ey_x=__shfl_sync(0xffffffffu,e[1][j],(sx+1)&31);\n    if(zs!=hz&&sy!=hy&&sx!=hx){")
    body = once(body,
        "if(!dz&&!ddy&&!ddx)return e[c][j];return frame[(c+3)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
        "if(!dz&&!ddy&&!ddx)return e[c][j];\n      if(c==1){if(ddx)return ey_x;if(dz){if(j+1<Slots)return e[1][j+1];return ey_snapshot[((zbase/4)^1)*512+sy*32+sx];}return e[1][j];}\n      return frame[(c==0?2:3)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];")
    body = once(body,
        "}h[c][j]=value;frame[c*PAD+Slot(zs,sy,sx)]=value;}",
        "}h[c][j]=value;if(c!=1)frame[(c==0?0:1)*PAD+Slot(zs,sy,sx)]=value;}")
    body = once(body,
        "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;\n   frame[Slot(zs,sy,sx)]=v.hx;frame[PAD+Slot(zs,sy,sx)]=v.hy;frame[2*PAD+Slot(zs,sy,sx)]=v.hz;",
        "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;\n   // Old private Ey is a warp-shuffle provider until PF1; defer all E ghosts.\n   frame[Slot(zs,sy,sx)]=v.hx;frame[PAD+Slot(zs,sy,sx)]=v.hz;")
    body = once(body,
        "  StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);",
        "  // Snapshot last-plane Hy unless its low-X value has a different\n  // coalesced writer. The skipped location is filled by that writer below.\n  {const int zs=zbase+Slots-1;if(!(sx==hx&&sy!=hy&&zs!=hz))hy_snapshot[(zbase/4)*512+sy*32+sx]=h[1][Slots-1];}\n  StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);")
    body = once(body,
        "for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,ys,hx)]=low_stage[c*LowXPoints+i];",
        "for(int c=0;c<3;++c)if(c!=1)frame[(c==0?0:1)*PAD+Slot(zs,ys,hx)]=low_stage[c*LowXPoints+i];\n   if((zs&3)==3)hy_snapshot[(zs/4)*512+ys*32+hx]=low_stage[LowXPoints+i];")
    body = once(body,
        "  RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);",
        "  // Point owners gather their coalesced low-X packet only after PF1.\n  if(sx==hx&&sy!=hy){\n   #pragma unroll\n   for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs!=hz){const int key=(((zs-hz)&7)-1)*BY+dy-1;\n    #pragma unroll\n    for(int c=0;c<3;++c){h[c][j]=low_stage[c*LowXPoints+key];e[c][j]=low_stage[(c+3)*LowXPoints+key];}\n   }}\n  }\n  // Z/Y ghosts retain old private Ey until all H shuffles have retired.\n  #pragma unroll\n  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs==hz||sy==hy){\n   const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,(zs-hz)&7,dy,dx);\n   e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;\n  }}")
    body = once(body,
        "for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);\n    if(zs!=ehz&&sy!=ehy&&sx!=ehx){",
        "for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);\n    const float hy_x=__shfl_sync(0xffffffffu,h[1][j],(sx+31)&31);\n    if(zs!=ehz&&sy!=ehy&&sx!=ehx){")
    body = once(body,
        "if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
        "if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];\n      if(field==1){if(ddx)return hy_x;if(dz){if(j)return h[1][j-1];return hy_snapshot[((zbase/4)^1)*512+sy*32+sx];}return h[1][j];}\n      return frame[(field==0?0:1)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];")
    body = once(body,
        "for(int c=0;c<3;++c)frame[(c+3)*PAD+Slot(zs,sy,sx)]=e[c][j];",
        "for(int c=0;c<3;++c)if(c!=1)frame[(c==0?2:3)*PAD+Slot(zs,sy,sx)]=e[c][j];\n   if(j==0)ey_snapshot[(zbase/4)*512+sy*32+sx]=e[1][j];")
    if not defer_zy_e:
        body = once(body,
            "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;\n   // Old private Ey is a warp-shuffle provider until PF1; defer all E ghosts.\n",
            "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;\n   // Own-warp H readers are retired; other warps use immutable snapshots.\n   e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;\n")
        body = once(body,
            "  // Z/Y ghosts retain old private Ey until all H shuffles have retired.\n  #pragma unroll\n  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs==hz||sy==hy){\n   const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,(zs-hz)&7,dy,dx);\n   e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;\n  }}", "")
    changed = full[:begin] + body + full[end:]
    changed = once(changed,
        "constexpr size_t PermanentFamilySharedBytes=6ULL*PAD*sizeof(float)+8*sizeof(size_t)+4*sizeof(int)+6ULL*LowXPoints*sizeof(float);\nstatic_assert(PermanentFamilySharedBytes==100904",
        "constexpr size_t PermanentFamilySharedBytes=(4ULL*PAD+2048)*sizeof(float)+8*sizeof(size_t)+4*sizeof(int)+6ULL*LowXPoints*sizeof(float);\nstatic_assert(PermanentFamilySharedBytes==76328")
    proof.update({"shared_bytes": 76328, "full_shared_field_banks": ["Hx", "Hz", "Ex", "Ez"],
                  "chunk_snapshots": {"Ey": "first Z plane/group", "Hy": "last Z plane/group"},
                  "snapshot_float_words": 2048, "private_field_floats_per_thread": 24,
                  "defer_private_zy_e_until_PF1": defer_zy_e,
                  "field_owner_mapping": "(tid//512)*4+j, (tid//32)%16, tid%32",
                  "new_obligations": ["all-lane Ey/Hy shuffles", "private Z/Y Ey imports after complete own-warp H loop; lowX afterPF1",
                                      "unique Hy snapshot writer for coalesced lowX points", "PF2 retires Hy snapshot readers"]})
    return changed, proof


def patch_copy(source, output, *, scope=7, defer_zy_e=False):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    for name, expected in PARENT_HASHES.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 parent required: {name}")
    changed, proof = transformed((source / "src/asymmetric_native.cuh").read_text(), scope=scope, defer_zy_e=defer_zy_e)
    proof.update({"patcher_sha256": sha(Path(__file__).read_bytes()),
                  "result_source_sha256": sha(changed.encode()), "gpu_executed": False})
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(changed)
    (output.parent / "contiguous-family-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--scope", type=int, default=7)
    p.add_argument("--defer-zy-e", action="store_true")
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, scope=a.scope, defer_zy_e=a.defer_zy_e)))


if __name__ == "__main__":
    main()
