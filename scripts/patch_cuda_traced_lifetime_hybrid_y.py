"""512-thread H-private/E-shared frontier with exact compact H Y exchange.

All eight Z points of an X/Y row belong to one thread. E-phase -Z H reads are
private, -X Hy/Hz reads are unconditional full-warp exchanges, and only -Y
Hx/Hz reads use shared memory. The latter footprint includes current low-Y
ghosts and excludes high-Y owned rows: dz1..7,dy0..14,dx1..31.
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
from scripts.patch_cuda_traced_lifetime_private_psi import HOST_PARENT
from scripts.patch_cuda_traced_lifetime_shared_fields import transform as all_shared

HELPERS = r"""
constexpr int HybridYVolume=BZ*BY*BX;
constexpr size_t HybridYSharedBytes=(3ULL*PAD+2ULL*HybridYVolume)*sizeof(float)+8*sizeof(size_t)+4*sizeof(int)+6ULL*LowXPoints*sizeof(float);
static_assert(HybridYVolume==3255&&HybridYSharedBytes==77792,"Exact compactY footprint/budget");
__device__ __forceinline__ int HybridYSlot(int dz,int dy,int dx){return ((dz-1)*BY+dy)*BX+dx-1;}
__device__ __forceinline__ float HybridWarpX32(float value,int lane){
 uint32_t result;
 asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
              : "=r"(result) : "r"(__float_as_uint(value)), "r"(lane));
 return __uint_as_float(result);
}
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def generated_body(consumer):
    parent, _ = all_shared(consumer, scope=2)
    begin = parent.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void PermanentFamilyBody")
    end = parent.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", begin)
    body = parent[begin:end]
    body = once(body, "void PermanentFamilyBody(", "void HybridYBody(")
    body = once(body,
        "constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;",
        "constexpr int T=512,Slots=8,ZStep=1;")
    body = once(body,
        'static_assert(T==1024,"Permanent-family frontier requires admitted1024-thread owner mapping");',
        'static_assert(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE,"Only certified NoIO hybridY");')
    body = once(body, "extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+6*PAD);",
                "extern __shared__ float frame[];float*h_y=frame+3*PAD;size_t*bases=reinterpret_cast<size_t*>(h_y+2*HybridYVolume);")
    body = once(body, "float*low_stage=frame+6*PAD+8*sizeof(size_t)/sizeof(float)+4;",
                "float*low_stage=h_y+2*HybridYVolume+8*sizeof(size_t)/sizeof(float)+4;")
    body = once(body, " const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/512;",
                " const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=0;\n float h[3][Slots];")
    body = once(body,
        "for(int c=0;c<6;++c)frame[c*PAD+Slot(zs,sy,sx)]=Initial(p,c,z,y,x);",
        "for(int c=0;c<3;++c){h[c][j]=Initial(p,c,z,y,x);frame[c*PAD+Slot(zs,sy,sx)]=Initial(p,c+3,z,y,x);}")
    body = once(body,
        "   frame[Slot(zs,sy,sx)]=v.hx;frame[PAD+Slot(zs,sy,sx)]=v.hy;frame[2*PAD+Slot(zs,sy,sx)]=v.hz;\n   frame[3*PAD+Slot(zs,sy,sx)]=v.ex;",
        "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;\n   frame[3*PAD+Slot(zs,sy,sx)]=v.ex;")
    # Point updates remain the actual canonical calls; only their field routing changes.
    start = body.index("  {const int step=*h_step,")
    finish = body.index("  const int hz=(-n-1)", start)
    region = body[start:finish]
    region = once(region, "return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
                  "if(field<3)return h[field][j];return frame[(field-3)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];")
    region = once(region, "}frame[c*PAD+Slot(zs,sy,sx)]=value;}",
                  "}h[c][j]=value;if((c==0||c==2)&&dy<BY)h_y[(c==0?0:HybridYVolume)+HybridYSlot((zs-hz)&7,dy,dx)]=value;}")
    body = body[:start] + region + body[finish:]
    body = once(body,
        "   frame[Slot(zs,sy,sx)]=v.hx;frame[PAD+Slot(zs,sy,sx)]=v.hy;frame[2*PAD+Slot(zs,sy,sx)]=v.hz;",
        "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;\n"
        "   if(sy==hy&&zs!=hz&&sx!=hx){const int key=HybridYSlot(dz,0,dx);h_y[key]=v.hx;h_y[HybridYVolume+key]=v.hz;}")
    body = once(body, """  // Coalesced H imports precede PF1. E ghosts remain untouched until PF1
  // retires every old-E reader; six-field low-X staging is read only after it.
  for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,ddy=1+i%BY;
   const int zs=(hz+dz)&7,ys=(hy+ddy)&15;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,ys,hx)]=low_stage[c*LowXPoints+i];
  }
""", "  // Only low-Y H values communicate across warps. Low-X H stays staged until PF1.\n")
    body = once(body, "  // PF1 retired all old-E neighbors. Z/Y and low-X E writers are disjoint.",
        """  // Low-X H private owners can now consume the coalesced import stage.
  if(sx==hx&&sy!=hy){
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=j;if(zs!=hz){const int key=(((zs-hz)&7)-1)*BY+dy-1;
    #pragma unroll
    for(int c=0;c<3;++c)h[c][j]=low_stage[c*LowXPoints+key];
   }}
  }
  // PF1 retired all old-E neighbors. Z/Y and low-X E writers are disjoint.""")
    start = body.index("  {const int step=*e_step,")
    finish = body.index("  const int y=oy+low+dy", start)
    region = body[start:finish]
    region = once(region,
        "   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);\n    if(zs!=ehz&&sy!=ehy&&sx!=ehx){",
        "   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);\n"
        "    const float hy_x=HybridWarpX32(h[1][j],(sx+31)&31),hz_x=HybridWarpX32(h[2][j],(sx+31)&31);\n"
        "    if(zs!=ehz&&sy!=ehy&&sx!=ehx){")
    region = once(region, "return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
        """if(field>=3)return frame[(field-3)*PAD+Slot(zs,sy,sx)];
      if(!dz&&!ddy&&!ddx)return h[field][j];if(dz)return h[field][(j+7)&7];
      if(ddx)return field==1?hy_x:hz_x;
      return h_y[(field==0?0:HybridYVolume)+HybridYSlot((zs-ehz)&7,((sy-ehy)&15)-1,(sx-ehx)&31)];""")
    body = body[:start] + region + body[finish:]
    body = once(body,
        "const float hv=frame[c*PAD+Slot(zs,sy,sx)],ev=frame[(c+3)*PAD+Slot(zs,sy,sx)];",
        "const float hv=h[c][j],ev=frame[(c+3)*PAD+Slot(zs,sy,sx)];")
    # Every remaining bank+3 access denotes E and becomes its permanent bank0.
    for old, new in (("frame[3*PAD+Slot", "frame[Slot"),
                     ("frame[4*PAD+Slot", "frame[PAD+Slot"),
                     ("frame[5*PAD+Slot", "frame[2*PAD+Slot"),
                     ("frame[(c+3)*PAD+Slot", "frame[c*PAD+Slot")):
        body = body.replace(old, new)
    if "frame[field*PAD" in body or "frame[3*PAD" in body or "frame[(c+3)*PAD" in body:
        raise ValueError("Unlowered family bank alias remains")
    return body


def transform(consumer, host, header, *, enabled=True):
    if sha(consumer.encode()) != PARENT_HASHES["asymmetric_native.cuh"] or sha(host.encode()) != HOST_PARENT or sha(header.encode()) != PARENT_HASHES["native_moving.h"]:
        raise ValueError("Exact selected52865 consumer/host/workspace required")
    body = generated_body(consumer)
    anchor = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody"
    text = once(consumer, anchor, HELPERS + body + anchor)
    entry = """template<int Depth>
__global__ __launch_bounds__(512,1) void AsymmetricNoIOHybridY512(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14){
  case 2:HybridYBody<Depth,1,false,true,1,true,1>(p,tiles);return;
  case 4:HybridYBody<Depth,1,false,true,1,true,2>(p,tiles);return;
  case 8:HybridYBody<Depth,1,false,true,1,true,4>(p,tiles);return;
  default:HybridYBody<Depth,1,false,true,1,true>(p,tiles);return;
 }
}

"""
    text = once(text, "template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024",
                entry + "template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024")
    text = once(text, "  const auto kernel=AsymmetricNoIOUniform1024<Depth>;",
        """  if(BeamzResearchHybridYEnabled){
   const auto hybrid=AsymmetricNoIOHybridY512<Depth>;
   status=cudaFuncSetAttribute(hybrid,cudaFuncAttributeMaxDynamicSharedMemorySize,HybridYSharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricNoIOHybridY512<Depth><<<uniform_count,512,HybridYSharedBytes,stream>>>(p,ids);return cudaGetLastError();
  }
  const auto kernel=AsymmetricNoIOUniform1024<Depth>;""")
    header = once(header, "struct BeamzNativeMovingWorkspace {",
                  'extern const bool BeamzResearchHybridYEnabled __attribute__((visibility("hidden")));\n\nstruct BeamzNativeMovingWorkspace {')
    host = once(host, "const bool BeamzResearchSourcePendingEnabled=true;",
                f"const bool BeamzResearchSourcePendingEnabled=true;\nconst bool BeamzResearchHybridYEnabled={'true' if enabled else 'false'};")
    host = once(host, "    const bool source_selector=",
                "    const bool hybrid_y_selector=BeamzResearchHybridYEnabled;append_bytes(&hybrid_y_selector,sizeof(bool));\n    const bool source_selector=")
    host = once(host, '    result["moving_source_presence_mask"]=',
                '    result["moving_hybrid_y_selector_enabled"]=BeamzResearchHybridYEnabled;result["moving_hybrid_y_threads"]=512;result["moving_hybrid_y_shared_bytes"]=77792;\n    result["moving_source_presence_mask"]=')
    host = once(host, "NB_MODULE(_cuda_native, module) {",
                'NB_MODULE(_cuda_native, module) {\n  module.attr("moving_hybrid_y")=true;module.attr("moving_hybrid_y_threads")=512;module.attr("moving_hybrid_y_shared_bytes")=77792;')
    return text, host, header, {"threads": 512, "zslots_per_thread": 8, "private_H_floats_per_thread": 24,
        "private_E_floats_per_thread": 0, "compact_H_Y_words": 6510, "shared_bytes": 77792,
        "E_Y_neighbor_source_footprint": "dz1..7 dy0..14 dx1..31 forHxHz; lowYincluded/highYexcluded",
        "E_Z_neighbor": "same-threadh[field][(j+7)&7]", "E_X_neighbor": "unconditionalfullwarpHy/Hzshuffle",
        "loop_barriers": 2, "all_publications_prePF2": True,
        "uncached_original_NoIO_entry_1024_and_other_classes_retained": True,
        "selector_bound_to_graph_identity": True, "selector_enabled": enabled}


def patch_copy(source, output, *, enabled=True):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint private source required")
    parents = dict(PARENT_HASHES, **{"native_extension.cc": HOST_PARENT})
    for name, expected in parents.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 source required: {name}")
    src = source / "src"
    text, host, header, proof = transform((src / "asymmetric_native.cuh").read_text(),
        (src / "native_extension.cc").read_text(), (src / "native_moving.h").read_text(), enabled=enabled)
    files = {"asymmetric_native.cuh": text, "native_extension.cc": host, "native_moving.h": header}
    proof.update({"parent_hashes": parents, "result_hashes": {name: sha(value.encode()) for name, value in files.items()},
                  "producer_sha256": sha(Path(__file__).read_bytes()), "gpu_executed": False,
                  "requires": ["actual512privateH/permanentEcompactYdataflowandhazardoracle",
                               "zero stack/local/spill64kernelcompile", "all30exactGPUparity",
                               "memcheck/synccheck/racecheck", "matchedfullworkloadcomparison"]})
    shutil.copytree(source, output)
    for name, value in files.items():
        (output / "src" / name).write_text(value)
    (output.parent / "hybrid-y-frontier-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector-off", action="store_true")
    args = parser.parse_args()
    print(json.dumps(patch_copy(args.source, args.output, enabled=not args.selector_off)))


if __name__ == "__main__":
    main()
