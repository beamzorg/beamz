"""Copy-only NoIO warp4Y×8X/all8Z register frontier on selected52865.

The actual ordered field graph is unchanged. A generated compact shared layout
contains only producer faces needed by other warps. Within-warp X/Y reads use
unconditional full-mask shuffles; all Z reads stay in their owner's registers.
Two opposite-family barriers publish faces and retire their readers. The
512-thread launch allows a128-register budget; compiled resource fit is a gate.
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

from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES, once, transformed as permanent
from scripts.cuda_warp_frontier_layout import descriptor, emit_header, widths


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(original, *, warp_y=4):
    wy, wx = widths(warp_y)
    full, base_proof = permanent(original, scope=2)
    begin = full.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void PermanentFamilyBody")
    end = full.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", begin)
    body = full[begin:end]
    body = once(body,
        " constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;",
        " constexpr int T=512,Slots=8,ZStep=1;\n static_assert(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE,\"Initial warp frontier admits certified NoIO only\");")
    body = once(body,
        " static_assert(T==1024,\"Permanent-family frontier requires admitted1024-thread owner mapping\");",
        " static_assert(T==512&&Slots==8,\"One lane owns every Z point in its XY column\");")
    body = once(body, "frame+6*PAD);", "frame+WarpFrontierWords);")
    body = once(body, "float*low_stage=frame+6*PAD+8*sizeof(size_t)/sizeof(float)+4;",
        "float*low_stage=frame+WarpFrontierWords+8*sizeof(size_t)/sizeof(float)+4;")
    body = once(body,
        " const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/512;",
        " const int lane=threadIdx.x&31;\n const int sx=((threadIdx.x/32)&3)*8+(lane&7),sy=(threadIdx.x/128)*4+(lane/8),zbase=0;")
    body = once(body,
        "for(int c=0;c<3;++c){h[c][j]=Initial(p,c,z,y,x);e[c][j]=Initial(p,c+3,z,y,x);frame[(c+3)*PAD+Slot(zs,sy,sx)]=e[c][j];}",
        "for(int c=0;c<3;++c){h[c][j]=Initial(p,c,z,y,x);e[c][j]=Initial(p,c+3,z,y,x);WarpFrontierStore(frame,c+3,zs,sy,sx,e[c][j]);}")
    body = once(body,
        "   frame[3*PAD+Slot(zs,sy,sx)]=v.ex;frame[4*PAD+Slot(zs,sy,sx)]=v.ey;frame[5*PAD+Slot(zs,sy,sx)]=v.ez;",
        "   WarpFrontierStore(frame,3,zs,sy,sx,v.ex);WarpFrontierStore(frame,4,zs,sy,sx,v.ey);WarpFrontierStore(frame,5,zs,sy,sx,v.ez);")
    body = once(body,
        "for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);\n    if(zs!=hz&&sy!=hy&&sx!=hx){",
        "for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);\n    // Every lane participates, including inactive physical boundary/ghost lanes.\n    const float ex_y=__shfl_sync(0xffffffffu,e[0][j],(lane+8)&31);\n    const float ez_y=__shfl_sync(0xffffffffu,e[2][j],(lane+8)&31);\n    const float ey_x=__shfl_sync(0xffffffffu,e[1][j],(lane&24)|((lane+1)&7));\n    const float ez_x=__shfl_sync(0xffffffffu,e[2][j],(lane&24)|((lane+1)&7));\n    if(zs!=hz&&sy!=hy&&sx!=hx){")
    body = once(body,
        "if(!dz&&!ddy&&!ddx)return e[c][j];return frame[(c+3)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
        "if(dz)return e[c][(j+1)&7];\n      if(ddy){if((sy&3)!=3)return c==0?ex_y:ez_y;return WarpFrontierLoad(frame,field,zs,sy+1,sx);}\n      if(ddx){if((sx&7)!=7)return c==1?ey_x:ez_x;return WarpFrontierLoad(frame,field,zs,sy,sx+1);}\n      return e[c][j];")
    body = once(body,
        "}h[c][j]=value;frame[c*PAD+Slot(zs,sy,sx)]=value;}",
        "}h[c][j]=value;WarpFrontierStore(frame,c,zs,sy,sx,value);}")
    body = once(body,
        "   frame[Slot(zs,sy,sx)]=v.hx;frame[PAD+Slot(zs,sy,sx)]=v.hy;frame[2*PAD+Slot(zs,sy,sx)]=v.hz;",
        "   WarpFrontierStore(frame,0,zs,sy,sx,v.hx);WarpFrontierStore(frame,1,zs,sy,sx,v.hy);WarpFrontierStore(frame,2,zs,sy,sx,v.hz);")
    body = once(body,
        "for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,ys,hx)]=low_stage[c*LowXPoints+i];",
        "for(int c=0;c<3;++c)WarpFrontierStore(frame,c,zs,ys,hx,low_stage[c*LowXPoints+i]);")
    body = once(body,
        "  RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);",
        "  // Unlike the original interleaved-Z mapping, this point owner holds\n  // all eight Z slots. Gather the staged low-X packet only after PF1.\n  if(sx==hx&&sy!=hy){\n   #pragma unroll\n   for(int j=0;j<Slots;++j){const int zs=j;if(zs!=hz){const int key=(((zs-hz)&7)-1)*BY+dy-1;\n    #pragma unroll\n    for(int c=0;c<3;++c){h[c][j]=low_stage[c*LowXPoints+key];e[c][j]=low_stage[(c+3)*LowXPoints+key];}\n   }}\n  }")
    body = once(body,
        "for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);\n    if(zs!=ehz&&sy!=ehy&&sx!=ehx){",
        "for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);\n    const float hx_y=__shfl_sync(0xffffffffu,h[0][j],(lane+24)&31);\n    const float hz_y=__shfl_sync(0xffffffffu,h[2][j],(lane+24)&31);\n    const float hy_x=__shfl_sync(0xffffffffu,h[1][j],(lane&24)|((lane+7)&7));\n    const float hz_x=__shfl_sync(0xffffffffu,h[2][j],(lane&24)|((lane+7)&7));\n    if(zs!=ehz&&sy!=ehy&&sx!=ehx){")
    body = once(body,
        "if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
        "if(field>=3)return e[field-3][j];\n      if(dz)return h[field][(j+7)&7];\n      if(ddy){if(sy&3)return field==0?hx_y:hz_y;return WarpFrontierLoad(frame,field,zs,sy-1,sx);}\n      if(ddx){if(sx&7)return field==1?hy_x:hz_x;return WarpFrontierLoad(frame,field,zs,sy,sx-1);}\n      return h[field][j];")
    body = once(body,
        "for(int c=0;c<3;++c)frame[(c+3)*PAD+Slot(zs,sy,sx)]=e[c][j];",
        "for(int c=0;c<3;++c)WarpFrontierStore(frame,c+3,zs,sy,sx,e[c][j]);")
    if warp_y != 4:
        body = once(body,
            " const int sx=((threadIdx.x/32)&3)*8+(lane&7),sy=(threadIdx.x/128)*4+(lane/8),zbase=0;",
            f" const int sx=((threadIdx.x/32)&{wy-1})*{wx}+(lane&{wx-1}),sy=(threadIdx.x/{32*wy})*{wy}+(lane/{wx}),zbase=0;")
        for old, new in (("(lane+8)&31", f"(lane+{wx})&31"),
                         ("(lane+24)&31", f"(lane+{32-wx})&31"),
                         ("(lane&24)|((lane+1)&7)", f"(lane&{32-wx})|((lane+1)&{wx-1})"),
                         ("(lane&24)|((lane+7)&7)", f"(lane&{32-wx})|((lane+{wx-1})&{wx-1})"),
                         ("(sy&3)!=3", f"(sy&{wy-1})!={wy-1}"),
                         ("(sx&7)!=7", f"(sx&{wx-1})!={wx-1}"),
                         ("if(sy&3)", f"if(sy&{wy-1})"),
                         ("if(sx&7)", f"if(sx&{wx-1})")):
            # Forward/backward Y source happens to coincide for two Y lanes;
            # replacements are made once from the original rather than
            # cascading equal substitution strings into each other.
            body = body.replace(old, new)
    if "frame[" in body.replace("float frame[]", ""):
        raise ValueError("Whole-plane shared field consumer remains")
    changed = full[:begin] + body + full[end:]
    shared_bytes = descriptor(warp_y)["shared_words"]*4+2600
    changed = once(changed,
        '#include "asymmetric_native_cell_generated.cuh"',
        '#include "asymmetric_native_cell_generated.cuh"\n#include "warp_frontier_layout_generated.cuh"')
    changed = once(changed,
        "constexpr size_t PermanentFamilySharedBytes=6ULL*PAD*sizeof(float)+8*sizeof(size_t)+4*sizeof(int)+6ULL*LowXPoints*sizeof(float);\nstatic_assert(PermanentFamilySharedBytes==100904",
        f"constexpr size_t PermanentFamilySharedBytes=WarpFrontierWords*sizeof(float)+8*sizeof(size_t)+4*sizeof(int)+6ULL*LowXPoints*sizeof(float);\nstatic_assert(PermanentFamilySharedBytes=={shared_bytes}")
    changed = once(changed,
        "__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(",
        "__global__ __launch_bounds__(512,1) void AsymmetricNoIOWarpFrontier512(")
    if changed.count("AsymmetricNoIOUniform1024<Depth>") != 2:
        raise ValueError("Expected exact NoIO host kernel binding")
    changed = changed.replace("AsymmetricNoIOUniform1024<Depth>", "AsymmetricNoIOWarpFrontier512<Depth>")
    changed = once(changed,
        "AsymmetricNoIOWarpFrontier512<Depth><<<uniform_count,1024,PermanentFamilySharedBytes,stream>>>",
        "AsymmetricNoIOWarpFrontier512<Depth><<<uniform_count,512,PermanentFamilySharedBytes,stream>>>")
    marker = "// Private source-only storage schedule; source-bound"
    if original[original.index(marker):original.index("inline bool MakeSource")] != changed[changed.index(marker):changed.index("inline bool MakeSource")]:
        raise ValueError("Paired source bodies changed")
    header = emit_header(warp_y)
    return changed, header, {"parent_permanent_family_proof": base_proof,
        "layout": descriptor(warp_y), "shared_bytes": shared_bytes,
        "loop_barriers": 2, "kernel_threads": 512, "launch_bound_register_budget": 128,
        "canonical_cell_byte_identical": True, "paired_source_bodies_byte_identical": True,
        "field_graphs_and_frontier_faces": "derived from actual canonical reads; center/Z registers, XY warp-or-shared-face",
        "counter_visibility": "dual opaque clocks published by opposite family barriers",
        "private_zy_import_order": "after whole own-warp H loop; cross-warp consumers use shared faces",
        "coalescing": f"{wy}rowsegments of{wx}floats/warp; sector overhead must be modeled/measured, not assumed aligned",
        "generated_header_sha256": sha(header.encode()), "body_sha256": sha(body.encode())}


def patch_copy(source, output, *, warp_y=4):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    for name, expected in PARENT_HASHES.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 parent required: {name}")
    changed, header, proof = transformed((source / "src/asymmetric_native.cuh").read_text(), warp_y=warp_y)
    proof.update({"parent_hashes": PARENT_HASHES, "patcher_sha256": sha(Path(__file__).read_bytes()),
                  "result_source_sha256": sha(changed.encode()), "gpu_executed": False})
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(changed)
    (output / "src/warp_frontier_layout_generated.cuh").write_text(header)
    (output.parent / "warp-frontier-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--warp-y", type=int, choices=(2, 4, 8), default=4)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, warp_y=a.warp_y)))


if __name__ == "__main__":
    main()
