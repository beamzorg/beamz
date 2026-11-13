"""Copy-only two-barrier permanent-family frontier on selected52865.

The admitted1024-thread UniformBody classes keep private H/E values and gain
separate shared H/E banks. H publications never overwrite old E, and E
publications never overwrite current H. Opaque per-family phase clocks retain
the compiler's rematerialization boundary without a single-clock race.
This changes storage/synchronization, not the generated ordered cell graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PARENT_MODULE = "52865a3ac8371e2390749953eb3af99746df108fe7bcb00fc4be8b35f4259b92"
PARENT_HASHES = {
    "asymmetric_native.cuh": "f082980ad0687ef165c7807b415e87cbc1003edfdbd2acf7a4ee018dae4d4547",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
    "native_moving.h": "d3d1abaca2ca066b75295df35ec26da2c1df53c75bdb9f140b067fea09288b3c",
    "native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one permanent-frontier anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(original, *, scope=7):
    if sha(original.encode()) != PARENT_HASHES["asymmetric_native.cuh"]:
        raise ValueError("Exact selected52865 parent source required")
    if scope not in range(1, 8):
        raise ValueError("Scope bits1bulk/2NoIO/4observer required")
    begin = original.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody")
    end = original.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__", begin)
    old_body = original[begin:end]
    new_body = once(old_body, "void UniformBody(", "void PermanentFamilyBody(")
    new_body = once(new_body,
        " extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);\n volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);\n volatile int*window=phase_step+1;\n float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;",
        " static_assert(T==1024,\"Permanent-family frontier requires admitted1024-thread owner mapping\");\n extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+6*PAD);\n volatile int*h_step=reinterpret_cast<volatile int*>(bases+8);\n volatile int*e_step=h_step+1;\n volatile int*window=h_step+2;\n float*low_stage=frame+6*PAD+8*sizeof(size_t)/sizeof(float)+4;")
    new_body = once(new_body,
        "frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];}",
        "frame[(c+3)*PAD+Slot(zs,sy,sx)]=e[c][j];}")
    new_body = once(new_body, "if(!threadIdx.x)*phase_step=1;__syncthreads();",
        "if(!threadIdx.x){*h_step=1;*e_step=1;}__syncthreads();")
    new_body = once(new_body,
        "   frame[Slot(zs,sy,sx)]=v.ex;frame[PAD+Slot(zs,sy,sx)]=v.ey;frame[2*PAD+Slot(zs,sy,sx)]=v.ez;",
        "   frame[3*PAD+Slot(zs,sy,sx)]=v.ex;frame[4*PAD+Slot(zs,sy,sx)]=v.ey;frame[5*PAD+Slot(zs,sy,sx)]=v.ez;")
    new_body = once(new_body, "if(!threadIdx.x)*phase_step=first;", "if(!threadIdx.x){*h_step=first;*e_step=first;}")
    new_body = once(new_body, "{const int step=*phase_step,hz=", "{const int step=*h_step,hz=")
    new_body = once(new_body,
        "if(!dz&&!ddy&&!ddx)return e[c][j];return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
        "if(!dz&&!ddy&&!ddx)return e[c][j];return frame[(c+3)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];")
    new_body = once(new_body, "     }h[c][j]=value;}",
        "     }h[c][j]=value;frame[c*PAD+Slot(zs,sy,sx)]=value;}")
    new_body = once(new_body,
        "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;\n  }}",
        "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;\n   frame[Slot(zs,sy,sx)]=v.hx;frame[PAD+Slot(zs,sy,sx)]=v.hy;frame[2*PAD+Slot(zs,sy,sx)]=v.hz;\n  }}")
    old_stage = """  StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // B1 retires all old-E frame readers and publishes low imports.
  RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]=h[c][j];
  }
  __syncthreads(); // B2 publishes all current H values.
  {const int step=*phase_step,ehz="""
    new_stage = """  StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);
  // Coalesced import writers publish only H before B1. Old shared E still
  // has cross-warp readers; its current ghost imports stay private/staged.
  for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,ddy=1+i%BY;
   const int zs=(hz+dz)&7,ys=(hy+ddy)&15;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,ys,hx)]=low_stage[c*LowXPoints+i];
  }
  __syncthreads(); // PF1 publishes current H and retires all prior E readers.
  RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);
  // Opposite barrier publishes each family clock and retires its readers.
  if(!threadIdx.x)*h_step=n+1;
  {const int step=*e_step,ehz="""
    new_body = once(new_body, old_stage, new_stage)
    new_body = once(new_body,
        "  __syncthreads(); // B3 retires current-H frame readers before E overwrite.\n", "")
    new_body = once(new_body,
        "for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];",
        "for(int c=0;c<3;++c)frame[(c+3)*PAD+Slot(zs,sy,sx)]=e[c][j];")
    new_body = once(new_body,
        "  if(!threadIdx.x)*phase_step=n+1;\n  __syncthreads(); // B4 publishes E and the next phase counter.",
        "  __syncthreads(); // PF2 publishes current E and retires all current H readers.\n  if(!threadIdx.x)*e_step=n+1;")
    if "phase_step" in new_body:
        raise ValueError("Single shared phase clock remains")
    old_renamed = once(old_body, "void UniformBody(", "void AliasedFamilyBody(")
    wrapper = f"""template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>
__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){{
 constexpr bool permanent=(Fast&&({scope}&1))||(Layout==1&&IsotropicH&&UniformE&&((Policy==1&&({scope}&2))||(Policy==2&&({scope}&4))));
 if constexpr(permanent)PermanentFamilyBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes>(p,tiles);
 else AliasedFamilyBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes>(p,tiles);
}}

"""
    changed = original[:begin] + old_renamed + new_body + wrapper + original[end:]
    marker = "struct Source {"
    changed = once(changed, marker,
        "constexpr size_t PermanentFamilySharedBytes=6ULL*PAD*sizeof(float)+8*sizeof(size_t)+4*sizeof(int)+6ULL*LowXPoints*sizeof(float);\n"
        "static_assert(PermanentFamilySharedBytes==100904&&PermanentFamilySharedBytes<=99*1024,\"SM86 permanent-family budget\");\n" + marker)
    launch_begin = changed.index("inline cudaError_t Launch(cudaStream_t")
    launch_end = changed.index("template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchNoIO", launch_begin)
    launch = changed[launch_begin:launch_end]
    launch = once(launch, " const auto kernel=Asymmetric<Depth,Layout,Fast,IsotropicH,Policy>;", f" const auto kernel=Asymmetric<Depth,Layout,Fast,IsotropicH,Policy>;\n constexpr size_t bytes=(Fast&&({scope}&1))?PermanentFamilySharedBytes:SharedBytes;")
    launch = once(launch, "cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes)", "cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)")
    launch = once(launch, "<<<count,(Fast?1024:512),SharedBytes,stream>>>", "<<<count,(Fast?1024:512),bytes,stream>>>")
    changed = changed[:launch_begin] + launch + changed[launch_end:]
    for klass, bit in (("NoIO", 2), ("Observer", 4)):
        if not scope & bit:
            continue
        changed = once(changed,
            f"const auto kernel=Asymmetric{klass}Uniform1024<Depth>;\n  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes)",
            f"const auto kernel=Asymmetric{klass}Uniform1024<Depth>;\n  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,PermanentFamilySharedBytes)")
        changed = once(changed,
            f"Asymmetric{klass}Uniform1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>",
            f"Asymmetric{klass}Uniform1024<Depth><<<uniform_count,1024,PermanentFamilySharedBytes,stream>>>")
    # No source-body, cell-expression, admission, trail layout or runtime key changes.
    marker = "// Private source-only storage schedule; source-bound"
    src_begin = original.index(marker)
    src_end = original.index("inline bool MakeSource", src_begin)
    dst_begin = changed.index(marker)
    dst_end = changed.index("inline bool MakeSource", dst_begin)
    if original[src_begin:src_end] != changed[dst_begin:dst_end]:
        raise ValueError("Paired source bodies changed")
    return changed, {"scope_bits": scope, "shared_bytes": 100904,
        "permanent_loop_barriers": 2, "parent_loop_barriers": 4,
        "paired_source_bodies_byte_identical": True,
        "original_body_retained_byte_identical_except_name": old_renamed.replace("void AliasedFamilyBody(", "void UniformBody(", 1) == old_body,
        "opaque_clocks": "H read beforePF1, nextH publish afterPF1; E read beforePF2, nextE publish afterPF2",
        "parent_module_sha256": PARENT_MODULE}


def patch_copy(source, output, *, scope=7):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    parent = source / "src/asymmetric_native.cuh"
    for name, expected in PARENT_HASHES.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 producer/cell source required: {name}")
    changed, proof = transformed(parent.read_text(), scope=scope)
    proof.update({"parent_source_sha256": sha(parent.read_bytes()),
                  "result_source_sha256": sha(changed.encode()),
                  "patcher_sha256": sha(Path(__file__).read_bytes()),
                  "gpu_executed": False})
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(changed)
    (output.parent / "permanent-family-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--scope", type=int, default=7)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, scope=a.scope)))


if __name__ == "__main__":
    main()
