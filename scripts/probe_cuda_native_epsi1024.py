"""Resource-only private Epsi storage on the admitted joint-phase NoIO body.

No runtime/host cohort changes. The two unlaunched wrappers require one immutable
CPML axis flag. Derived logical E-term colors occupy two arrays[Slots4], versus
the prior512-thread implementation's two arrays[Slots8]. GPU use is out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import yee_operator
from scripts.cuda_trace_guarded_storage import (
    FIXED_E_COLORS,
    OwnedRegion,
    terms_from_ir,
)
from scripts.patch_cuda_native_cpml_strides import ADMISSION, NEW, OLD, transform
from scripts.patch_cuda_native_runtime_fact_cell import once

PARENT = "b514c317c82a4e680d304745ec5a8ffb74f9be47bdd4874b3742954521f10906"
NVCC = "/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/nvcc"


def storage_source(cache_colors=2):
    if cache_colors not in (1, 2):
        raise ValueError("One or two admitted private colors required")
    terms, operator_sha = terms_from_ir(yee_operator(cpml_axes=(0, 1, 2)),
                                        OwnedRegion((128, 128, 128), (7, 45, 62), (7, 15, 31), 16))
    by_axis = {axis: [t["term"] for t in terms if t["axis"] == axis] for axis in range(3)}
    if any(len(pair) != 2 or FIXED_E_COLORS[pair[0]] == FIXED_E_COLORS[pair[1]] for pair in by_axis.values()):
        raise ValueError("IR no longer admits the two guarded colors")
    mask = sum(1 << term for term, color in enumerate(FIXED_E_COLORS) if color == 1)
    source = f"""
template<bool Enabled,int Slots> struct EPsiStorage {{}};
template<int Slots> struct EPsiStorage<true,Slots> {{float a[Slots],b[Slots];}};
template<class Base> struct PrivateEPsiPoint : Base {{
 float &a,&b;bool first,flush;
 __device__ __forceinline__ PrivateEPsiPoint(Base&base,float&pa,float&pb,bool fresh,bool publish)
 :Base(base),a(pa),b(pb),first(fresh),flush(publish){{}}
 __device__ __forceinline__ float& State(int slot)const{{return (({mask}u>>(slot%6))&1u)?b:a;}}
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{{
  if(slot<6)return Base::LoadLocal(slot,z,y,x);
  return first?Base::LoadLocal(slot,z,y,x):State(slot);
 }}
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{{
  if(slot<6){{Base::StoreLocal(slot,z,y,x,value);return;}}
  State(slot)=value;if(flush)Base::StoreLocal(slot,z,y,x,value);
 }}
}};
"""
    proof = {"operator_sha256": operator_sha, "axis_terms": by_axis,
                    "colors": FIXED_E_COLORS, "color_b_mask": mask,
                    "guard": "Wrapper rejects flags&14 other than2/4/8; LocalActive retains phase/support/profile guards"}
    if cache_colors == 1:
        selected = sum(1 << term for term, color in enumerate(FIXED_E_COLORS) if color == 0)
        source = source.replace("{float a[Slots],b[Slots];}", "{float a[Slots];}")
        source = source.replace("float &a,&b;", "float &a;")
        source = source.replace("float&pa,float&pb,bool fresh", "float&pa,bool fresh")
        source = source.replace(":Base(base),a(pa),b(pb),first", ":Base(base),a(pa),first")
        source = source.replace(f"return (({mask}u>>(slot%6))&1u)?b:a;", "return a;")
        source = source.replace("if(slot<6)return Base::LoadLocal", f"if(slot<6||!(({selected}u>>(slot%6))&1u))return Base::LoadLocal")
        source = source.replace("if(slot<6){Base::StoreLocal", f"if(slot<6||!(({selected}u>>(slot%6))&1u)){{Base::StoreLocal")
        proof.update(cached_colors=[0], global_colors=[1], cached_term_mask=selected)
    return source, proof


def patch(source, *, axis_specialized=False, cache_colors=2):
    if hashlib.sha256(source.encode()).hexdigest() != PARENT:
        raise ValueError("Exact frozen joint-phase parent required")
    helper, proof = storage_source(cache_colors)
    source = once(source, '#include "compact_full_edge.cuh"', '#include "compact_types.cuh"')
    source = once(source, "template<int Phase,int Component,bool UseRuntimeFactCell,bool AliasProfiles,class Env,class Read>",
                  helper + "\ntemplate<int Phase,int Component,bool UseRuntimeFactCell,bool AliasProfiles,class Env,class Read>")
    source = once(source, "bool JointPhase=false>\n__device__", "bool JointPhase=false,bool CacheE=false>\n__device__")
    source = once(source, " float h[3][Slots],e[3][Slots];", """ static_assert(!CacheE||(JointPhase&&Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE),"Private Epsi needs admitted NoIO joint phase");
 EPsiStorage<CacheE,Slots> epsi;
 float h[3][Slots],e[3][Slots];""")
    source = once(source, "    if(zs!=ehz&&sy!=ehy&&sx!=ehx){\n     const auto read=", """    if(zs!=ehz&&sy!=ehy&&sx!=ehx){
     auto point_env=[&] __device__(){
      if constexpr(CacheE){const int dz=(zs-ehz)&7,dy=(sy-ehy)&15,dx=(sx-ehx)&31;
       return PrivateEPsiPoint<decltype(env)>(env,epsi.a[j],epsi.b[j],n==window[0]||dz==1||dy==1||dx==1,n==Depth||dz==BZ||dy==BY||dx==BX);
      }else return env;
     }();
     const auto read=""")
    source = once(source, "trace_native_phase::AdvanceNativePhase<1>(env,read,n,z,y,x)",
                  "trace_native_phase::AdvanceNativePhase<1>(point_env,read,n,z,y,x)")
    if cache_colors == 1:
        source = once(source, "env,epsi.a[j],epsi.b[j],n==", "env,epsi.a[j],n==")
    if axis_specialized:
        source = once(source, "bool UniformE=false> struct Environment", "bool UniformE=false,int ActiveAxes=-1> struct Environment")
        source = once(source, "  if(!(flags&(2u<<axis)))return false;",
                      "  if constexpr(ActiveAxes>=0){if(!(ActiveAxes&(1u<<axis)))return false;}\n  else if(!(flags&(2u<<axis)))return false;")
        source = once(source, "bool CacheE=false>\n__device__", "bool CacheE=false,int ActiveAxes=-1>\n__device__")
        source = once(source, "Environment<Layout,Fast,IsotropicH,Policy,UniformE>env",
                      "Environment<Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes>env")
        proof["axis_guard_specialization"] = "Exact constexpr LocalActive lowering from the same one-hot guard fact; all coordinate/support/FP nodes retained"
    marker = "\ntemplate<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__"
    if source.count(marker) != 1:
        raise ValueError("Resource-only body boundary changed")
    return source[:source.index(marker)] + "\n} // namespace beamz::cuda::moving_native\n", proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--axis-specialized", action="store_true")
    p.add_argument("--axis-masks", type=int, nargs="+", choices=(1, 2, 4))
    p.add_argument("--canonical-psi-strides", action="store_true")
    p.add_argument("--cache-colors", type=int, choices=(1, 2), default=2)
    a = p.parse_args()
    parent = (a.parent / "src").resolve()
    if a.axis_masks and not a.axis_specialized:
        p.error("Axis mask selection requires typed axis specialization")
    text, proof = patch((parent / "asymmetric_native.cuh").read_text(), axis_specialized=a.axis_specialized, cache_colors=a.cache_colors)
    if a.canonical_psi_strides:
        original = (parent / "asymmetric_native.cuh").read_bytes()
        full_changed = transform(original, (parent / "native_extension.cc").read_bytes(),
                                 (parent / "asymmetric_native_cell_generated.cuh").read_bytes())
        if full_changed.replace(NEW.encode(), OLD.encode(), 1) != original:
            raise ValueError("Whole-parent stride inverse failed")
        before = text
        text = once(text, OLD, NEW)
        if text.replace(NEW, OLD, 1) != before:
            raise ValueError("Composed resource stride inverse failed")
        proof["canonical_psi_strides"] = {"inverse_source_binding": True, "host_admission_anchors": ADMISSION,
                                          "old_sha256": hashlib.sha256(OLD.encode()).hexdigest(),
                                          "new_sha256": hashlib.sha256(NEW.encode()).hexdigest(),
                                          "scope": "Exactly root's admitted integer PsiOffset helper; no float/pointer/state/schedule change"}
    a.output.mkdir(parents=True, exist_ok=True)
    consumer = a.output / "consumer.cuh"
    if consumer.exists() and consumer.read_text() != text:
        raise ValueError("Refusing to overwrite frozen resource source")
    consumer.write_text(text)
    types = (parent / "compact_full_edge.cuh").read_text().split("__device__ __forceinline__ int CompactEdgeOffset(", 1)[0]
    if "struct MstepCompactEdgeView" not in types or "__device__" in types:
        raise ValueError("Exact standalone compact descriptor prefix required")
    (a.output / "compact_types.cuh").write_text(types)
    source = '#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cstddef>\n#include <cstdint>\n#include "yee_primitives.cuh"\n#include "consumer.cuh"\nusing namespace beamz::cuda::moving_native;\n'
    for depth in (4, 16):
        for axes in (tuple(a.axis_masks or (1, 2, 4)) if a.axis_specialized else (-1,)):
            suffix = f"Axis{axes}" if a.axis_specialized else ""
            if a.cache_colors == 1:
                suffix += "ColorA"
            guard = f"mask!={axes << 1}u" if a.axis_specialized else "mask!=2u&&mask!=4u&&mask!=8u"
            template_axis = f",{axes}" if a.axis_specialized else ""
            source += f'''extern "C" __global__ __launch_bounds__(1024,1) void NoIOEPsiDepth{depth}{suffix}(const __grid_constant__ Parameters p,const int32_t*tiles){{
 const unsigned mask=p.flags[tiles[blockIdx.x]]&14u;if({guard})return;
 UniformBody<{depth},1,false,true,1,true,true,true,true,true{template_axis}>(p,tiles);
}}
'''
    probe = a.output / "probe.cu"
    if probe.exists() and probe.read_text() != source:
        raise ValueError("Refusing to overwrite frozen probe source")
    probe.write_text(source)
    command = [NVCC, "-c", str(probe.resolve()), "-o", str((a.output / "probe.o").resolve()),
               "-std=c++17", "-O3", "-DNDEBUG", "--generate-code=arch=compute_86,code=[compute_86,sm_86]",
               "--relocatable-device-code=true", "-Xcompiler=-fPIC", "--expt-relaxed-constexpr", "--extended-lambda",
               "-lineinfo", "-Xptxas=-v", "-DBEAMZ_CUDA_NATIVE_MOVING_POLICIES=1",
               "-DBEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES=1", "-DBEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION=1",
               "-I" + str(a.output.resolve()), "-I" + str(parent)]
    report = {"parent_consumer_sha256": PARENT, "consumer_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "compact_types_sha256": hashlib.sha256(types.encode()).hexdigest(),
              "probe_sha256": hashlib.sha256(source.encode()).hexdigest(), "storage": proof,
              "compile_command": command, "admission": "Resource-only, no host dispatch; NoIO1024/UniformE/profile/disjoint-state parent plus runtime one-hotCPML flag",
              "new_private_state": ("Two arrays[Slots4],8 FP32 scalars/thread; Hpsi remains global" if a.cache_colors == 2 else "One array[Slots4],4 FP32 scalars/thread; Hpsi and second E recurrence remain global"), "axis_specialized": a.axis_specialized,
              "axis_masks": a.axis_masks, "canonical_psi_strides": a.canonical_psi_strides,
              "gpu_executed": False}
    (a.output / "manifest-before-build.json").write_text(json.dumps(report, indent=2) + "\n")
    if a.compile:
        if (a.output / "probe.o").exists():
            raise ValueError("Refusing to replace immutable compiled resource probe")
        with (a.output / "compile.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        report["compile_exit_code"] = result.returncode
        (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        result.check_returncode()
    print(json.dumps({k: report[k] for k in ("consumer_sha256", "probe_sha256", "new_private_state", "gpu_executed")}))


if __name__ == "__main__":
    main()
