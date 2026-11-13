"""Private bundled NoIO dispatch with budgeted mixed register/global Epsi.

Parent is the admitted joint-phase build. One CTA-uniform wrapper keeps the
original complete certified prefix: Z retains one color, Y/X two, other masks
retain the old joint/global body. No cohort sorting or additional launches.
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

from scripts.patch_cuda_native_cpml_strides import ADMISSION, NEW, transform
from scripts.patch_cuda_native_runtime_fact_cell import once
from scripts.probe_cuda_native_epsi1024 import PARENT, storage_source

HOST_PARENT = "c7bd59160aa12fc86b0a54bc4f355d4dbd45c81b7276fb51f2e2e25f9e2cc9a0"


def point_helpers(*, debug=False):
    result = []
    for count, suffix in [(1, "A"), (2, "Both")]:
        source, _ = storage_source(count)
        source = source.replace("EPsiStorage", "BudgetedStorage" + suffix).replace("PrivateEPsiPoint", "BudgetedPoint" + suffix)
        offset = NEW.replace("Packed(", "this->Packed(").replace("p.", "this->p.")
        methods = """ __device__ __forceinline__ int PsiOffset(int slot,int z,int y,int x)const{
@OFFSET@
 }
 __device__ __forceinline__ float GlobalLoad(int slot,int z,int y,int x)const{
  const auto&psi=this->p.phase[slot/6].psi[slot%6];return psi.input[PsiOffset(slot,z,y,x)];
 }
 __device__ __forceinline__ void GlobalStore(int slot,int z,int y,int x,float value)const{
  const auto&psi=this->p.phase[slot/6].psi[slot%6];psi.output[PsiOffset(slot,z,y,x)]=value;
 }
""".replace("@OFFSET@", offset)
        source = once(source, " __device__ __forceinline__ float& State", methods + " __device__ __forceinline__ float& State")
        source = source.replace("Base::LoadLocal(slot,z,y,x)", "GlobalLoad(slot,z,y,x)")
        source = source.replace("Base::StoreLocal(slot,z,y,x,value)", "GlobalStore(slot,z,y,x,value)")
        if debug:
            source = once(source, "  return first?GlobalLoad(slot,z,y,x):State(slot);", """  if(first)return GlobalLoad(slot,z,y,x);
  atomicOr(this->p.bad,8<<beamz::cuda::yee::CpmlAxis(slot%6)); // Validation only: an actual retained private read.
  return State(slot);""")
        result.append(source)
    result.append("""
template<int Mask,int Slots> struct BudgetedState {};
template<int Slots> struct BudgetedState<1,Slots> : BudgetedStorageA<true,Slots> {};
template<int Slots> struct BudgetedState<3,Slots> : BudgetedStorageBoth<true,Slots> {};
""")
    return "\n".join(result)


def patch_consumer(source, *, debug=False):
    if hashlib.sha256(source.encode()).hexdigest() != PARENT:
        raise ValueError("Exact joint-phase parent required")
    source = once(source, "bool UniformE=false> struct Environment", "bool UniformE=false,int ActiveAxes=-1> struct Environment")
    source = once(source, "  if(!(flags&(2u<<axis)))return false;",
                  "  if constexpr(ActiveAxes>=0){if(!(ActiveAxes&(1u<<axis)))return false;}\n  else if(!(flags&(2u<<axis)))return false;")
    source = once(source, "template<int Phase,int Component,bool UseRuntimeFactCell,bool AliasProfiles,class Env,class Read>",
                  point_helpers(debug=debug) + "\ntemplate<int Phase,int Component,bool UseRuntimeFactCell,bool AliasProfiles,class Env,class Read>")
    source = once(source, "bool JointPhase=false>\n__device__", "bool JointPhase=false,int ActiveAxes=-1,int PsiCacheMask=0>\n__device__")
    source = once(source, " float h[3][Slots],e[3][Slots];", """ static_assert(PsiCacheMask==0||PsiCacheMask==1||PsiCacheMask==3,"Supported register budgets only");
 static_assert(PsiCacheMask==0||(JointPhase&&AliasProfiles&&UseRuntimeFactCell&&Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE&&(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4)),"Private psi requires the complete admitted one-axis NoIO class");
 BudgetedState<PsiCacheMask,Slots> epsi;
 float h[3][Slots],e[3][Slots];""")
    source = once(source, " Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};",
                  " Environment<Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes>env{p,flags};")
    source = once(source, "    if(zs!=ehz&&sy!=ehy&&sx!=ehx){\n     const auto read=", """    if(zs!=ehz&&sy!=ehy&&sx!=ehx){
     auto point_env=[&] __device__(){
      if constexpr(PsiCacheMask){const int dz=(zs-ehz)&7,dy=(sy-ehy)&15,dx=(sx-ehx)&31;
       const bool fresh=n==window[0]||dz==1||dy==1||dx==1,flush=n==Depth||dz==BZ||dy==BY||dx==BX;
       if constexpr(PsiCacheMask==1)return BudgetedPointA<decltype(env)>(env,epsi.a[j],fresh,flush);
       else return BudgetedPointBoth<decltype(env)>(env,epsi.a[j],epsi.b[j],fresh,flush);
      }else return env;
     }();
     const auto read=""")
    source = once(source, "trace_native_phase::AdvanceNativePhase<1>(env,read,n,z,y,x)",
                  "trace_native_phase::AdvanceNativePhase<1>(point_env,read,n,z,y,x)")
    old = """ UniformBody<Depth,1,false,true,1,true,true,true,true>(p,tiles);
}"""
    new = """ // Immutable per-CTA flag: every thread follows the same complete barrier body.
 const unsigned axes=p.flags[tiles[blockIdx.x]]&14u;
 if(axes==2u){UniformBody<Depth,1,false,true,1,true,true,true,true,1,1>(p,tiles);return;}
 if(axes==4u){UniformBody<Depth,1,false,true,1,true,true,true,true,2,3>(p,tiles);return;}
 if(axes==8u){UniformBody<Depth,1,false,true,1,true,true,true,true,4,3>(p,tiles);return;}
 UniformBody<Depth,1,false,true,1,true,true,true,true,-1,0>(p,tiles);
}"""
    source = once(source, old, new)
    return source.replace("AsymmetricNoIOProfilePhase1024", "AsymmetricNoIOBudgetedEPsi1024")


def patch_host(source, *, debug=False):
    if hashlib.sha256(source.encode()).hexdigest() != HOST_PARENT:
        raise ValueError("Exact joint host/profile/state admission required")
    source = once(source, '    result["moving_joint_state_independent"]', """    size_t budgeted_axes[3]{};
    if((moving_workspace_.data_flags&beamz::cuda::moving::kEqualAxisProfiles)&&!moving_plan_.policy_offsets.empty()){
      for(size_t rank=0;rank<moving_plan_.uniform_noio_counts.size();++rank){const int first=moving_plan_.policy_offsets[7*rank+1],count=moving_plan_.uniform_noio_counts[rank];
        for(int i=0;i<count;++i){const unsigned bits=moving_plan_.tile_flags[moving_plan_.tile_ids[first+i]]&14u;
          if(bits==2)++budgeted_axes[0];else if(bits==4)++budgeted_axes[1];else if(bits==8)++budgeted_axes[2];}}
    }
    result["moving_budgeted_epsi_tiles_zyx"]=nb::make_tuple(budgeted_axes[0],budgeted_axes[1],budgeted_axes[2]);
    result["moving_budgeted_epsi_tiles"]=budgeted_axes[0]+budgeted_axes[1]+budgeted_axes[2];
    result["moving_budgeted_epsi_private_floats_zyx"]=nb::make_tuple(4,8,8);
    result["moving_joint_state_independent"]""")
    source = once(source, '  module.attr("moving_joint_profile_phase")=true;',
                '  module.attr("moving_joint_profile_phase")=true;\n  module.attr("moving_budgeted_epsi")=true;\n  module.attr("moving_budgeted_epsi_dispatch")="bundled-certified-NoIO-prefix";')
    if debug:
        source = once(source, "      if(bad)throw std::runtime_error", "      moving_budgeted_epsi_debug_raw_=bad;\n      if(bad&~56)throw std::runtime_error")
        source = once(source, "  bool moving_equal_axis_profiles_=false,", "  int32_t moving_budgeted_epsi_debug_raw_=0;\n  bool moving_equal_axis_profiles_=false,")
        source = once(source, '    result["moving_budgeted_epsi_tiles_zyx"]',
                      '    result["moving_budgeted_epsi_debug_raw"]=moving_budgeted_epsi_debug_raw_;result["moving_budgeted_epsi_debug_real_errors"]=moving_budgeted_epsi_debug_raw_&~56;result["moving_budgeted_epsi_debug_hits_zyx"]=nb::make_tuple(bool(moving_budgeted_epsi_debug_raw_&8),bool(moving_budgeted_epsi_debug_raw_&16),bool(moving_budgeted_epsi_debug_raw_&32));\n    result["moving_budgeted_epsi_tiles_zyx"]')
        source = once(source, '  module.attr("moving_budgeted_epsi")=true;', '  module.attr("moving_budgeted_epsi")=true;\n  module.attr("moving_budgeted_epsi_debug")=true;')
    return source


def patch_copy(parent, output, *, debug=False):
    if output.exists() or output.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        raise ValueError("New private source tree required")
    src = parent / "src"
    original = (src / "asymmetric_native.cuh").read_bytes()
    # Verify all actual canonical dense24/inplace/int32 admission anchors before
    # installing the same integer formula ONLY in the cached point accessors.
    transform(original, (src / "native_extension.cc").read_bytes(), (src / "asymmetric_native_cell_generated.cuh").read_bytes())
    consumer, host = patch_consumer(original.decode(), debug=debug), patch_host((src / "native_extension.cc").read_text(), debug=debug)
    shutil.copytree(parent, output)
    (output / "src/asymmetric_native.cuh").write_text(consumer)
    (output / "src/native_extension.cc").write_text(host)
    proof = {"parent_consumer_sha256": PARENT, "parent_host_sha256": HOST_PARENT,
             "consumer_sha256": hashlib.sha256(consumer.encode()).hexdigest(),
             "host_sha256": hashlib.sha256(host.encode()).hexdigest(),
             "point_helpers_sha256": hashlib.sha256(point_helpers(debug=debug).encode()).hexdigest(),
             "canonical_stride_admission": ADMISSION,
             "dispatch": "One existing profile/disjoint-gated NoIO kernel; uniform axes2→Z/colorA,4→Y/both,8→X/both; other masks old joint/global; profile/state fact false old scalar wholeprefix",
             "unchanged": ["NoIO IDs/order/slots/counts", "rank/stream/event schedule", "field/source/DFT/clocks", "Hpsi global", "all old fallback pointer addressing"],
             "storage": "Z4,Y8,X8 private floats/thread; unchanged firstowned/windowstart reload and outgoinghigh/depth flush; otherErecurrence ordinary global onZ",
             "scope": "Full runtime candidate; resource/GPU all-state/sanitizer/timing gates still required",
             "validation_only": debug, "diagnostic_reserved_bits": [8, 16, 32] if debug else []}
    (output.parent / "budgeted-epsi-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--debug-cache", action="store_true", help="Validation only; actual retained reads mark reserved diagnostic bits, never time this build")
    a = p.parse_args()
    result = patch_copy(a.parent, a.output, debug=a.debug_cache)
    print(json.dumps({k: result[k] for k in ("consumer_sha256", "host_sha256", "point_helpers_sha256")}))


if __name__ == "__main__":
    main()
