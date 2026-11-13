"""Copy-only canonical-cell shared Epsi runtime on the frozen paired-source winner.

All existing point arithmetic stays in the exact canonical generated cell header.
Only single-axis certified NoIO E recurrences retain their state in exclusive
volatile shared SoA slots. No equal-profile substitution is performed.
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
from scripts.cuda_native_phase_state_fact import generate_guard

PARENT = "3b4039564fad23a88c8e801768a93b1ced302643d640886095e0f0ad1fc29aaf"
HOST = "ec7f48eb42ce5a6c5136e88ea2542c809d786762e26fe9bae8f8a2c211787f45"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
HEADER = "917fe9043362071cd1cac5ba503571a72de3d34b51b5e3c68eeea1fa35dbd7a9"
POINT = r"""
// Canonical scalar recurrence adapter: profiles and all arithmetic are unchanged.
// Each fixed (thread,slot,color) state is private despite shared placement.
template<class Base> struct SharedEPsiPoint : Base {
 volatile float &a,&b;bool first,flush,enabled;
 __device__ __forceinline__ SharedEPsiPoint(Base&base,volatile float&pa,volatile float&pb,bool fresh,bool publish,bool admit)
 :Base(base),a(pa),b(pb),first(fresh),flush(publish),enabled(admit){}
 __device__ __forceinline__ volatile float& State(int slot)const{
  const int term=slot%6;return term==2||term==4||term==5?b:a;
 }
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if(slot<6||!enabled||first)return Base::LoadLocal(slot,z,y,x);
  return State(slot);
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  if(slot<6||!enabled){Base::StoreLocal(slot,z,y,x,value);return;}
  State(slot)=value;if(flush)Base::StoreLocal(slot,z,y,x,value);
 }
};
constexpr size_t SharedEPsiBytes=SharedBytes+8ULL*1024*sizeof(float);
static_assert(SharedBytes==51748&&SharedEPsiBytes==84516,"Frozen shared-bank allocation");
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one anchor: {old!r}")
    return text.replace(old, new, 1)


def transform(c, e, h, *, debug_hits=False):
    if sha(c) != PARENT or sha(e) != HOST or sha(h) != HEADER:
        raise ValueError("Exact frozen v12 consumer, extension and workspace required")
    point = POINT
    if debug_hits:
        point = once(
            point,
            "  return State(slot);",
            "  atomicOr(this->p.bad,8u<<beamz::cuda::yee::CpmlAxis(slot%6)); // Validation-only retained read.\n  return State(slot);",
        )
    text = c.decode()
    text = once(
        text,
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>\n__device__ __forceinline__ void UniformBody",
        point
        + "\ntemplate<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,bool CacheE=false>\n__device__ __forceinline__ void UniformBody",
    )
    text = once(
        text,
        " float h[3][Slots],e[3][Slots];",
        """ static_assert(!CacheE||(T==1024&&Slots==4&&Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE),"Exact shared Epsi class");
 volatile float*epsi_bank=nullptr;
 if constexpr(CacheE)epsi_bank=frame+SharedBytes/sizeof(float);
 float h[3][Slots],e[3][Slots];""",
    )
    start = text.index("  {const int step=*phase_step,ehz=")
    end = text.index("  __syncthreads(); // B3", start)
    region = text[start:end]
    anchor = "    if(zs!=ehz&&sy!=ehy&&sx!=ehx){\n     const auto read="
    replacement = """    if(zs!=ehz&&sy!=ehy&&sx!=ehx){
     auto point_env=[&] __device__(){
      if constexpr(CacheE){
       const int dz=(zs-ehz)&7,dy=(sy-ehy)&15,dx=(sx-ehx)&31;
       const unsigned axes=flags&14u;
       return SharedEPsiPoint<decltype(env)>(env,epsi_bank[j*1024+threadIdx.x],epsi_bank[(Slots+j)*1024+threadIdx.x],
        n==window[0]||dz==1||dy==1||dx==1,n==Depth||dz==BZ||dy==BY||dx==BX,axes==2u||axes==4u||axes==8u);
      }else return env;
     }();
     const auto read="""
    region = once(region, anchor, replacement)
    for comp in range(3):
        region = once(
            region,
            f"AdvanceNativeCell<1,{comp},false>(env,read,n,z,y,x)",
            f"AdvanceNativeCell<1,{comp},false>(point_env,read,n,z,y,x)",
        )
    text = text[:start] + region + text[end:]
    entry = """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOSharedEPsi1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 UniformBody<Depth,1,false,true,1,true,true>(p,tiles);
}

"""
    text = once(
        text,
        "template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024",
        entry
        + "template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024",
    )
    text = once(
        text,
        "int count,int uniform_count){\n if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;\n if constexpr(Layout==1&&IsotropicH){\n  // Existing policy",
        "int count,int uniform_count,bool shared_epsi){\n if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;\n if constexpr(Layout==1&&IsotropicH){\n  // Existing policy",
    )
    text = once(
        text,
        "  const auto kernel=AsymmetricNoIOUniform1024<Depth>;",
        """  if(shared_epsi){
   const auto cached=AsymmetricNoIOSharedEPsi1024<Depth>;
   status=cudaFuncSetAttribute(cached,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedEPsiBytes);if(status!=cudaSuccess)return status;
   AsymmetricNoIOSharedEPsi1024<Depth><<<uniform_count,1024,SharedEPsiBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }
  const auto kernel=AsymmetricNoIOUniform1024<Depth>;""",
    )
    text = once(
        text,
        "count,w.uniform_noio_counts_host[rank]);break;",
        "count,w.uniform_noio_counts_host[rank],BeamzResearchSharedEPsiEnabled&&w.shared_epsi_state_independent);break;",
    )
    header = h.decode()
    header = once(
        header,
        "struct BeamzNativeMovingWorkspace {",
        'extern const bool BeamzResearchSharedEPsiEnabled __attribute__((visibility("hidden")));\n\nstruct BeamzNativeMovingWorkspace {',
    )
    header = once(
        header,
        "  uint32_t source_presence_mask=0;",
        "  uint32_t source_presence_mask=0;\n  bool shared_epsi_state_independent=false; // actual prepared psi/profile byte ranges",
    )
    host = e.decode()
    host = once(
        host,
        "const bool BeamzResearchSourcePendingEnabled=true;",
        "const bool BeamzResearchSourcePendingEnabled=true;\nconst bool BeamzResearchSharedEPsiEnabled=true;",
    )
    host = once(
        host,
        '#include "native_moving.h"',
        '#include "native_moving.h"\n#include "runtime_shared_epsi_state_fact.h"',
    )
    host = once(
        host,
        "    BeamzNativeDeepPlan observation;",
        """    BeamzBuffer cache_state[12]{},cache_profile[36]{};
    for(int family=0;family<2;++family){const auto&v=family?validate.program.e_ab:validate.program.h_ab;
      for(int term=0;term<6;++term){cache_state[6*family+term]=v.inputs[31+term];
        for(int kind=0;kind<3;++kind)cache_profile[18*family+3*term+kind]=v.inputs[13+3*term+kind];}}
    moving_workspace_.shared_epsi_state_independent=BeamzNativeIndependentPhaseState(cache_state,cache_profile);
    BeamzNativeDeepPlan observation;""",
    )
    host = once(
        host,
        "    moving_uniform_bulk_=certified_bulk;",
        """    moving_shared_epsi_axes_={0,0,0};moving_shared_epsi_active_=0;moving_shared_epsi_retained_=0;
    moving_shared_epsi_witness_={-1,-1,-1,-1,-1,-1,-1,-1};
    // Actual CPML support intersection; witness survives one further update.
    if(moving_workspace_.shared_epsi_state_independent&&(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH)){
      for(size_t rank=0;rank<moving_plan_.uniform_noio_counts.size();++rank){
        const int begin=moving_plan_.policy_offsets[7*rank+1],count=moving_plan_.uniform_noio_counts[rank];
        for(int i=0;i<count;++i){const int tile=moving_plan_.tile_ids[begin+i];const unsigned mask=moving_plan_.tile_flags[tile]&14u;
          if(mask!=2&&mask!=4&&mask!=8)continue;const int axis=mask==2?0:mask==4?1:2;++moving_shared_epsi_axes_[axis];
          const int q[3]={tile/(moving_plan_.counts[1]*moving_plan_.counts[2]),(tile/moving_plan_.counts[2])%moving_plan_.counts[1],tile%moving_plan_.counts[2]};
          bool active=false,retained=false;
          for(int c=0;c<3;++c){if(axes[2*c]!=axis&&axes[2*c+1]!=axis)continue;const auto&f=base_inputs_[3+c];
            for(int side=0;side<2;++side){int64_t first=1,last=depth,keep=depth-1,lower[3]{},upper[3]{},origin[3]{};
              for(int a=0;a<3;++a){origin[a]=int64_t(q[a])*moving_plan_.owned[a];upper[a]=f.dims[a];
                if(a==axis){if(side)lower[a]=f.dims[a]-12;else upper[a]=12;}
                first=std::max<int64_t>(first,origin[a]-upper[a]+1);
                last=std::min<int64_t>(last,origin[a]+moving_plan_.owned[a]-lower[a]-1);
                keep=std::min<int64_t>(keep,origin[a]+moving_plan_.owned[a]-lower[a]-2);}
              active|=first<=last;
              if(first<=keep){retained=true;if(moving_shared_epsi_witness_[0]<0){const int term=axes[2*c]==axis?2*c:2*c+1;
                moving_shared_epsi_witness_={tile,int(first+1),6+term,int(std::max(lower[0],origin[0]-first)),int(std::max(lower[1],origin[1]-first)),int(std::max(lower[2],origin[2]-first)),axis,int(first)};}}
            }}
          moving_shared_epsi_active_+=active;moving_shared_epsi_retained_+=retained;
        }
      }
    }
    moving_uniform_bulk_=certified_bulk;""",
    )
    host = once(
        host,
        "    const bool source_selector=",
        "    const bool shared_selector=BeamzResearchSharedEPsiEnabled;append_bytes(&shared_selector,sizeof(bool));append_bytes(&moving_workspace_.shared_epsi_state_independent,sizeof(bool));\n    const bool source_selector=",
    )
    host = once(
        host,
        '    result["moving_source_presence_mask"]=',
        """    result["moving_shared_epsi_selector_enabled"]=BeamzResearchSharedEPsiEnabled;
    result["moving_shared_epsi_state_independent"]=moving_workspace_.shared_epsi_state_independent;
    result["moving_shared_epsi_dispatch_active"]=BeamzResearchSharedEPsiEnabled&&moving_workspace_.shared_epsi_state_independent&&moving_uniform_noio_>0;
    result["moving_shared_epsi_single_axis_tiles_zyx"]=nb::make_tuple(moving_shared_epsi_axes_[0],moving_shared_epsi_axes_[1],moving_shared_epsi_axes_[2]);
    result["moving_shared_epsi_active_tiles"]=moving_shared_epsi_active_;result["moving_shared_epsi_retained_tiles"]=moving_shared_epsi_retained_;
    result["moving_shared_epsi_retained_witness"]=nb::make_tuple(moving_shared_epsi_witness_[0],moving_shared_epsi_witness_[1],moving_shared_epsi_witness_[2],moving_shared_epsi_witness_[3],moving_shared_epsi_witness_[4],moving_shared_epsi_witness_[5],moving_shared_epsi_witness_[6],moving_shared_epsi_witness_[7]);
    result["moving_shared_epsi_bytes"]=84516;result["moving_shared_epsi_equal_profiles_required"]=false;
    result["moving_source_presence_mask"]=""",
    )
    host = once(
        host,
        "  size_t moving_uniform_bulk_=0,",
        "  std::array<size_t,3>moving_shared_epsi_axes_{};size_t moving_shared_epsi_active_=0,moving_shared_epsi_retained_=0;\n  std::array<int,8>moving_shared_epsi_witness_{{-1,-1,-1,-1,-1,-1,-1,-1}};\n  size_t moving_uniform_bulk_=0,",
    )
    host = once(
        host,
        "NB_MODULE(_cuda_native, module) {",
        """NB_MODULE(_cuda_native, module) {
  module.attr("moving_shared_epsi")=true;module.attr("moving_shared_epsi_canonical_cell")=true;
  module.attr("moving_shared_epsi_equal_profiles_required")=false;module.attr("moving_shared_epsi_shared_bytes")=84516;
  module.attr("moving_shared_epsi_selector_enabled")=BeamzResearchSharedEPsiEnabled;""",
    )
    inventory_start = host.index("    moving_shared_epsi_axes_={0,0,0};")
    inventory_end = host.index(
        "    moving_uniform_bulk_=certified_bulk;", inventory_start
    )
    inventory = host[inventory_start:inventory_end]
    host = host[:inventory_start] + host[inventory_end:]
    host = once(
        host,
        "    BeamzNativeMovingSparseObservation sparse;",
        inventory + "    BeamzNativeMovingSparseObservation sparse;",
    )
    if debug_hits:
        host = once(
            host,
            '      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");',
            '      moving_shared_epsi_debug_raw_=0;\n      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");',
        )
        host = once(
            host,
            '      if(bad)throw std::runtime_error("native moving producer/staging diagnostic failed: "+std::to_string(bad));',
            '      moving_shared_epsi_debug_raw_=bad;\n      if(bad&~56)throw std::runtime_error("native moving producer/staging diagnostic failed: "+std::to_string(bad));',
        )
        host = once(
            host,
            '    result["moving_shared_epsi_selector_enabled"]=',
            '    result["moving_shared_epsi_debug_raw"]=moving_shared_epsi_debug_raw_;result["moving_shared_epsi_debug_real_errors"]=moving_shared_epsi_debug_raw_&~56;\n    result["moving_shared_epsi_debug_hits_zyx"]=nb::make_tuple(bool(moving_shared_epsi_debug_raw_&8),bool(moving_shared_epsi_debug_raw_&16),bool(moving_shared_epsi_debug_raw_&32));\n    result["moving_shared_epsi_validation_only"]=true;\n    result["moving_shared_epsi_selector_enabled"]=',
        )
        host = once(
            host,
            "  std::array<size_t,3>moving_shared_epsi_axes_{};",
            "  int32_t moving_shared_epsi_debug_raw_=0;\n  std::array<size_t,3>moving_shared_epsi_axes_{};",
        )
        host = once(
            host,
            '  module.attr("moving_shared_epsi")=true;',
            '  module.attr("moving_shared_epsi_validation_only")=true;\n  module.attr("moving_shared_epsi")=true;',
        )
    return text.encode(), host.encode(), header.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--debug-cache-hits",
        action="store_true",
        help="Validation only; actual retained reads mark diagnostic bits8/16/32; never time",
    )
    a = p.parse_args()
    parent, output = a.parent.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        output.exists()
        or output == canonical
        or canonical in output.parents
        or parent == output
        or parent in output.parents
        or output in parent.parents
    ):
        p.error("New disjoint private output required")
    src = parent / "src"
    cell = (src / "asymmetric_native_cell_generated.cuh").read_bytes()
    if sha(cell) != CELL:
        p.error("Canonical cell changed")
    c, e, h = transform(
        (src / "asymmetric_native.cuh").read_bytes(),
        (src / "native_extension.cc").read_bytes(),
        (src / "native_moving.h").read_bytes(),
        debug_hits=a.debug_cache_hits,
    )
    shutil.copytree(parent, output)
    for name, data in [
        ("asymmetric_native.cuh", c),
        ("native_extension.cc", e),
        ("native_moving.h", h),
    ]:
        (output / "src" / name).write_bytes(data)
    guard = generate_guard()
    (output / "src/runtime_shared_epsi_state_fact.h").write_text(guard["source"])
    proof = {
        "parent_consumer_sha256": PARENT,
        "parent_extension_sha256": HOST,
        "canonical_cell_sha256": CELL,
        "consumer_sha256": sha(c),
        "extension_sha256": sha(e),
        "workspace_sha256": sha(h),
        "state_guard_sha256": guard["source_sha256"],
        "point_adapter_sha256": sha(POINT.encode()),
        "validation_only_cache_hits": a.debug_cache_hits,
        "arithmetic": "Canonical scalar header unchanged; no joint/profile substitution; unequal profiles supported",
        "mapping": "T1024,Slots4,ZStep2,zbase=tid/512; bank[(color*4+j)*1024+tid]",
        "admission": "existing certified bytepacked/isotropic NoIO; actual prepared state/profile ranges disjoint; onehot CPML2/4/8 cache, otherwise global",
        "storage_bytes": {"old": 51748, "source": 58680, "shared_epsi": 84516},
        "expected_kernels": 60,
        "scope": "Full runtime; no resource/GPU pass implied",
    }
    (output.parent / "shared-epsi-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
