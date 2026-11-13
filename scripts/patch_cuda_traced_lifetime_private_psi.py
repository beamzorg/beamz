"""Cache both H/E single-axis CPML histories in no-field-private frontier.

Sixteen plain constant-index floats replace repeated in-place global psi reads
and stores for certified NoIO trajectories. First-owned points reload; outgoing
high faces and final depth publish. Multi-axis tiles and aliasing state packets
keep the uncached all-shared frontier. All scalar arithmetic/profiles stay in
the canonical generated cell header. No equal-profile assumption is introduced.
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
from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES, once
from scripts.patch_cuda_traced_lifetime_shared_fields import transform as all_shared

HOST_PARENT = "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27"
POINT = r"""
template<bool Enabled,int Slots> struct PrivatePsiHistory {};
template<int Slots> struct PrivatePsiHistory<true,Slots>{float ha[Slots],hb[Slots],ea[Slots],eb[Slots];};
// Each active phase-axis has two independent terms, assigned different colors.
template<class Base> struct PrivatePsiHistoryPoint : Base {
 float &a,&b;bool first,flush;
 __device__ __forceinline__ PrivatePsiHistoryPoint(Base&base,float&pa,float&pb,bool fresh,bool publish)
 :Base(base),a(pa),b(pb),first(fresh),flush(publish){}
 __device__ __forceinline__ float& State(int slot)const{
  const int term=slot%6;return term==2||term==4||term==5?b:a;
 }
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if(first)return Base::LoadLocal(slot,z,y,x);
  return State(slot);
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  State(slot)=value;if(flush)Base::StoreLocal(slot,z,y,x,value);
 }
};
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform(consumer, host, header, *, scope=3, debug_hits=False, enabled=True):
    if sha(host.encode()) != HOST_PARENT or sha(header.encode()) != PARENT_HASHES["native_moving.h"]:
        raise ValueError("Exact selected52865 host/workspace required")
    if not scope & 2:
        raise ValueError("Private psi requires all-shared NoIO scope")
    text, baseline_proof = all_shared(consumer, scope=scope)
    begin = text.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void PermanentFamilyBody")
    end = text.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", begin)
    body = text[begin:end]
    body = once(body, "bool UniformE,int ActiveAxes=-1>", "bool UniformE,int ActiveAxes=-1,bool CachePsi=false>")
    body = once(body,
        " const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/512;",
        " static_assert(!CachePsi||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE&&Slots==4&&(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4)),\"Exact single-axis NoIO private-psi class\");\n"
        " PrivatePsiHistory<CachePsi,Slots>history;\n"
        " const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/512;")
    for phase, prefix, clock_prefix in ((0, "h", ""), (1, "e", "e")):
        hz, hy, hx = (clock_prefix + n for n in ("hz", "hy", "hx"))
        old = f"    if(zs!={hz}&&sy!={hy}&&sx!={hx}){{\n     const auto read="
        new = f"""    if(zs!={hz}&&sy!={hy}&&sx!={hx}){{
     auto point_env=[&] __device__(){{
      if constexpr(CachePsi){{
       const int dz=(zs-{hz})&7,dy=(sy-{hy})&15,dx=(sx-{hx})&31;
       return PrivatePsiHistoryPoint<decltype(env)>(env,history.{prefix}a[j],history.{prefix}b[j],
        n==window[0]||dz==1||dy==1||dx==1,n==Depth||dz==BZ||dy==BY||dx==BX);
      }}else return env;
     }}();
     const auto read="""
        body = once(body, old, new)
        for component in range(3):
            body = once(body,
                f"AdvanceNativeCell<{phase},{component},false>(env,read,n,z,y,x)",
                f"AdvanceNativeCell<{phase},{component},false>(point_env,read,n,z,y,x)")
    point = POINT
    if debug_hits:
        point = once(point, "  return State(slot);",
                     "  atomicOr(this->p.bad,(slot<6?8u:64u)<<beamz::cuda::yee::CpmlAxis(slot%6)); // Retained-state validation only.\n  return State(slot);")
    text = text[:begin] + point + body + text[end:]
    entry = """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOPrivatePsi1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14){
  case 2:PermanentFamilyBody<Depth,1,false,true,1,true,1,true>(p,tiles);return;
  case 4:PermanentFamilyBody<Depth,1,false,true,1,true,2,true>(p,tiles);return;
  case 8:PermanentFamilyBody<Depth,1,false,true,1,true,4,true>(p,tiles);return;
  default:PermanentFamilyBody<Depth,1,false,true,1,true>(p,tiles);return;
 }
}

"""
    text = once(text,
        "template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024",
        entry + "template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024")
    text = once(text,
        "int count,int uniform_count){\n if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;\n if constexpr(Layout==1&&IsotropicH){\n  // Existing policy",
        "int count,int uniform_count,bool private_psi){\n if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;\n if constexpr(Layout==1&&IsotropicH){\n  // Existing policy")
    text = once(text, "  const auto kernel=AsymmetricNoIOUniform1024<Depth>;",
        """  if(private_psi){
   const auto cached=AsymmetricNoIOPrivatePsi1024<Depth>;
   status=cudaFuncSetAttribute(cached,cudaFuncAttributeMaxDynamicSharedMemorySize,PermanentFamilySharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricNoIOPrivatePsi1024<Depth><<<uniform_count,1024,PermanentFamilySharedBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }
  const auto kernel=AsymmetricNoIOUniform1024<Depth>;""")
    text = once(text,
        "count,w.uniform_noio_counts_host[rank]);break;",
        "count,w.uniform_noio_counts_host[rank],BeamzResearchPrivatePsiEnabled&&w.private_psi_state_independent);break;")
    header = once(header, "struct BeamzNativeMovingWorkspace {",
        'extern const bool BeamzResearchPrivatePsiEnabled __attribute__((visibility("hidden")));\n\nstruct BeamzNativeMovingWorkspace {')
    header = once(header, "  uint32_t source_presence_mask=0;",
        "  uint32_t source_presence_mask=0;\n  bool private_psi_state_independent=false;")
    host = once(host, "const bool BeamzResearchSourcePendingEnabled=true;",
        f"const bool BeamzResearchSourcePendingEnabled=true;\nconst bool BeamzResearchPrivatePsiEnabled={'true' if enabled else 'false'};")
    host = once(host, '#include "native_moving.h"', '#include "native_moving.h"\n#include "runtime_private_psi_state_fact.h"')
    host = once(host, "    BeamzNativeDeepPlan observation;",
        """    BeamzBuffer cache_state[12]{},cache_profile[36]{};
    for(int family=0;family<2;++family){const auto&v=family?validate.program.e_ab:validate.program.h_ab;
      for(int term=0;term<6;++term){cache_state[6*family+term]=v.inputs[31+term];
        for(int kind=0;kind<3;++kind)cache_profile[18*family+3*term+kind]=v.inputs[13+3*term+kind];}}
    moving_workspace_.private_psi_state_independent=BeamzNativeIndependentPhaseState(cache_state,cache_profile);
    BeamzNativeDeepPlan observation;""")
    host = once(host, "    BeamzNativeMovingSparseObservation sparse;",
        """    moving_private_psi_axes_={0,0,0};
    for(size_t rank=0;rank<moving_plan_.uniform_noio_counts.size();++rank){
      const int begin=moving_plan_.policy_offsets[7*rank+1],count=moving_plan_.uniform_noio_counts[rank];
      for(int i=0;i<count;++i){const unsigned mask=moving_plan_.tile_flags[moving_plan_.tile_ids[begin+i]]&14u;
        if(mask==2||mask==4||mask==8)++moving_private_psi_axes_[mask==2?0:mask==4?1:2];}}
    BeamzNativeMovingSparseObservation sparse;""")
    host = once(host, "    const bool source_selector=",
        "    const bool private_psi_selector=BeamzResearchPrivatePsiEnabled;append_bytes(&private_psi_selector,sizeof(bool));append_bytes(&moving_workspace_.private_psi_state_independent,sizeof(bool));\n    const bool source_selector=")
    host = once(host, '    result["moving_source_presence_mask"]=',
        """    result["moving_private_psi_selector_enabled"]=BeamzResearchPrivatePsiEnabled;
    result["moving_private_psi_state_independent"]=moving_workspace_.private_psi_state_independent;
    result["moving_private_psi_single_axis_tiles_zyx"]=nb::make_tuple(moving_private_psi_axes_[0],moving_private_psi_axes_[1],moving_private_psi_axes_[2]);
    result["moving_private_psi_floats_per_thread"]=16;result["moving_private_psi_equal_profiles_required"]=false;
    result["moving_source_presence_mask"]=""")
    host = once(host, "  size_t moving_uniform_bulk_=0,", "  std::array<size_t,3>moving_private_psi_axes_{};\n  size_t moving_uniform_bulk_=0,")
    host = once(host, "NB_MODULE(_cuda_native, module) {",
        """NB_MODULE(_cuda_native, module) {
  module.attr("moving_private_psi_history")=true;module.attr("moving_private_psi_floats_per_thread")=16;
  module.attr("moving_private_psi_selector_enabled")=BeamzResearchPrivatePsiEnabled;
  module.attr("moving_private_psi_equal_profiles_required")=false;""")
    if debug_hits:
        host = once(host, '      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");',
                    '      moving_private_psi_debug_=0;\n      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");')
        host = once(host, '      if(bad)throw std::runtime_error("native moving producer/staging diagnostic failed: "+std::to_string(bad));',
                    '      moving_private_psi_debug_=bad;\n      if(bad&~504)throw std::runtime_error("native moving producer/staging diagnostic failed: "+std::to_string(bad));')
        host = once(host, '    result["moving_private_psi_selector_enabled"]=',
                    '    result["moving_private_psi_debug_raw"]=moving_private_psi_debug_;result["moving_private_psi_validation_only"]=true;\n'
                    '    result["moving_private_psi_debug_H_zyx"]=nb::make_tuple(bool(moving_private_psi_debug_&8),bool(moving_private_psi_debug_&16),bool(moving_private_psi_debug_&32));\n'
                    '    result["moving_private_psi_debug_E_zyx"]=nb::make_tuple(bool(moving_private_psi_debug_&64),bool(moving_private_psi_debug_&128),bool(moving_private_psi_debug_&256));\n'
                    '    result["moving_private_psi_selector_enabled"]=')
        host = once(host, "  std::array<size_t,3>moving_private_psi_axes_{};", "  int32_t moving_private_psi_debug_=0;\n  std::array<size_t,3>moving_private_psi_axes_{};")
        host = once(host, '  module.attr("moving_private_psi_history")=true;', '  module.attr("moving_private_psi_validation_only")=true;\n  module.attr("moving_private_psi_history")=true;')
    return text, host, header, {"all_shared_baseline": baseline_proof, "scope_bits": scope,
        "private_psi_floats_per_thread": 16, "shared_bytes": 100904,
        "H_E_each_two_colors_times_four_point_slots": True,
        "first_owned_reload": "n==window[0]||dz==1||dy==1||dx==1",
        "global_publish": "n==Depth||dz==BZ||dy==BY||dx==BX",
        "independent_state_guard": generate_guard()["source_sha256"],
        "selector_enabled": enabled, "debug_hits": debug_hits,
        "same_scalar_cell_header_profiles_and_arithmetic": True,
        "host_key_captures_selector_and_actual_state_fact": True}


def patch_copy(source, output, *, scope=3, debug_hits=False, enabled=True):
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
    consumer, host, header, proof = transform((src / "asymmetric_native.cuh").read_text(),
        (src / "native_extension.cc").read_text(), (src / "native_moving.h").read_text(),
        scope=scope, debug_hits=debug_hits, enabled=enabled)
    guard = generate_guard()
    files = {"asymmetric_native.cuh": consumer, "native_extension.cc": host,
             "native_moving.h": header, "runtime_private_psi_state_fact.h": guard["source"]}
    proof.update({"parent_hashes": parents, "result_hashes": {name: sha(text.encode()) for name, text in files.items()},
                  "producer_sha256": sha(Path(__file__).read_bytes()), "gpu_executed": False,
                  "requires": ["canonical owner/packed-slab recurrence proof forowned7x15x31",
                               "zero stack/local/spill64-kernel exact-context resource gate",
                               "all30 exact outputs plus unequalH/Eprofiles andcontinuation",
                               "actual retainedH/Ehit diagnostics", "memcheck/synccheck/racecheck", "matched full workload timing"]})
    shutil.copytree(source, output)
    for name, text in files.items():
        (output / "src" / name).write_text(text)
    (output.parent / "private-psi-history-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=int, default=3)
    parser.add_argument("--debug-hits", action="store_true")
    parser.add_argument("--selector-off", action="store_true")
    args = parser.parse_args()
    print(json.dumps(patch_copy(args.source, args.output, scope=args.scope,
                               debug_hits=args.debug_hits, enabled=not args.selector_off)))


if __name__ == "__main__":
    main()
