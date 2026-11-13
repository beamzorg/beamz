"""Certified runtime source-plane topology; exact v12 fallback everywhere else."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_v12_coincident_source_geometry import ORIGINAL
    from scripts.patch_cuda_v12_noio_axis_bundle import PARENT_HASHES, PARENT_MODULE
except ModuleNotFoundError:
    from patch_cuda_v12_coincident_source_geometry import ORIGINAL
    from patch_cuda_v12_noio_axis_bundle import PARENT_HASHES, PARENT_MODULE

PLANNER_SHA256 = "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1"

PLANE = """    if constexpr(PlanePair){
     // Captured topology and CPML-free Policy3/4 prove transverse bounds.
     if(x==s.starts[2]){
      int time=*s.current_step+p.macro_step+version-1;time=max(0,min(time,s.ntime-1));
      const int offset=(z-12)*s.ny+(y-12);
      value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+offset),__ldg(s.waveforms+time)));
      value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+s.nz*s.ny+offset),__ldg(s.waveforms+s.ntime+time)));
     }
    }else{
""" + ORIGINAL + "\n    }"

FACT = r'''
// Program-wide topology of privately captured/uploaded groups, not their FP values.
inline bool BeamzNativeSourcePlanePairFact(const BeamzSourceGroupLaunch*groups,int count,
 const int32_t*const*starts,const int*minimum,uint32_t presence){
 if(!groups||!starts||!minimum||count!=9||presence!=20||minimum[0]<=24||minimum[1]<=24)return false;
 for(int g:{5,7}){
  const auto&c=groups[g].coefficients;const auto*s=starts[g];
  if(c.rank!=4||c.element_type!=kBeamzF32||c.dims[0]!=2||c.dims[1]!=minimum[0]-24||
     c.dims[2]!=minimum[1]-24||c.dims[3]!=1||!s)return false;
  if(s[0]!=12||s[1]!=12||s[3]!=12||s[4]!=12||s[2]!=s[5]||s[2]<0||s[2]>=minimum[2])return false;
 }
 return true;
}
'''
ENTRY = """template<int Depth,int Policy>
__global__ __launch_bounds__(1024,1) void AsymmetricSourcePlanePair20(
 const __grid_constant__ Parameters p,const int32_t*ids){
 static_assert(Policy==3||Policy==4,"Source-plane transverse support requires CPML-free source policies");
 SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20,true>(p,ids);
}

"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one source-plane anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for n, h in PARENT_HASHES.items():
        if sha(parent[n]) != h:
            raise ValueError(f"Exact v12 a939 parent required: {n}")
    c, e, h = (parent[n].decode() for n in ("asymmetric_native.cuh", "native_extension.cc", "native_moving.h"))
    edits = [
        ("int SourceMask=-1> struct Environment", "int SourceMask=-1,bool PlanePair=false> struct Environment"),
        (" static constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);",
         " static_assert(!PlanePair||(SourceMask==20&&(Policy==3||Policy==4)),\"Plane source requires exact CPML-free class\");\n static constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);"),
        (ORIGINAL, PLANE),
        ("bool UniformE,int BlockT=0,int SourceMask=-1>\n__device__ __forceinline__ void SourcePendingPairedBody",
         "bool UniformE,int BlockT=0,int SourceMask=-1,bool PlanePair=false>\n__device__ __forceinline__ void SourcePendingPairedBody"),
        (" Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask>env{p,flags};",
         " Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair>env{p,flags};"),
        ("inline bool MakeSource(", ENTRY + "inline bool MakeSource("),
        ("int count,uint32_t presence){", "int count,uint32_t presence,bool plane_pair){"),
        ("   const auto kernel=AsymmetricSourcePending20<Depth,Policy>;",
         """   if constexpr(Policy==3||Policy==4){
    if(BeamzResearchSourcePlanePairEnabled&&plane_pair){
     const auto specialized=AsymmetricSourcePlanePair20<Depth,Policy>;
     auto status=cudaFuncSetAttribute(specialized,cudaFuncAttributeMaxDynamicSharedMemorySize,SourcePendingSharedBytes);if(status!=cudaSuccess)return status;
     AsymmetricSourcePlanePair20<Depth,Policy><<<count,1024,SourcePendingSharedBytes,stream>>>(p,ids);
     return cudaGetLastError();
    }
   }
   const auto kernel=AsymmetricSourcePending20<Depth,Policy>;"""),
    ]
    for policy in (3, 4, 5):
        old = f"<Depth,Layout,IsotropicH,{policy}>(branch,p,w.tile_ids_device+start,count,w.source_presence_mask);break;"
        edits.append((old, old.replace("w.source_presence_mask)", "w.source_presence_mask,bool(w.data_flags&8u))")))
    old = "<Depth,Layout,IsotropicH,5>(stream,p,w.tile_ids_device+start+bulk,count-bulk,w.source_presence_mask);"
    edits.append((old, old.replace("w.source_presence_mask)", "w.source_presence_mask,bool(w.data_flags&8u))")))
    out = c
    for old, new in edits:
        out = once(out, old, new)
    inverse = out
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    if inverse != c:
        raise ValueError("Source-plane consumer inverse failed")
    # The entire paired storage operator itself changes only its template/Env
    # binding, never an instruction, barrier, read callback or publication.
    host_edits = [
        ("const bool BeamzResearchSourcePendingEnabled=true;", "const bool BeamzResearchSourcePendingEnabled=true;\nconst bool BeamzResearchSourcePlanePairEnabled=true;"),
        ("    moving_workspace_.source_presence_mask=BeamzNativeMovingSourcePresenceMask(validate.program.source_groups,validate.program.source_group_count);",
         """    moving_workspace_.source_presence_mask=BeamzNativeMovingSourcePresenceMask(validate.program.source_groups,validate.program.source_group_count);
    const int32_t*plane_starts[9]{};for(int g=0;g<9;++g)plane_starts[g]=static_cast<const int32_t*>(host_[source_offset_+3*g+2].data);
    if(BeamzNativeSourcePlanePairFact(validate.program.source_groups,validate.program.source_group_count,plane_starts,minimum,moving_workspace_.source_presence_mask))moving_workspace_.data_flags|=8u;"""),
        ('    result["moving_source_mask20_dispatch_active"]=BeamzResearchSourcePendingEnabled&&source_class&&(source_tiles[0]+source_tiles[1]+source_tiles[2]>0);',
         '''    result["moving_source_plane_pair_class"]=bool(moving_workspace_.data_flags&8u);
    result["moving_source_plane_pair_selector_enabled"]=BeamzResearchSourcePlanePairEnabled;
    result["moving_source_plane_pair_policy_tiles"]=nb::make_tuple(source_tiles[0],source_tiles[1]);
    result["moving_source_plane_pair_dispatch_active"]=BeamzResearchSourcePendingEnabled&&BeamzResearchSourcePlanePairEnabled&&source_class&&bool(moving_workspace_.data_flags&8u)&&(source_tiles[0]+source_tiles[1]>0);
    result["moving_source_mask20_dispatch_active"]=BeamzResearchSourcePendingEnabled&&source_class&&(source_tiles[0]+source_tiles[1]+source_tiles[2]>0);'''),
        ("    const bool source_selector=BeamzResearchSourcePendingEnabled;append_bytes(&source_selector,sizeof(bool));",
         "    const bool source_selector=BeamzResearchSourcePendingEnabled;append_bytes(&source_selector,sizeof(bool));\n    const bool plane_selector=BeamzResearchSourcePlanePairEnabled;append_bytes(&plane_selector,sizeof(bool));"),
        ('  module.attr("moving_source_pending_mask20_1024")=true;',
         '  module.attr("moving_source_plane_pair")=true;module.attr("moving_source_plane_pair_selector_enabled")=BeamzResearchSourcePlanePairEnabled;\n  module.attr("moving_source_pending_mask20_1024")=true;'),
    ]
    host = e
    for old, new in host_edits:
        host = once(host, old, new)
    invhost = host
    for old, new in reversed(host_edits):
        invhost = once(invhost, new, old)
    if invhost != e:
        raise ValueError("Source-plane host inverse failed")
    header = once(h, "struct BeamzNativeMovingWorkspace {", FACT + '\nextern const bool BeamzResearchSourcePlanePairEnabled __attribute__((visibility("hidden")));\n\nstruct BeamzNativeMovingWorkspace {')
    files = dict(parent)
    files["asymmetric_native.cuh"], files["native_extension.cc"], files["native_moving.h"] = out.encode(), host.encode(), header.encode()
    return files


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    if sha((source / "src/native_moving.cc").read_bytes()) != PLANNER_SHA256:
        raise ValueError("Actual planner trajectory flag formula changed")
    parent = {n: (source / "src" / n).read_bytes() for n in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for n, data in files.items():
        (output / "src" / n).write_bytes(data)
    proof = {"parent_module_sha256": PARENT_MODULE, "parent_hashes": PARENT_HASHES,
             "planner_sha256": PLANNER_SHA256,
             "result_hashes": {n: sha(d) for n, d in files.items()}, "patcher_sha256": sha(Path(__file__).read_bytes()),
             "exact_consumer_host_inverse": True, "canonical_cell_unchanged": True,
             "paired_storage_body_changes": "template and Env binding only; defaultfalse preserves original schedule/math",
             "admission": "privately captured startsZY12, equalX in-range; groups5/7 actualcount2 shape[2,Nz-24,Ny-24,1]; presence20; old layout/isotropic/source guards; onlyPolicy3/4",
             "bounds_proof": "Planner flags axisbits0 implies origin-depth>=12 and origin+owned-2<minimum-12 for allaxes; every owned update lies in[12,N-13]",
             "runtime_values": "device startsX/currentstep/waveforms/coefficients remain runtime; no resolution-dependent value baking",
             "program_fact_bit": 8, "graph_key": "existingdata_flags plusnewimmutablehiddenplane selector",
             "new_kernel_entries": 4, "expected_moving_kernels": 62,
             "required_gates": ["actualhostfactandcallbackCPUoracles", "allzero-localresource", "strictwholephysicsandcausalOnOffgates"], "gpu_executed": False}
    (output.parent / "source-plane-pair-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output)))


if __name__ == "__main__":
    main()
