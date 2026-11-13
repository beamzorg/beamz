"""Add mask10/Z-normal source classes to the frozen52865 CUDA kernel family.

Existing mask20 source kernels and fallback bodies remain byte-identical. The
new classes consume already cyclically transformed native inputs, and retain
the same ordered arithmetic, field storage, imports and synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PARENT = {
    "asymmetric_native.cuh": "f082980ad0687ef165c7807b415e87cbc1003edfdbd2acf7a4ee018dae4d4547",
    "native_moving.h": "d3d1abaca2ca066b75295df35ec26da2c1df53c75bdb9f140b067fea09288b3c",
    "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27",
    "native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"One exact cyclic-source anchor required: {old!r}")
    return text.replace(old, new, 1)


FACT10 = r'''
// Same privately captured topology certificate after the proper XYZ cycle.
inline bool BeamzNativeSourcePlanePairFact10(const BeamzSourceGroupLaunch*groups,int count,
 const int32_t*const*starts,const int*minimum,uint32_t presence){
 if(!groups||!starts||!minimum||count!=9||presence!=10||minimum[1]<=24||minimum[2]<=24)return false;
 for(int g:{4,6}){
  const auto&c=groups[g].coefficients;const auto*s=starts[g];
  if(c.rank!=4||c.element_type!=kBeamzF32||c.dims[0]!=2||c.dims[1]!=1||
     c.dims[2]!=minimum[1]-24||c.dims[3]!=minimum[2]-24||!s)return false;
  if(s[1]!=12||s[2]!=12||s[4]!=12||s[5]!=12||s[0]!=s[3]||s[0]<0||s[0]>=minimum[0])return false;
 }
 return true;
}
'''

PLANE10 = r'''     if constexpr(SourceMask==10){
      if(z==s.starts[0]){
       int time=*s.current_step+p.macro_step+version-1;time=max(0,min(time,s.ntime-1));
       const int offset=(y-12)*s.nx+(x-12);
       value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+offset),__ldg(s.waveforms+time)));
       value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+s.ny*s.nx+offset),__ldg(s.waveforms+s.ntime+time)));
      }
     }else{
'''

ENTRY10 = r'''
template<int Depth,int Policy>
__global__ __launch_bounds__(1024,1) void AsymmetricSourcePending10(
 const __grid_constant__ Parameters p,const int32_t*ids){
 static_assert(Policy>=3&&Policy<=5,"Only source-bearing policies");
 SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,10>(p,ids);
}
template<int Depth,int Policy>
__global__ __launch_bounds__(1024,1) void AsymmetricSourcePlanePair10(
 const __grid_constant__ Parameters p,const int32_t*ids){
 static_assert(Policy==3||Policy==4,"Source-plane transverse support requires CPML-free source policies");
 SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,10,true>(p,ids);
}

'''

DISPATCH10 = r'''   if(presence==10u){
    if constexpr(Policy==3||Policy==4){
     if(BeamzResearchSourcePlanePairEnabled&&plane_pair){
      const auto specialized=AsymmetricSourcePlanePair10<Depth,Policy>;
      auto status=cudaFuncSetAttribute(specialized,cudaFuncAttributeMaxDynamicSharedMemorySize,SourcePendingSharedBytes);if(status!=cudaSuccess)return status;
      AsymmetricSourcePlanePair10<Depth,Policy><<<count,1024,SourcePendingSharedBytes,stream>>>(p,ids);
      return cudaGetLastError();
     }
    }
    const auto kernel=AsymmetricSourcePending10<Depth,Policy>;
    auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SourcePendingSharedBytes);if(status!=cudaSuccess)return status;
    AsymmetricSourcePending10<Depth,Policy><<<count,1024,SourcePendingSharedBytes,stream>>>(p,ids);
    return cudaGetLastError();
   }
'''


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError(f"Exact frozen52865 parent required: {name}")
    result = dict(files)
    c = files["asymmetric_native.cuh"].decode()
    c = once(c, "SourceMask==20&&(Policy==3||Policy==4)", "(SourceMask==20||SourceMask==10)&&(Policy==3||Policy==4)")
    c = once(c, "     if(x==s.starts[2]){", PLANE10 + "     if(x==s.starts[2]){")
    c = once(c, "    }else{\n    for(int source=0;source<s.count;", "     }\n    }else{\n    for(int source=0;source<s.count;")
    c = once(c, "inline bool MakeSource(", ENTRY10 + "inline bool MakeSource(")
    c = once(c, "  if(BeamzResearchSourcePendingEnabled&&BeamzNativeMovingSourceMask20Class(Layout,IsotropicH,presence)){", "  if(BeamzResearchSourcePendingEnabled&&BeamzNativeMovingSourceMask20Class(Layout,IsotropicH,presence)){\n" + DISPATCH10)
    result["asymmetric_native.cuh"] = c.encode()
    h = files["native_moving.h"].decode()
    h = once(h, " return layout==1&&isotropic&&presence==20u;", " return layout==1&&isotropic&&(presence==20u||presence==10u);")
    h = once(h, "// CUDA consumer owns the private epoch increment", FACT10 + "\n// CUDA consumer owns the private epoch increment")
    result["native_moving.h"] = h.encode()
    e = files["native_extension.cc"].decode()
    old = "if(BeamzNativeSourcePlanePairFact(validate.program.source_groups,validate.program.source_group_count,plane_starts,minimum,moving_workspace_.source_presence_mask))moving_workspace_.data_flags|=8u;"
    new = old.replace("))moving_workspace_", ")||BeamzNativeSourcePlanePairFact10(validate.program.source_groups,validate.program.source_group_count,plane_starts,minimum,moving_workspace_.source_presence_mask))moving_workspace_")
    e = once(e, old, new)
    # Keep existing diagnostics so the complete unrotated regression suite can
    # validate this extension as well. The new label disambiguates mask10.
    diagnostic = '    result["moving_source_mask20_policy_tiles"]=nb::make_tuple(source_tiles[0],source_tiles[1],source_tiles[2]);'
    e = once(e, diagnostic, diagnostic + '\n    result["moving_source_mask10_or20_policy_tiles"]=nb::make_tuple(source_tiles[0],source_tiles[1],source_tiles[2]);')
    result["native_extension.cc"] = e.encode()
    return result


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source directory required")
    before = {n: (source / "src" / n).read_bytes() for n in PARENT}
    after = transformed(before)
    shutil.copytree(source, output)
    for n, value in after.items():
        (output / "src" / n).write_bytes(value)
    report = {"schema": "beamz.cyclic-source-lowering.v1", "parent_module": "52865a3ac8371e2390749953eb3af99746df108fe7bcb00fc4be8b35f4259b92",
        "parent_hashes": PARENT, "result_hashes": {n: sha(v) for n, v in after.items()},
        "source_normal_oldX_to_internalZ": True, "existing_source20_kernels_retained": True,
        "new_source_mask": 10, "new_groups": [4, 6], "new_moving_kernel_count": 72,
        "ordered_scalar_graph_and_paired_storage_unchanged": True,
        "requires": ["exact parent compile mode", "zero spill resource gate", "ordinary Program parity including all30 world-inverse leaves", "positive and negative topology admission", "same-world complete-workload timing"],
        "gpu_executed": False, "producer_sha256": sha(Path(__file__).read_bytes())}
    (output.parent / "cyclic-source-proof.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
