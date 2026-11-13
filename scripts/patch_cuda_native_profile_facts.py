"""Bind verified private CPML upload profiles to an admitted native IR kernel.

Only a copied compact-NoIO consumer is accepted. A separate kernel uses readonly
same-phase/axis profile representatives; all mutable psi IDs remain distinct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_profile_fact import generate_guard
from scripts.generate_cuda_runtime_fact_cell import generate
from scripts.patch_cuda_native_runtime_fact_cell import CANONICAL_CELL, once, patch

PARENT = "a111536eba602a638aeec536ae58af5728f64af8bad0609c7b64e4f672369c62"
HOST_PARENT = "0f812bcf1e1fa8fdb70e3f1e15b20dc921612d85e19f0682be091035f75e97f5"
MOVING_PARENT = "657ac01e5bc3acf02133bafa601f5cf4ea279ad501a6abdddb61e322ea470f8b"


def patch_consumer(source):
    if hashlib.sha256(source.encode()).hexdigest() != PARENT:
        raise ValueError("Exact compact-NoIO source required")
    source = patch(source)
    source = once(source, '#include "asymmetric_native_cell_runtime_generated.cuh"',
                  '#include "asymmetric_native_cell_runtime_generated.cuh"\n#include "asymmetric_native_cell_runtime_profiles_generated.cuh"')
    source = once(source, "bool UseRuntimeFactCell,class Env,class Read>",
                  "bool UseRuntimeFactCell,bool AliasProfiles,class Env,class Read>")
    source = once(source, " if constexpr(UseRuntimeFactCell)return beamz::cuda::trace_native_runtime::AdvanceNativeCell",
                  " if constexpr(AliasProfiles)return beamz::cuda::trace_native_runtime_profiles::AdvanceNativeCell<Phase,Component,false>(env,read,n,z,y,x);\n else if constexpr(UseRuntimeFactCell)return beamz::cuda::trace_native_runtime::AdvanceNativeCell")
    source = once(source, "bool UseRuntimeFactCell=false>\n__device__", "bool UseRuntimeFactCell=false,bool AliasProfiles=false>\n__device__")
    source = once(source, ' static_assert(!UseRuntimeFactCell||', ' static_assert(!AliasProfiles||UseRuntimeFactCell,"Profile alias requires runtime fact cell");\n static_assert(!UseRuntimeFactCell||')
    for phase in (0, 1):
        for c in range(3):
            source = once(source, f"AdvanceRuntimeSelected<{phase},{c},UseRuntimeFactCell>",
                          f"AdvanceRuntimeSelected<{phase},{c},UseRuntimeFactCell,AliasProfiles>")
    old = " UniformBody<Depth,1,false,true,1,true,true>(p,tiles);\n}"
    source = once(source, old, old + """
template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOProfileAlias1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 UniformBody<Depth,1,false,true,1,true,true,true>(p,tiles);
}
""")
    source = once(source, "int count,int uniform_count){", "int count,int uniform_count,bool profile_alias){")
    source = once(source, "  const auto kernel=AsymmetricNoIOUniform1024<Depth>;", """  if(profile_alias){
   const auto kernel=AsymmetricNoIOProfileAlias1024<Depth>;
   status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricNoIOProfileAlias1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }
  const auto kernel=AsymmetricNoIOUniform1024<Depth>;""")
    return once(source, "count,w.uniform_noio_counts_host[rank]);break;",
                "count,w.uniform_noio_counts_host[rank],bool(w.data_flags&beamz::cuda::moving::kEqualAxisProfiles));break;")


def patch_host(source):
    if hashlib.sha256(source.encode()).hexdigest() != HOST_PARENT:
        raise ValueError("Exact compact-NoIO host source required")
    source = once(source, '#include "native_moving.h"', '#include "native_moving.h"\n#include "runtime_profile_fact_guard.h"')
    source = once(source, "((i>=6&&i<12)||(i>=37&&i<43)||", "((i>=6&&i<12)||(i>=37&&i<43)||(i>=13&&i<31)||(i>=44&&i<62)||")
    source = once(source, "    BeamzNativeDeepPlan observation;", """    const void*profile_host[62]{};
    for(int i=13;i<31;++i)profile_host[i]=host_[i].data;
    for(int i=44;i<62;++i)profile_host[i]=host_[i].data;
    moving_equal_axis_profiles_=BeamzNativeEqualAxisProfileFact(base_inputs_.data(),base_inputs_.size(),profile_host);
    if(moving_equal_axis_profiles_)moving_workspace_.data_flags|=beamz::cuda::moving::kEqualAxisProfiles;
    BeamzNativeDeepPlan observation;""")
    source = once(source, "    moving_uniform_bulk_=certified_bulk;", "    moving_profile_alias_noio_tiles_=moving_equal_axis_profiles_?certified_noio:0;\n    moving_uniform_bulk_=certified_bulk;")
    source = once(source, '    result["moving_uniform_noio_tiles"]',
                  '    result["moving_equal_axis_profiles"]=moving_equal_axis_profiles_;result["moving_profile_alias_noio_tiles"]=moving_profile_alias_noio_tiles_;\n    result["moving_uniform_noio_tiles"]')
    source = once(source, "  size_t moving_uniform_bulk_=0,", "  bool moving_equal_axis_profiles_=false;\n  size_t moving_profile_alias_noio_tiles_=0;\n  size_t moving_uniform_bulk_=0,")
    return once(source, "NB_MODULE(_cuda_native, module) {", 'NB_MODULE(_cuda_native, module) {\n  module.attr("moving_runtime_profile_alias")=true;')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("Only private copied source may be patched")
    src = args.source / "src"
    if hashlib.sha256((src / "asymmetric_native_cell_generated.cuh").read_bytes()).hexdigest() != CANONICAL_CELL:
        raise ValueError("Canonical physics header changed")
    consumer = src / "asymmetric_native.cuh"
    host = src / "native_extension.cc"
    parent = consumer.read_text()
    result = patch_consumer(parent)
    host_parent = host.read_text()
    host_result = patch_host(host_parent)
    moving = src / "native_moving.h"
    if hashlib.sha256(moving.read_bytes()).hexdigest() != MOVING_PARENT:
        raise ValueError("Exact compact-NoIO moving ABI required")
    moving_result = once(moving.read_text(), "kBlockPackedE=4;", "kBlockPackedE=4,kEqualAxisProfiles=8;")
    scalar, aliased, guard = generate(), generate(profiles=True), generate_guard()
    (src / "asymmetric_native_cell_runtime_generated.cuh").write_text(scalar["source"])
    (src / "asymmetric_native_cell_runtime_profiles_generated.cuh").write_text(aliased["source"])
    (src / "runtime_profile_fact_guard.h").write_text(guard["source"])
    consumer.write_text(result)
    host.write_text(host_result)
    moving.write_text(moving_result)
    proof = {"schema": "beamz.private-native-runtime-profile-facts.v1", "parent_consumer_sha256": PARENT,
             "consumer_sha256": hashlib.sha256(result.encode()).hexdigest(),
             "host_parent_sha256": hashlib.sha256(host_parent.encode()).hexdigest(),
             "host_sha256": hashlib.sha256(host_result.encode()).hexdigest(),
             "canonical_cell_sha256": CANONICAL_CELL,
             "guard": {k: v for k, v in guard.items() if k != "source"},
             "scalar_ir": {k: v for k, v in scalar.items() if k != "source"},
             "alias_ir": {k: v for k, v in aliased.items() if k != "source"},
             "upload_snapshots": [*range(13, 31), *range(44, 62)], "data_flags_bit": 8,
             "scope": "Only certified compact NoIO1024 class; host branch is graph-keyed; all other kernels unchanged"}
    (args.source.parent / "profile-facts-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"consumer_sha256": proof["consumer_sha256"], "profile_cell_sha256": aliased["source_sha256"]}))


if __name__ == "__main__":
    main()
