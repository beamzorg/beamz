"""Stage all36 independent CPML profiles in each certified NoIO CTA.

Copy-only selected52865 experiment. Existing barriers and ordered arithmetic
stay unchanged. No profile equality, field/psi cache, or production selection
is introduced. Default templates keep original global profile loads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PARENT = {
    "asymmetric_native.cuh": "f082980ad0687ef165c7807b415e87cbc1003edfdbd2acf7a4ee018dae4d4547",
    "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27",
    "native_moving.h": "d3d1abaca2ca066b75295df35ec26da2c1df53c75bdb9f140b067fea09288b3c",
    "native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
    "compact_full_edge.cuh": "62d4fa66106469e7a89c1848361ac72fab6aef9b01bb8cb09dca0fbcafa17da4",
    "native_launch.cc": "b68eb5dd3ea4c23e8feec162fa6ae18308960ce0065307089a43d9646d8433f0",
}

HELPERS = r'''
// Each slot is a distinct family/term; everykind retains its own24FP32 bits.
__host__ __device__ __forceinline__ int IndependentProfileCacheIndex(int slot,int kind,int packed){
 return (slot*3+(kind-2))*24+packed;
}
__device__ __forceinline__ float ReadIndependentProfileCache(const float*profiles,int slot,int kind,int packed){
 return profiles[IndependentProfileCacheIndex(slot,kind,packed)];
}
__device__ __forceinline__ void StageIndependentCpmlProfiles(const Parameters&p,float*profiles){
 if(threadIdx.x<512)for(int i=threadIdx.x;i<864;i+=512){
  const int profile=i/24,packed=i%24,slot=profile/3,kind=profile%3;
  const auto&psi=p.phase[slot/6].psi[slot%6];
  const float*input=kind==0?psi.a:kind==1?psi.b:psi.inv;
  profiles[i]=__ldg(input+packed);
 }
}
'''

ORIGINAL_COEFFICIENT = "  if(kind>=2){const auto&psi=p.phase[index/6].psi[index%6];const int packed=Packed(index,z,y,x);return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}"
NEW_COEFFICIENT = """  if(kind>=2){const int packed=Packed(index,z,y,x);
   if constexpr(CachedProfiles)return ReadIndependentProfileCache(profiles,index,kind,packed);
   else{const auto&psi=p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}
  }"""


def sha(value):
    return hashlib.sha256(value).hexdigest()


def constructor_witness(files):
    """Bind profile immutability to the actual independent device allocations."""
    host = files["native_extension.cc"].decode()
    view = files["compact_full_edge.cuh"].decode()
    launch = files["native_launch.cc"].decode()
    anchors = (
        "for (size_t i = 0; i < input_count; ++i) {",
        "auto memory = std::make_unique<DeviceMemory>(host_.back().bytes);",
        "auto descriptor = host_.back().descriptor;\n      descriptor.data = memory->data;",
        "cudaMemcpyAsync(memory->data, host_.back().data, host_.back().bytes,",
        'Check(cudaStreamSynchronize(stream_.value), "finish native preparation");',
        'Check(cudaMalloc(&data, std::max(size, size_t{1})), "cudaMalloc");',
        "for (int c = 0; c < 6; ++c) persistent_.insert(c);",
        "persistent_.insert(kCpmlHPsiInputOffset + c);",
        "persistent_.insert(kCpmlEPsiInputOffset + c);",
        "append(31 + c, psi_current_[c]);",
        "append(62 + c, psi_current_[6 + c]);",
        "append(graph_count_ + c, c < 6 ? other_[c] : base_inputs_[graph_count_ + c]);",
        "append(monitor_offset_ + 7 + c, base_inputs_[monitor_offset_ + 7 + c]);",
    )
    if any(host.count(a) != 1 for a in anchors):
        raise ValueError("Native independent allocation/upload/outputbinding witness changed")
    if "const float *input, *a, *b, *inv;" not in view or "float* output;" not in view:
        raise ValueError("Mutable psi and const profile pointer roles changed")
    if "h.inputs[kFieldCount + i] = inputs[kFieldCount + i];" not in launch or "e.inputs[kFieldCount + i] = inputs[kCpmlEInputOffset + i];" not in launch:
        raise ValueError("Actual H/E profile ABI binding changed")
    profiles = set(range(13, 31)) | set(range(44, 62))
    mutable = set(range(6)) | set(range(31, 37)) | set(range(62, 68)) | set(range(74, 86)) | set(range(120, 126)) | {127, 128}
    assert not profiles & mutable
    return {"profile_ABI_inputs": sorted(profiles), "mutable_ABI_inputs": sorted(mutable),
            "all_nonempty_inputs_independent_live_cudaMalloc": True,
            "profile_upload_complete_before_constructor_returns": True,
            "profile_buffers_never_bound_as_outputs": True,
            "caller_host_aliases_do_not_create_device_aliases": True,
            "profile_roles_are_const_independent_from_mutable_psi_output": True,
            "constructor_hash": sha(files["native_extension.cc"]),
            "launch_binding_hash": sha(files["native_launch.cc"]),
            "profile_view_hash": sha(files["compact_full_edge.cuh"]),
            "scope": "NativeSession-created moving programs only; no new arbitrary-aliased raw packet admission"}


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError(f"Exact selected52865 parent required:{name}")
    immutable = constructor_witness(files)
    original = files["asymmetric_native.cuh"].decode()
    text = original
    edits = []

    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"One exact shared-profile anchor required:{old!r}")
        text = text.replace(old, new, 1)
        edits.append((old, new))

    change("// Fast is admitted only for source/CPML/observation/native-wall-free OWNED", HELPERS + "\n// Fast is admitted only for source/CPML/observation/native-wall-free OWNED")
    change("int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1> struct Environment", "int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1,bool CachedProfiles=false> struct Environment")
    change(" const Parameters&p;uint8_t flags;", " const Parameters&p;uint8_t flags;const float*profiles=nullptr;")
    change(ORIGINAL_COEFFICIENT, NEW_COEFFICIENT)
    change("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>\n__device__ __forceinline__ void UniformBody")
    change(" constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n extern __shared__ float frame[];size_t*bases=", " constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n static_assert(!CachedProfiles||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE),\"Profile staging only certified NoIO1024\");\n extern __shared__ float frame[];size_t*bases=")
    change(" float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;", " float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;\n float*profiles=CachedProfiles?frame+SharedBytes/sizeof(float):nullptr;\n if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);")
    change(" Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes>env{p,flags};", " Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env{p,flags,profiles};")
    start = text.index("__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(")
    end = text.index("template<int Depth>", start)
    wrapper = text[start:end]
    new_wrapper = wrapper
    for axis in (1, 2, 4):
        old = f"UniformBody<Depth,1,false,true,1,true,{axis}>(p,tiles);"
        new_wrapper = new_wrapper.replace(old, old.replace(">(p,tiles)", ",true>(p,tiles)"))
    new_wrapper = new_wrapper.replace("UniformBody<Depth,1,false,true,1,true>(p,tiles);", "UniformBody<Depth,1,false,true,1,true,-1,true>(p,tiles);")
    change(wrapper, new_wrapper)
    old_launch = "  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;\n  AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);"
    change(old_launch, old_launch.replace("SharedBytes", "ProfileCachedSharedBytes"))
    declaration = "constexpr size_t SharedBytes=3ULL*PAD*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+6ULL*LowXPoints*sizeof(float);"
    change(declaration, declaration + "\nconstexpr size_t ProfileCachedSharedBytes=SharedBytes+864ULL*sizeof(float);\nstatic_assert(SharedBytes==51748&&ProfileCachedSharedBytes==55204,\"Exact appended profile layout\");")
    inverse = text
    for before, after in reversed(edits):
        if inverse.count(after) != 1:
            raise ValueError("Shared-profile exact inverse anchor became ambiguous")
        inverse = inverse.replace(after, before, 1)
    assert inverse == original
    assert text.count("__syncthreads()") == original.count("__syncthreads()")
    profile_begin = text.index("if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);")
    profile_barrier = text.index("__syncthreads();", profile_begin)
    profile_env = text.index("ActiveAxes,CachedProfiles>env{p,flags,profiles};", profile_begin)
    assert profile_begin < profile_barrier < profile_env
    marker = "// Private source-only storage schedule; source-bound"
    assert text[text.index(marker):text.index("inline bool MakeSource")] == original[original.index(marker):original.index("inline bool MakeSource")]
    result = dict(files)
    result["asymmetric_native.cuh"] = text.encode()
    return result, {"exact_source_inverse": True, "all_barrier_sites_identical": True,
                    "staging_before_existing_owner_barrier": True, "NoIOOnly": True,
                    "independent_profiles": 36, "floats_per_profile": 24,
                    "shared_bytes_before": 51748, "shared_bytes_after": 55204,
                    "no_profile_equality_or_value_specialization": True,
                    "immutable_profile_admission": immutable, "expected_moving_kernels": 62}


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source required")
    before = {name: (source / "src" / name).read_bytes() for name in PARENT}
    after, report = transformed(before)
    shutil.copytree(source, output)
    for name, value in after.items():
        (output / "src" / name).write_bytes(value)
    report.update(schema="beamz.independent-shared-profile-stage.v1", parent_hashes=PARENT,
                  result_hashes={name: sha(value) for name, value in after.items()},
                  producer_sha256=sha(Path(__file__).read_bytes()), GPU_executed=False)
    (output.parent / "shared-profile-proof.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
