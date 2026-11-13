"""Compile-only UniformE resource specialization; no true-path dispatch.

Eligible native tiles would use three common bit-exact FP32 source values after
host proof. This patch only measures the resulting device body. Normal launch
selection remains UniformE=false; the true specialization is referenced solely
by a resource-query function and must not be timed as a physics candidate yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one uniform-E anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source, all_policies=False, noio_min_two=False):
    policy = "int Policy=(Fast?0:5)" in source
    if all_policies and not policy:
        raise ValueError("All-policy resources require the six-policy consumer")
    if noio_min_two and not all_policies:
        raise ValueError("NoIO residency diagnostic requires all-policy queries")
    env_arguments = "int Layout,bool Fast,bool IsotropicH" + (
        ",int Policy=(Fast?0:5)" if policy else ""
    )
    kernel_arguments = "int Depth,int Layout,bool Fast,bool IsotropicH" + (
        ",int Policy=(Fast?0:5)" if policy else ""
    )
    env_use = "Layout,Fast,IsotropicH" + (",Policy" if policy else "")
    source = once(
        source,
        "int nz,ny,nx,frontier,macro_step;float h_source;",
        "int nz,ny,nx,frontier,macro_step;float h_source;\n float uniform_e[3]; // Resource-only until host tile proof populates/selects it.",
    )
    source = once(
        source,
        f"template<{env_arguments}> struct Environment",
        f"template<{env_arguments},bool UniformE=false> struct Environment",
    )
    start = source.index(
        f"template<{env_arguments},bool UniformE=false> struct Environment"
    )
    end = source.index("// Canonical staggering was proved", start)
    env = source[start:end]
    env = once(
        env,
        " const Parameters&p;uint8_t flags;",
        " const Parameters&p;uint8_t flags;\n float uniform_ex=0.f,uniform_ey=0.f,uniform_ez=0.f;",
    )
    env = once(
        env,
        "  const int c=index-3;const auto&v=p.phase[1];",
        "  const int c=index-3;\n  if constexpr(UniformE)return c==0?uniform_ex:c==1?uniform_ey:uniform_ez;\n  const auto&v=p.phase[1];",
    )
    source = source[:start] + env + source[end:]
    source = once(
        source,
        f"template<{kernel_arguments}>\n__global__",
        f"template<{kernel_arguments},bool UniformE=false>\n__global__",
    )
    source = once(
        source,
        f"Environment<{env_use}>env{{p,flags}};",
        f"Environment<{env_use},UniformE>env{{p,flags}};\n if constexpr(UniformE){{env.uniform_ex=p.uniform_e[0];env.uniform_ey=p.uniform_e[1];env.uniform_ez=p.uniform_e[2];}}",
    )
    source += r"""
// This query forces compilation ONLY. Normal host dispatch never selects true.
extern "C" int BeamzNativeUniformEResourceQuery(int*registers,int*max_threads,size_t*shared_bytes){
 using namespace beamz::cuda::moving_native;
 cudaFuncAttributes attributes{};
 const auto status=cudaFuncGetAttributes(&attributes,Asymmetric<16,1,true,true,true>);
 if(status!=cudaSuccess)return status;
 if(registers)*registers=attributes.numRegs;
 if(max_threads)*max_threads=attributes.maxThreadsPerBlock;
 if(shared_bytes)*shared_bytes=SharedBytes;
 return cudaSuccess;
}
"""
    if policy:
        source = once(
            source,
            "Asymmetric<16,1,true,true,true>",
            "Asymmetric<16,1,true,true,0,true>",
        )
    if all_policies:
        source += """
// Compile-only class comparisons; no physics dispatcher calls this function.
extern "C" int BeamzNativeUniformEPolicyResourceQuery(int policy,int*registers){
 using namespace beamz::cuda::moving_native;
 cudaFuncAttributes attributes{};cudaError_t status=cudaErrorInvalidValue;
 switch(policy){
"""
        for index in range(6):
            fast = "true" if index == 0 else "false"
            source += (
                f" case {index}:status=cudaFuncGetAttributes(&attributes,"
                f"Asymmetric<16,1,{fast},true,{index},true>);break;\n"
            )
        source += """ }
 if(status==cudaSuccess&&registers)*registers=attributes.numRegs;
 return status;
}
"""
    if noio_min_two:
        source = once(
            source,
            "__launch_bounds__(Threads,(Fast?2:1))",
            "__launch_bounds__(Threads,((Fast||(UniformE&&Policy==1))?2:1))",
        )
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all-policies", action="store_true")
    p.add_argument("--noio-min-two", action="store_true")
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve():
        p.error("Input is frozen; output must differ")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(patch(a.input.read_text(), a.all_policies, a.noio_min_two))
    print(
        json.dumps(
            {
                "input": str(a.input),
                "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
                "output": str(a.output),
                "output_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
                "resource_only": True,
                "uniform_true_dispatched": False,
            }
        )
    )


if __name__ == "__main__":
    main()
