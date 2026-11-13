"""Use the actual phase-profile CSE IR in a copied certified NoIO consumer.

The field/trail/ghost/barrier schedule is unchanged. Host admission additionally
requires all prepared mutable psi ranges disjoint, including readonly profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_phase_state_fact import generate_guard
from scripts.generate_cuda_native_phase import generate
from scripts.patch_cuda_native_runtime_fact_cell import CANONICAL_CELL, once

PARENT = "2355f8a3a40866ee4e59110633628a36b2796f7c7947f831fd6d5dcad35fd4f3"
HOST_PARENT = "fdbb7d35a23da7bf3ed31bc413b558ebf933ba62a00d35c951c430a8cad4623e"


def patch_consumer(source):
    if hashlib.sha256(source.encode()).hexdigest() != PARENT:
        raise ValueError("Exact runtime profile-alias parent required")
    source = once(source, '#include "asymmetric_native_cell_runtime_profiles_generated.cuh"',
                  '#include "asymmetric_native_cell_runtime_profiles_generated.cuh"\n#include "asymmetric_native_phase_generated.cuh"')
    source = once(source, "bool AliasProfiles=false>\n__device__", "bool AliasProfiles=false,bool JointPhase=false>\n__device__")
    source = once(source, ' static_assert(!AliasProfiles||', ' static_assert(!JointPhase||(UseRuntimeFactCell&&AliasProfiles),"Joint phase requires admitted runtime/profile classes");\n static_assert(!AliasProfiles||')
    for phase, family in ((0, "h"), (1, "e")):
        offset = "+3" if phase else ""
        old = f"""     #pragma unroll
     for(int c=0;c<3;++c){{float value=0.f;if(env.Valid(c{offset},z,y,x)){{
      if(c==0)value=AdvanceRuntimeSelected<{phase},0,UseRuntimeFactCell,AliasProfiles>(env,read,n,z,y,x);
      else if(c==1)value=AdvanceRuntimeSelected<{phase},1,UseRuntimeFactCell,AliasProfiles>(env,read,n,z,y,x);
      else value=AdvanceRuntimeSelected<{phase},2,UseRuntimeFactCell,AliasProfiles>(env,read,n,z,y,x);
     }}{family}[c][j]=value;}}"""
        new = f"""     if constexpr(JointPhase){{
      const auto value=beamz::cuda::trace_native_phase::AdvanceNativePhase<{phase}>(env,read,n,z,y,x);
      {family}[0][j]=value.x;{family}[1][j]=value.y;{family}[2][j]=value.z;
     }}else{{
{old}
     }}"""
        source = once(source, old, new)
    source = once(source, "UniformBody<Depth,1,false,true,1,true,true,true>(p,tiles);",
                  "UniformBody<Depth,1,false,true,1,true,true,true,true>(p,tiles);")
    return source.replace("AsymmetricNoIOProfileAlias1024", "AsymmetricNoIOProfilePhase1024")


def patch_host(source, generated):
    if hashlib.sha256(source.encode()).hexdigest() != HOST_PARENT:
        raise ValueError("Exact profile alias host source required")
    source = once(source, '#include "runtime_profile_fact_guard.h"',
                  '#include "runtime_profile_fact_guard.h"\n#include "runtime_phase_state_fact.h"')
    source = once(source, "    if(moving_equal_axis_profiles_)moving_workspace_.data_flags|=beamz::cuda::moving::kEqualAxisProfiles;", """    BeamzBuffer phase_state[12]{},phase_profile[36]{};
    for(int family=0;family<2;++family){const auto&v=family?validate.program.e_ab:validate.program.h_ab;
      for(int term=0;term<6;++term){phase_state[6*family+term]=v.inputs[31+term];
        for(int kind=0;kind<3;++kind)phase_profile[18*family+3*term+kind]=v.inputs[13+3*term+kind];}}
    moving_joint_state_independent_=BeamzNativeIndependentPhaseState(phase_state,phase_profile);
    if(moving_equal_axis_profiles_&&moving_joint_state_independent_)moving_workspace_.data_flags|=beamz::cuda::moving::kEqualAxisProfiles;""")
    source = once(source, "moving_profile_alias_noio_tiles_=moving_equal_axis_profiles_?certified_noio:0;",
                  "moving_profile_alias_noio_tiles_=(moving_workspace_.data_flags&beamz::cuda::moving::kEqualAxisProfiles)?certified_noio:0;")
    source = once(source, "  bool moving_equal_axis_profiles_=false;", "  bool moving_equal_axis_profiles_=false,moving_joint_state_independent_=false;")
    source = once(source, '    result["moving_equal_axis_profiles"]',
                  '    result["moving_joint_state_independent"]=moving_joint_state_independent_;result["moving_joint_profile_noio_tiles"]=moving_profile_alias_noio_tiles_;\n    result["moving_equal_axis_profiles"]')
    return once(source, '  module.attr("moving_runtime_profile_alias")=true;',
                '  module.attr("moving_runtime_profile_alias")=true;\n  module.attr("moving_joint_profile_phase")=true;\n'
                f'  module.attr("moving_joint_profile_proof_sha256")="{generated["proof"]["proof_sha256"]}";')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    a = parser.parse_args()
    if a.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("Only private copied source may be patched")
    src = a.source / "src"
    if hashlib.sha256((src / "asymmetric_native_cell_generated.cuh").read_bytes()).hexdigest() != CANONICAL_CELL:
        raise ValueError("Canonical physics header changed")
    generated, state = generate(), generate_guard()
    consumer, host = src / "asymmetric_native.cuh", src / "native_extension.cc"
    result, host_result = patch_consumer(consumer.read_text()), patch_host(host.read_text(), generated)
    (src / "asymmetric_native_phase_generated.cuh").write_text(generated["source"])
    (src / "runtime_phase_state_fact.h").write_text(state["source"])
    consumer.write_text(result)
    host.write_text(host_result)
    proof = {"parent_consumer_sha256": PARENT, "parent_host_sha256": HOST_PARENT,
             "canonical_cell_sha256": CANONICAL_CELL,
             "consumer_sha256": hashlib.sha256(result.encode()).hexdigest(),
             "host_sha256": hashlib.sha256(host_result.encode()).hexdigest(),
             "phase": {k: v for k, v in generated.items() if k != "source"},
             "state_admission": {k: v for k, v in state.items() if k != "source"},
             "binding": "Actual prepared phase-state/profile ranges checked before data_flags8; per-component Valid retained; all field/trail/barrier sites unchanged"}
    (a.source.parent / "joint-phase-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"consumer_sha256": proof["consumer_sha256"], "phase_header_sha256": generated["source_sha256"]}))


if __name__ == "__main__":
    main()
