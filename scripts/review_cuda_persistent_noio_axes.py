"""Read-only actual LocalActive equivalence for persistent one-hot NoIO axes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_persistent_noio_axes import transform
from scripts.validate_cuda_cpml_point_context import method


HARNESS = r'''
#include <array>
#include <cstdint>
#include <cstdio>
#define __device__
#define __forceinline__ inline
namespace beamz::cuda::yee{inline int CpmlAxis(int t){const int axes[6]={1,0,0,2,2,1};return axes[t];}}
struct Parameters{int physical_n[3];};
template<int Policy,int ActiveAxes=-1>struct ActualEnvironment{
 const Parameters&p;uint8_t flags;static constexpr bool HasCpml=Policy==1||Policy==5;
@LOCAL@
};
bool selected(const Parameters&p,uint8_t flags,int slot,int z,int y,int x){
 switch(flags&14){
  case 2:return ActualEnvironment<1,1>{p,flags}.LocalActive(slot,z,y,x);
  case 4:return ActualEnvironment<1,2>{p,flags}.LocalActive(slot,z,y,x);
  case 8:return ActualEnvironment<1,4>{p,flags}.LocalActive(slot,z,y,x);
  default:return ActualEnvironment<1>{p,flags}.LocalActive(slot,z,y,x);
 }
}
int main(){size_t checks=0,fresh_checks=0,bulk_checks=0,wrong=0,errors=0;
 for(auto dims:std::array<std::array<int,3>,4>{{{25,31,37},{32,32,32},{136,272,512},{512,564,680}}}){
  Parameters p{{dims[0],dims[1],dims[2]}};
  for(int facts=0;facts<256;++facts)for(int slot=0;slot<12;++slot){
   const int axis=beamz::cuda::yee::CpmlAxis(slot%6),size=p.physical_n[axis]+slot/6;
   for(int coordinate:{-2,-1,0,1,11,12,size-13,size-12,size-11,size-1,size,size+1,size+2}){
    const int z=axis==0?coordinate:p.physical_n[0]-1,y=axis==1?coordinate:1,x=axis==2?coordinate:p.physical_n[2];
    const uint8_t flags=uint8_t(facts);const bool want=ActualEnvironment<1>{p,flags}.LocalActive(slot,z,y,x);
    errors+=selected(p,flags,slot,z,y,x)!=want;++checks;
    errors+=selected(p,uint8_t(flags^128u),slot,z,y,x)!=want;++fresh_checks;
    errors+=ActualEnvironment<0>{p,flags}.LocalActive(slot,z,y,x);++bulk_checks;
    if((flags&14)==2)wrong+=ActualEnvironment<1,2>{p,flags}.LocalActive(slot,z,y,x)!=want;
    if((flags&14)==4)wrong+=ActualEnvironment<1,4>{p,flags}.LocalActive(slot,z,y,x)!=want;
    if((flags&14)==8)wrong+=ActualEnvironment<1,1>{p,flags}.LocalActive(slot,z,y,x)!=want;
   }
  }
 }
 errors+=wrong==0;
 std::printf("{\"actual_LocalActive_checks\":%zu,\"bit128_invariance_checks\":%zu,\"Fast_policy0_CPML_false_checks\":%zu,\"wrong_axis_counterexamples\":%zu,\"errors\":%zu}\n",checks,fresh_checks,bulk_checks,wrong,errors);return errors!=0;
}
'''


def review(parent, source, output):
    parent, source, output = Path(parent), Path(source), Path(output)
    original = (parent / "src/native_persistent.cuh").read_text()
    actual = (source / "src/native_persistent.cuh").read_text()
    if actual != transform(original):
        raise ValueError("Staged actual source differs from the reviewed exact transform")
    environment = (source / "src/asymmetric_native.cuh").read_text()
    declaration = "template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false,int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1> struct Environment"
    if declaration not in environment:
        raise ValueError("Actual Base template argument positions changed")
    base = "Environment<1,Fast,true,Policy,UniformE,-1,false,ActiveAxes>"
    if actual.count(base) != 2:
        raise ValueError("Both inheritance and Base alias must use identical actual argument positions")
    local = method(environment, "bool LocalActive(")
    if "if constexpr(ActiveAxes>=0)" not in local or "flags&(2u<<axis)" not in local:
        raise ValueError("Actual axis-vs-runtime flag semantics changed")
    expected_fast = "if(C().body_kind==0)PersistentUniformBody<Depth,true,0,false>(p);"
    if expected_fast not in actual:
        raise ValueError("Actual Fast caller no longer retains default ActiveAxes")
    for file in ("native_persistent.h", "asymmetric_native.cuh", "asymmetric_native_cell_generated.cuh"):
        if (parent / "src" / file).read_bytes() != (source / "src" / file).read_bytes():
            raise ValueError("An unrelated host/slot/field graph changed:" + file)
    output.mkdir(parents=True, exist_ok=True)
    program = HARNESS.replace("@LOCAL@", local)
    cpp, binary = output / "actual-axis-review.cc", output / "actual-axis-review"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(binary)]
    compilation = subprocess.run(command, capture_output=True, text=True)
    (output / "compile.log").write_text(compilation.stdout + compilation.stderr)
    compilation.check_returncode()
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    result = {"schema": "beamz.persistent-onehot-independent-review.v1", "result": "no-blocking-issue-found-CPU-only",
              "actual_CXX": json.loads(run.stdout), "consumer_sha256": hashlib.sha256(actual.encode()).hexdigest(),
              "Base_argument_order": [1, "Fast", True, "Policy", "UniformE", -1, False, "ActiveAxes"],
              "Fast_bulk_keeps_default_mask_minus1_and_CPML_false": True,
              "multi_axis_fallback_retains_runtime_flags": True, "future_bit128_ignored_by_dispatch_and_LocalActive": True,
              "dispatch_uniformity": "private immutable p.flags indexed by shared C().tile, fixed between queue assignment and body completion barriers; all CTA threads execute one selected body",
              "future_mixed_patch_composition": "current mixed-psi patcher intentionally rejects new parent hash; a reviewed composition must preserve masked low7 facts and point-site assignments",
              "ordered_FP_barriers_slot_workspace_unchanged": True,
              "oracle_cpp_sha256": hashlib.sha256(program.encode()).hexdigest(), "command": command,
              "GPU_executed": False,
              "scope": "actual original LocalActive compiled under UBSan versus new captured-mask cases/fallback over all256 flag bytes, H/E slots, rectangular supports and outside/CPML/native boundary coordinates; no GPU resource/parity claim"}
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(review(a.parent, a.source, a.output), indent=2))
