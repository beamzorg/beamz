"""Compile the phase-CSE lowering against the existing independent Yee oracle.

Logical staggered fields, all psi outputs, boundaries, signed zeros and pure
Finish functions are checked. This does not validate CUDA synchronization,
native packed addressing or the dispatcher's disjoint allocation certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import StreamShape
from scripts.cuda_trace_axis_guards import AxisGuardClass
from scripts.generate_cuda_native_phase import generate
from scripts.validate_cuda_streamed_host import host_source


def validate(output, *, inactive=False, zero=False, metallic=0, depth=3, cached_read_fact=None, axis_guard=None):
    generated = generate(cached_read_fact=cached_read_fact, axis_guard=axis_guard)
    shape = StreamShape((2, 2, 3), depth, 128)
    source = host_source(generated, shape, cpml=True, native=True, metallic=metallic,
                         inactive=inactive, zero_fields=zero, native_cell=True, scalar_h_class=True)
    first = source.index(" Env env{data,0,0,0};")
    last = source.index(" int errors=0;", first)
    execution = r"""
 Env env{data,0,0,0};
 for(int v=1;v<=T;++v)for(int phase=0;phase<2;++phase){
  for(int z=0;z<=NZ;++z)for(int y=0;y<=NY;++y)for(int x=0;x<=NX;++x){
   auto read=[&](int field,int version,int dz,int dy,int dx){int zz=z+dz,yy=y+dy,xx=x+dx;
    if(!valid(field,zz,yy,xx))return 0.f;int i=(version*6+field)*P+at(zz,yy,xx);
    if(version&&data.written[i]!=1)std::abort();return data.field[i];};
   const auto values=phase?beamz::cuda::trace_native_phase::AdvanceNativePhase<1>(env,read,v,z,y,x):
                          beamz::cuda::trace_native_phase::AdvanceNativePhase<0>(env,read,v,z,y,x);
   env.Publish(3*phase,v,z,y,x,values.x);env.Publish(3*phase+1,v,z,y,x,values.y);env.Publish(3*phase+2,v,z,y,x,values.z);
  }
 }
"""
    source = source[:first] + execution + source[last:]
    if cached_read_fact is not None:
        source = source.replace("beamz::cuda::trace_native_phase::AdvanceNativePhase", "beamz::cuda::trace_native_phase_speculated::AdvanceNativePhase")
        source = source.replace("struct Env {Data&data;", "struct Env {static constexpr bool HasSources=false,HasObserve=false;Data&data;")
    if axis_guard is not None:
        source = source.replace("beamz::cuda::trace_native_phase::AdvanceNativePhase", f"beamz::cuda::trace_native_phase_axes_{axis_guard.native_mask}::AdvanceNativePhase")
        source = source.replace("AdvanceNativePhase<1>", f"AdvanceNativePhase<1,{axis_guard.native_mask}>").replace("AdvanceNativePhase<0>", f"AdvanceNativePhase<0,{axis_guard.native_mask}>")
        logical_mask = axis_guard.native_mask >> 1
        source = source.replace("return !Inactive&&(p[axis]==0", f"return (({logical_mask}u>>axis)&1u)&&!Inactive&&(p[axis]==0")
        source = source.replace("bool active=!Native||(!Inactive&&", f"bool active=((({logical_mask}u>>TermAxis[term])&1u)!=0)&&(!Native||(!Inactive&&")
        source = source.replace("D[f][TermAxis[term]]-1));", "D[f][TermAxis[term]]-1)));")
        source = source.replace("Native&&(Inactive||(coord!=0", f"Native&&(!(({logical_mask}u>>axis)&1u)||Inactive||(coord!=0")
        source = source.replace("if(Inactive&&kind>=2)std::abort();", f"if(kind>=2&&(Inactive||!(({logical_mask}u>>TermAxis[f%6])&1u)))std::abort();")
    source = source.replace("struct Data{std::vector<float>field,psi;", "struct Data{uint64_t profile_loads=0;std::vector<float>field,psi;")
    source = source.replace("float Coefficient(int kind,int f,int z,int y,int x)const{return coefficient(kind,f,z,y,x);}",
                            "float Coefficient(int kind,int f,int z,int y,int x)const{if(kind>=2)++data.profile_loads;return coefficient(kind,f,z,y,x);}")
    counter = r'''
 uint64_t expected_profile_loads=0;const int pairs[3][3]={{0,0,2},{1,0,1},{3,1,2}};
 for(int v=1;v<=T;++v)for(int phase=0;phase<2;++phase)
  for(int z=0;z<=NZ;++z)for(int y=0;y<=NY;++y)for(int x=0;x<=NX;++x)for(const auto&pair:pairs)
   if((valid(3*phase+pair[1],z,y,x)||valid(3*phase+pair[2],z,y,x))&&env.LocalActive(6*phase+pair[0],z,y,x))expected_profile_loads+=3;
 if(expected_profile_loads!=data.profile_loads){std::fprintf(stderr,"profile loads got%llu expected%llu\n",(unsigned long long)data.profile_loads,(unsigned long long)expected_profile_loads);return 2;}
 std::printf("{\"profile_loads\":%llu,\"expected_shared_pair_loads\":%llu}\n",(unsigned long long)data.profile_loads,(unsigned long long)expected_profile_loads);
'''
    source = source.replace(" int errors=0;", counter + " int errors=0;", 1)
    anchor = " if(kind<2)return kind?(.03f+.002f*f+.0001f*((z+2*y+3*x)%5)):(.99f-.0003f*(f%3));"
    if source.count(anchor) != 1:
        raise ValueError("Independent oracle coefficient anchor changed")
    source = source.replace(anchor, " if(kind==1&&f>=3)return .03f+.002f*f;\n" + anchor, 1)
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "host.cc", output / "host"
    cpp.write_text(source)
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("C++20 compiler required")
    command = [compiler, "-std=c++20", "-O1", "-pthread", "-ffp-contract=off", str(cpp), "-o", str(binary)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (output / "build.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    run = subprocess.run([str(binary.resolve())], capture_output=True, text=True, timeout=90)
    (output / "run.log").write_text(run.stdout + run.stderr)
    report = {"scope": __doc__, "generated_sha256": generated["source_sha256"],
              "proof_sha256": generated["proof"]["proof_sha256"],
              "host_sha256": hashlib.sha256(source.encode()).hexdigest(), "command": command,
              "inactive": inactive, "zero_fields": zero, "metallic": metallic, "depth": depth,
              "run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    if axis_guard is not None:
        report["axis_guard_zyx"] = axis_guard.axes
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    run.check_returncode()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--inactive", action="store_true")
    p.add_argument("--zero", action="store_true")
    p.add_argument("--metallic", type=int, default=0)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--active-axes", type=int, nargs="*", choices=(0, 1, 2))
    a = p.parse_args()
    result = validate(a.output_dir, inactive=a.inactive, zero=a.zero, metallic=a.metallic, depth=a.depth,
                      axis_guard=None if a.active_axes is None else AxisGuardClass(tuple(a.active_axes)))
    print(result["stdout"].strip())


if __name__ == "__main__":
    main()
