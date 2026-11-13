"""Compile the actual parent/new LocalActive methods as a CPU guard oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import derive_geometry
from scripts.patch_cuda_v12_noio_axis_bundle import PARENT_HASHES, compose_text, sha


def method(text):
    start = text.index(" __device__ __forceinline__ bool LocalActive(")
    end = text.index(" __device__ __forceinline__ int PsiOffset(", start)
    return text[start:end].replace("__device__", "").replace("__forceinline__", "")


def validate(source, output, compiler="c++"):
    output.mkdir(parents=True, exist_ok=True)
    original = (source / "src/asymmetric_native.cuh").read_text()
    if sha(original.encode()) != PARENT_HASHES["asymmetric_native.cuh"]:
        raise ValueError("Exact v12 consumer required")
    changed = compose_text(original)
    geometry = derive_geometry()
    term_axes = {int(t["term"][-1]): t["axis_zyx"] for t in geometry["cpml_terms"]}
    assert all(t["axis_zyx"] == term_axes[int(t["term"][-1])] for t in geometry["cpml_terms"])
    # Actual trace geometry independently fixes term IDs, not an assumed label.
    assert term_axes == {0: 1, 1: 0, 2: 0, 3: 2, 4: 2, 5: 1}
    axes = ",".join(str(term_axes[i]) for i in range(6))
    cpp = f"""#include <cstdint>
#include <cstdio>
namespace beamz::cuda::yee {{constexpr int CpmlAxis(int t){{constexpr int a[]={{ {axes} }};return a[t];}}}}
struct P {{int physical_n[3];}};
template<bool HasCpml>struct Before{{const P&p;uint8_t flags;{method(original)}}};
template<bool HasCpml,int ActiveAxes=-1>struct After{{const P&p;uint8_t flags;{method(changed)}}};
int main(){{long long checks=0;int errors=0,wrong=0;
 for(P p:{{P{{{{35,49,65}}}},P{{{{136,136,136}}}},P{{{{29,30,31}}}}}})
 for(int cpml=0;cpml<2;++cpml)for(int f=0;f<256;++f)for(int slot=0;slot<12;++slot)
 for(int coord=-2;coord<=((p.physical_n[0]>p.physical_n[1]?p.physical_n[0]:p.physical_n[1])>p.physical_n[2]?(p.physical_n[0]>p.physical_n[1]?p.physical_n[0]:p.physical_n[1]):p.physical_n[2])+2;++coord){{
  int z=coord,y=coord,x=coord;bool a,b,w;
  if(cpml){{a=Before<true>{{p,uint8_t(f)}}.LocalActive(slot,z,y,x);
   switch(f&14){{case2:b=After<true,1>{{p,uint8_t(f)}}.LocalActive(slot,z,y,x);break;
   case4:b=After<true,2>{{p,uint8_t(f)}}.LocalActive(slot,z,y,x);break;
   case8:b=After<true,4>{{p,uint8_t(f)}}.LocalActive(slot,z,y,x);break;
   default:b=After<true>{{p,uint8_t(f)}}.LocalActive(slot,z,y,x);break;}}
   w=After<true,4>{{p,uint8_t(f)}}.LocalActive(slot,z,y,x);
  }}else{{a=Before<false>{{p,uint8_t(f)}}.LocalActive(slot,z,y,x);b=After<false,7>{{p,uint8_t(f)}}.LocalActive(slot,z,y,x);w=a;}}
  errors+=a!=b;wrong+=(f&14)==2&&a!=w;++checks;
 }}
 printf("{{\\\"checks\\\":%lld,\\\"errors\\\":%d,\\\"wrong_axis_witnesses\\\":%d}}\\n",checks,errors,wrong);
 return errors||!wrong;
}}
""".replace("case2:", "case 2:").replace("case4:", "case 4:").replace("case8:", "case 8:")
    # The explicit aggregate range needs this standard header on all toolchains.
    cpp = "#include <initializer_list>\n" + cpp
    cfile = output / "actual-local-active.cc"
    executable = output / "actual-local-active"
    cfile.write_text(cpp)
    command = [compiler, "-std=c++17", "-O2", str(cfile), "-o", str(executable)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    run = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    result.update({"parent_sha256": sha(original.encode()), "new_sha256": sha(changed.encode()),
                   "oracle_cpp_sha256": hashlib.sha256(cpp.encode()).hexdigest(),
                   "compiler_command": command, "scope": "actual LocalActive bodies; no GPU or FP execution",
                   "source_mask_binding": "source mask remains argument6; active axes argument7"})
    (output / "actual-guard-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(validate(a.source, a.output)))


if __name__ == "__main__":
    main()
