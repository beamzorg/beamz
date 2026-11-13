"""CG composition preserves every guarded scalar address and payload bit."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.patch_cuda_source_plane_axis_cg import (
    ORIGINAL,
    PARENTS,
    REPLACEMENT,
    transformed,
)


def test_parent_and_planner_dependency_fail_closed():
    with pytest.raises(ValueError, match="parent and planner"):
        transformed({n: b"wrong" for n in PARENTS})


def test_actual_composition_has_only_six_load_expression_changes():
    source = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-v1/source/src")
    if not source.exists():
        pytest.skip("Frozen research parent required")
    parent = {n: (source / n).read_bytes() for n in PARENTS}
    files = transformed(parent)
    changed = files["asymmetric_native.cuh"].decode()
    assert changed.count(REPLACEMENT) == 1
    assert changed.replace(REPLACEMENT, ORIGINAL, 1).encode() == parent["asymmetric_native.cuh"]
    assert all(files[n] == parent[n] for n in PARENTS if n != "asymmetric_native.cuh")


def test_exact_return_guards_addresses_and_words_for_all64_masks(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ compiler required")
    cpp = r'''#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
constexpr int SurfacePoints=735;
struct FieldPacket{float hx,hy,hz,ex,ey,ez;};
const float*base;std::vector<int>loads;
float __ldcg(const float*p){loads.push_back(int(p-base));return *p;}
FieldPacket original(unsigned mask,const float*point){
@ORIGINAL@
}
FieldPacket candidate(unsigned mask,const float*point){
@REPLACEMENT@
}
int main(){uint32_t patterns[]={0,0x80000000u,1,0x80000001u,0x3f800001u,0x7f800000u,0xff800000u,0x7fc12345u};int errors=0,cases=0;
 for(int trial=0;trial<8;++trial){float data[6*SurfacePoints]{};base=data;
  for(int f=0;f<6;++f)std::memcpy(data+f*SurfacePoints,&patterns[(trial+f)%8],4);
  for(unsigned mask=0;mask<64;++mask){loads.clear();const float*p=mask?data:nullptr;
   auto a=original(mask,p),b=candidate(mask,p);errors+=std::memcmp(&a,&b,sizeof(a))!=0;
   int j=0;for(int f=0;f<6;++f)if(mask&(1u<<f)){errors+=j>=int(loads.size())||loads[j]!=f*SurfacePoints;++j;}
   errors+=j!=int(loads.size());++cases;
  }
 }
 std::printf("{\"cases\":%d,\"errors\":%d}\n",cases,errors);return errors!=0;
}
'''.replace("@ORIGINAL@", ORIGINAL).replace("@REPLACEMENT@", REPLACEMENT)
    source, binary = tmp_path / "cg.cc", tmp_path / "cg"
    source.write_text(cpp)
    subprocess.run(["c++", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    run = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == {"cases": 512, "errors": 0}
