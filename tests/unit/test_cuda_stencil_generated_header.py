"""Compile generated CUDA cell arithmetic on host with independent Yee oracle."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_generated_header_matches_independent_float32_fma_cells(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required for generated cell arithmetic oracle")
    source = tmp_path / "cell_oracle.cc"
    binary = tmp_path / "cell_oracle"
    source.write_text(r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#define __host__
#define __device__
#define __forceinline__ inline
inline float __fmul_rn(float a,float b) {return a*b;}
inline float __fsub_rn(float a,float b) {return a-b;}
inline float __fadd_rn(float a,float b) {return a+b;}
inline float __fmaf_rn(float a,float b,float c) {return std::fma(a,b,c);}
#include "cuda/src/physics/yee_cell_generated.cuh"
struct Values {
  float fields[6][2][3][3][3];
  float coefficients[2][3];
  float read(int field,int version,int z,int y,int x) const {
    return fields[field][version][z+1][y+1][x+1];
  }
};
float expected(const Values& v,int phase,int component,float inv) {
  auto r=[&](int f,int z=0,int y=0,int x=0) {return v.read(f,phase,z,y,x);};
  float a,b;
  if(phase==0) {
    if(component==0) {a=(r(5,0,1,0)-r(5))*inv;b=(r(4,1,0,0)-r(4))*inv;}
    else if(component==1) {a=(r(3,1,0,0)-r(3))*inv;b=(r(5,0,0,1)-r(5))*inv;}
    else {a=(r(4,0,0,1)-r(4))*inv;b=(r(3,0,1,0)-r(3))*inv;}
  } else {
    if(component==0) {a=(r(2)-r(2,0,-1,0))*inv;b=(r(1)-r(1,-1,0,0))*inv;}
    else if(component==1) {a=(r(0)-r(0,-1,0,0))*inv;b=(r(2)-r(2,0,0,-1))*inv;}
    else {a=(r(1)-r(1,0,0,-1))*inv;b=(r(0)-r(0,0,-1,0))*inv;}
  }
  float source=v.coefficients[1][component],decay=v.coefficients[0][component];
  float old=v.read(phase*3+component,0,0,0,0);
  return std::fma(phase?source:-source,a-b,decay*old);
}
template<int Phase,int Component> bool check(const Values& v,float inv) {
  auto read=[&](int f,int version,int dz,int dy,int dx) {return v.read(f,version,dz,dy,dx);};
  auto coefficient=[&](int kind,int component) {return v.coefficients[kind][component];};
  float actual=beamz::cuda::generated::AdvanceYeeCell<Phase,Component>(inv,read,coefficient);
  float reference=expected(v,Phase,Component,inv);
  if(std::memcmp(&actual,&reference,4)) {
    std::printf("phase%d component%d actual%.9g reference%.9g\n",Phase,Component,actual,reference);
    return false;
  }
  return true;
}
int main() {
  uint32_t rng=77;
  auto random=[&] {rng=rng*1664525u+1013904223u;return (int(rng>>8)-8388608)*1e-6f;};
  for(int trial=0;trial<1024;++trial) {
    Values v;
    for(auto& f:v.fields)for(auto& t:f)for(auto& z:t)for(auto& y:z)for(auto& x:y)x=random();
    for(auto& kind:v.coefficients)for(auto& component:kind)component=random();
    float inv=trial%3==0?.73f:trial%3==1?80e6f:.00123f;
    if(!check<0,0>(v,inv)||!check<0,1>(v,inv)||!check<0,2>(v,inv)||
       !check<1,0>(v,inv)||!check<1,1>(v,inv)||!check<1,2>(v,inv))return 1;
  }
  return 0;
}
""")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-I",
            str(ROOT),
            "-I",
            str(ROOT / "cuda/src/common"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
