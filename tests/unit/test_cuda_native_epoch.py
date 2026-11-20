"""Boundary-test the actual CUDA epoch primitive as a single-thread host body."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_epoch_exhaustion_is_sticky_across_cleared_run_diagnostics(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    text = (ROOT / "cuda/src/temporal/kernel.cuh").read_text()
    start = text.index("__global__ void BeginEpoch(")
    body_start = text.index("{", start)
    depth, end = 1, body_start + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    primitive = text[start:end]
    source = tmp_path / "epoch.cc"
    source.write_text(
        "#include <cstdint>\n#include <cassert>\n#define __global__\n"
        "struct { unsigned x; } threadIdx{0};\n"
        "void atomicOr(int32_t*p,int32_t v){*p|=v;}\n"
        + primitive
        + r"""
int main(){
 uint32_t epoch=0;int32_t bad=0;
 BeginEpoch(&epoch,&bad);assert(epoch==1&&bad==0);
 epoch=0xfffffffeu;BeginEpoch(&epoch,&bad);assert(epoch==0xffffffffu&&bad==0);
 for(int retry=0;retry<4;++retry){
  bad=0;BeginEpoch(&epoch,&bad);assert(epoch==0xffffffffu&&bad==4);
 }
 bad=1;BeginEpoch(&epoch,&bad);assert(epoch==0xffffffffu&&bad==5);
 threadIdx.x=1;epoch=9;bad=0;BeginEpoch(&epoch,&bad);assert(epoch==9&&bad==0);
}
"""
    )
    binary = tmp_path / "epoch"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True
    )
    subprocess.run([str(binary)], check=True)
