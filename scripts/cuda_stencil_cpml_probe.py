#!/usr/bin/env python3
"""Generate a CUDA correctness probe for typed CPML cell traces.

This probes local region equations, not production slab packing or throughput.
Common-extent periodic fields and full-volume auxiliary arrays are deliberate:
all 8 active-axis signatures and FP32/BF16 recurrence semantics are compared
with independent handwritten C++ Yee/CPML updates, including exact FMA order.
No GPU work occurs without --run. nvcc compilation is optional with --compile.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
from pathlib import Path

try:
    from scripts.cuda_stencil_trace import emit_cuda_block, trace_cpml_step
except ModuleNotFoundError:
    from cuda_stencil_trace import emit_cuda_block, trace_cpml_step


def _state_id(name):
    if name.startswith("psi_"):
        return 6 + (0 if name[4] == "H" else 6) + int(name[5:])
    return (0 if name[0] == "H" else 3) + "xyz".index(name[1])


def generate_cpml_probe():
    kernels, calls = [], []
    for mask, half in itertools.product(range(8), range(2)):
        axes = tuple(a for a in range(3) if mask & (1 << a))
        updates = trace_cpml_step(
            active_axes=axes, precision="bfloat16" if half else "float32"
        )
        for phase, family in enumerate("HE"):
            selected = tuple(
                u
                for u in updates
                if u.output.field.startswith(family)
                or u.output.field.startswith("psi_" + family)
            )

            def load(read):
                return f"state[{_state_id(read.field)}*n+p]"

            def coefficient(read):
                if read.name == "inv_spacing":
                    return ".73f"
                if read.name.startswith("decay_"):
                    return ".999f" if read.name[6] == "H" else ".998f"
                if read.name.startswith("source_"):
                    return f"source_coefficient({_state_id(read.name[7:])},p)"
                if read.name.startswith("inv_kappa_"):
                    name, kind = read.name[10:], 2
                elif read.name.startswith("a_"):
                    name, kind = read.name[2:], 0
                else:
                    name, kind = read.name[2:], 1
                term = int(name[5:]) + (0 if name[4] == "H" else 6)
                return f"profile_coefficient({term},{kind},{('z', 'y', 'x')[read.axes[0]]})"

            def affine_load(read):
                dz, dy, dx = read.offset
                if dz == dy == dx == 0:
                    return load(read)
                return f"state[{_state_id(read.field)}*n+(wrap(z+({dz}),nz)*ny+wrap(y+({dy}),ny))*nx+wrap(x+({dx}),nx)]"

            block = emit_cuda_block(
                selected,
                read_name=affine_load,
                coefficient_name=coefficient,
                write_name=load,
            )["source"]
            kernels.append(
                f"__global__ void phase_{mask}_{half}_{phase}(float* state,int nx,int ny,int nz) {{\n"
                "const int n=nx*ny*nz,p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;\n"
                "const int x=p%nx,y=p/nx%ny,z=p/(nx*ny);\n" + block + "\n}"
            )
        calls.append(
            f"if(mask=={mask} && half=={half}) {{ phase_{mask}_{half}_0<<<blocks,256>>>(device,nx,ny,nz); phase_{mask}_{half}_1<<<blocks,256>>>(device,nx,ny,nz); }}"
        )
    return r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));std::exit(2); } } while(0)
__host__ __device__ int wrap(int p,int n) { int q=p%n;return q<0?q+n:q; }
__host__ __device__ float source_coefficient(int c,int p) {return (c<3?.07f:.09f)*(1.f+.01f*((p+c)%7));}
__host__ __device__ float profile_coefficient(int term,int kind,int coordinate) {
  if(kind==0)return -.01f*(1.f+.1f*((coordinate+term)%7));
  if(kind==1)return .97f-.01f*((coordinate+term)%5);
  return .8f+.025f*((coordinate+term)%3);
}
@KERNELS@
float bf16(float value) { uint32_t word;std::memcpy(&word,&value,4);word+=0x7fff+((word>>16)&1);word&=0xffff0000;std::memcpy(&value,&word,4);return value; }
void reference_step(std::vector<float>& state,int nx,int ny,int nz,int mask,bool half) {
  const int n=nx*ny*nz;
  auto at=[&](int c,int z,int y,int x) {return state[c*n+(wrap(z,nz)*ny+wrap(y,ny))*nx+wrap(x,nx)];};
  const int axes[6]={1,0,0,2,2,1};
  for(int phase=0;phase<2;++phase) {
    auto output=state;
    for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nx;++x) {
      const int p=(z*ny+y)*nx+x,coordinates[3]={z,y,x};
      float d[6];
      if(phase==0) {
        d[0]=(at(5,z,y+1,x)-at(5,z,y,x))*.73f;
        d[1]=(at(4,z+1,y,x)-at(4,z,y,x))*.73f;
        d[2]=(at(3,z+1,y,x)-at(3,z,y,x))*.73f;
        d[3]=(at(5,z,y,x+1)-at(5,z,y,x))*.73f;
        d[4]=(at(4,z,y,x+1)-at(4,z,y,x))*.73f;
        d[5]=(at(3,z,y+1,x)-at(3,z,y,x))*.73f;
      } else {
        d[0]=(at(2,z,y,x)-at(2,z,y-1,x))*.73f;
        d[1]=(at(1,z,y,x)-at(1,z-1,y,x))*.73f;
        d[2]=(at(0,z,y,x)-at(0,z-1,y,x))*.73f;
        d[3]=(at(2,z,y,x)-at(2,z,y,x-1))*.73f;
        d[4]=(at(1,z,y,x)-at(1,z,y,x-1))*.73f;
        d[5]=(at(0,z,y,x)-at(0,z,y-1,x))*.73f;
      }
      for(int term=0;term<6;++term) {
        if(mask&(1<<axes[term])) {
          int t=phase*6+term,coord=coordinates[axes[term]];
          float next=std::fma(profile_coefficient(t,1,coord),state[(6+t)*n+p],profile_coefficient(t,0,coord)*d[term]);
          output[(6+t)*n+p]=half?bf16(next):next;
          d[term]=std::fma(d[term],profile_coefficient(t,2,coord),next);
        }
        d[term]=(term%2?-1.f:1.f)*d[term];
      }
      for(int c=0;c<3;++c) {
        int field=phase*3+c;
        float source=source_coefficient(field,p),decay=phase?.998f:.999f;
        output[field*n+p]=std::fma(phase?source:-source,d[2*c]+d[2*c+1],decay*state[field*n+p]);
      }
    }
    state.swap(output);
  }
}
int main(int argc,char**argv) {
  int nx=argc>1?std::atoi(argv[1]):11,ny=argc>2?std::atoi(argv[2]):9,nz=argc>3?std::atoi(argv[3]):7,steps=argc>4?std::atoi(argv[4]):5;
  if(nx<1||ny<1||nz<1||steps<1||int64_t(nx)*ny*nz>INT32_MAX/18)return 2;
  int n=nx*ny*nz,blocks=(n+255)/256;size_t bytes=size_t(18)*n*sizeof(float);
  std::vector<float> seed(18*n),actual(18*n);uint32_t rng=77;
  for(auto& value:seed){rng=rng*1664525u+1013904223u;value=(int(rng>>8)-8388608)*1e-8f;}
  float* device;CHECK(cudaMalloc(&device,bytes));
  for(int mask=0;mask<8;++mask)for(int half=0;half<2;++half) {
    auto initial=seed;
    if(half)for(size_t i=size_t(6)*n;i<initial.size();++i)initial[i]=bf16(initial[i]);
    auto expected=initial;
    for(int step=0;step<steps;++step)reference_step(expected,nx,ny,nz,mask,half);
    CHECK(cudaMemcpy(device,initial.data(),bytes,cudaMemcpyHostToDevice));
    for(int step=0;step<steps;++step){@CALLS@ CHECK(cudaGetLastError());}
    CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(actual.data(),device,bytes,cudaMemcpyDeviceToHost));
    size_t mismatches=0;float error=0;
    for(size_t i=0;i<actual.size();++i) {
      if(!std::isfinite(actual[i]))return 3;
      error=std::max(error,std::abs(actual[i]-expected[i]));
      mismatches+=std::memcmp(&actual[i],&expected[i],4)!=0;
    }
    std::printf("{\"mask\":%d,\"bf16\":%d,\"bit_mismatches\":%zu,\"max_abs_error\":%.9g}\n",mask,half,mismatches,error);
    if(mismatches)return 4;
  }
  CHECK(cudaFree(device));
}
""".replace("@KERNELS@", "\n".join(kernels)).replace("@CALLS@", "\n".join(calls))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, binary = args.output_dir / "cpml_trace.cu", args.output_dir / "cpml_trace"
    source.write_text(generate_cpml_probe())
    if args.compile or args.run:
        if not args.nvcc:
            parser.error("nvcc is required for --compile/--run")
        subprocess.run(
            [
                args.nvcc,
                "-std=c++17",
                "-O3",
                "--fmad=false",
                "-Xcompiler=-ffp-contract=off",
                "-arch=sm_86",
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
            timeout=180,
        )
    if args.run:
        result = subprocess.run(
            [str(binary.resolve())],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        report = {
            "scope": "local CPML equation/codegen validation; not production layout/throughput",
            "shape_zyx": [7, 9, 11],
            "steps": 5,
            "cases": [json.loads(line) for line in result.stdout.splitlines()],
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
