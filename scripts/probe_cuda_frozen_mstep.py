#!/usr/bin/env python3
"""Compile a graph-free probe of an immutable production packed CUDA core.

The supplied build's update.cu is included verbatim, with effective macros
recovered from its frozen CMake cache. This neither copies the kernel nor uses JAX/XLA, graph.cu,
or BeamzLaunchGraph. GPU work requires running the produced binary separately.
The harness tests real staggered supports, nonzero fields, heterogeneous byte
codes/scalar H, partial quadrant masks, input immutability, and guard writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

try:
    from scripts.build_cuda_candidate import digest_tree
except ModuleNotFoundError:
    from build_cuda_candidate import digest_tree


def harness_source(update_path):
    return r"""
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include @UPDATE@
#define CHECK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 2;}}while(0)
constexpr int Guard=32;
constexpr uint32_t Sentinel=0x7fc12345u;
struct HostField { int nz,ny,nx; std::vector<float> initial; std::vector<uint32_t> guarded; float *in_raw,*out_raw; };
int offset(const HostField& f,int z,int y,int x) {return (z*f.ny+y)*f.nx+x;}
float value(const std::array<HostField,6>& fields,const std::array<std::vector<float>,6>& state,int c,int z,int y,int x) {
  const auto& f=fields[c];
  if(z<0||z>=f.nz||y<0||y>=f.ny||x<0||x>=f.nx)return 0.f;
  return state[c][offset(f,z,y,x)];
}
void reference_step(const std::array<HostField,6>& fields,std::array<std::vector<float>,6>& state,
                    const float (&hc)[6],const float (&tables)[3][8],float inv) {
  auto output=state;
  for(int phase=0;phase<2;++phase) {
    for(int c=0;c<3;++c) {
      const int field=phase*3+c;const auto& f=fields[field];
      for(int z=0;z<f.nz;++z)for(int y=0;y<f.ny;++y)for(int x=0;x<f.nx;++x) {
        auto r=[&](int component,int dz=0,int dy=0,int dx=0){return value(fields,state,component,z+dz,y+dy,x+dx);};
        float a,b;
        if(phase==0) {
          if(c==0){a=(r(5,0,1,0)-r(5))*inv;b=(r(4,1,0,0)-r(4))*inv;}
          else if(c==1){a=(r(3,1,0,0)-r(3))*inv;b=(r(5,0,0,1)-r(5))*inv;}
          else {a=(r(4,0,0,1)-r(4))*inv;b=(r(3,0,1,0)-r(3))*inv;}
        } else {
          if(c==0){a=(r(2)-r(2,0,-1,0))*inv;b=(r(1)-r(1,-1,0,0))*inv;}
          else if(c==1){a=(r(0)-r(0,-1,0,0))*inv;b=(r(2)-r(2,0,0,-1))*inv;}
          else {a=(r(1)-r(1,0,0,-1))*inv;b=(r(0)-r(0,0,-1,0))*inv;}
        }
        const int p=offset(f,z,y,x);
        const int code=(p*13+c*7)%8;
        const float source=phase?tables[c][code]:-hc[3+c],decay=phase?1.f:hc[c];
        output[field][p]=std::fma(source,a-b,decay*state[field][p]);
      }
    }
    state.swap(output);output=state;
  }
}
int main(int argc,char**argv) {
  const int segment=argc>1?std::atoi(argv[1]):0;
  if(segment<0)return 2;
  constexpr int Nz=11,Ny=40,Nx=72,Ntx=3,Nty=3;
  const int sizes[6][3]={{Nz,Ny,Nx+1},{Nz,Ny+1,Nx},{Nz+1,Ny,Nx},
                         {Nz+1,Ny+1,Nx},{Nz+1,Ny,Nx+1},{Nz,Ny+1,Nx+1}};
  std::array<HostField,6> fields{};
  std::array<std::vector<float>,6> expected;
  uint32_t rng=77;
  MstepPackedCoreView view{};view.min_y=Ny;view.min_x=Nx;view.inv=.73f;
  for(int c=0;c<6;++c) {
    auto& f=fields[c];f.nz=sizes[c][0];f.ny=sizes[c][1];f.nx=sizes[c][2];
    const int n=f.nz*f.ny*f.nx;
    f.initial.resize(n);f.guarded.assign(n+2*Guard,Sentinel);
    for(int p=0;p<n;++p){rng=rng*1664525u+1013904223u;f.initial[p]=(int(rng>>8)-8388608)*1e-6f;}
    std::memcpy(f.guarded.data()+Guard,f.initial.data(),n*sizeof(float));expected[c]=f.initial;
    CHECK(cudaMalloc(&f.in_raw,f.guarded.size()*sizeof(float)));
    CHECK(cudaMalloc(&f.out_raw,f.guarded.size()*sizeof(float)));
    CHECK(cudaMemcpy(f.in_raw,f.guarded.data(),f.guarded.size()*sizeof(float),cudaMemcpyHostToDevice));
    std::vector<uint32_t> output_guard(f.guarded.size(),Sentinel);
    CHECK(cudaMemcpy(f.out_raw,output_guard.data(),output_guard.size()*sizeof(float),cudaMemcpyHostToDevice));
    view.input[c]={f.in_raw+Guard,f.nx,f.ny*f.nx};
    view.output[c]={f.out_raw+Guard,f.nx,f.ny*f.nx};
  }
  float hc[6]={.999f,.997f,.995f,.07f,.075f,.08f};float* device_hc;
  CHECK(cudaMalloc(&device_hc,sizeof(hc)));CHECK(cudaMemcpy(device_hc,hc,sizeof(hc),cudaMemcpyHostToDevice));
  float tables[3][8];uint32_t *device_codes[3];float *device_tables[3];
  for(int c=0;c<3;++c) {
    const auto& f=fields[3+c];const int n=f.nz*f.ny*f.nx;
    std::vector<uint32_t> codes((n+3)/4,0);
    for(int p=0;p<n;++p)codes[p/4]|=uint32_t((p*13+c*7)%8)<<(8*(p%4));
    for(int code=0;code<8;++code)tables[c][code]=.03f+.002f*(code+c);
    CHECK(cudaMalloc(&device_codes[c],codes.size()*sizeof(uint32_t)));
    CHECK(cudaMemcpy(device_codes[c],codes.data(),codes.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&device_tables[c],8*sizeof(float)));CHECK(cudaMemcpy(device_tables[c],tables[c],8*sizeof(float),cudaMemcpyHostToDevice));
    view.codes[c]=device_codes[c];view.tables[c]=device_tables[c];view.table_sizes[c]=8;view.code_ranks[c]=1;
    view.h_decay[c]=device_hc+c;view.h_source[c]=device_hc+3+c;
  }
  std::array<int,Ntx*Nty> masks{};masks[0]=8;masks[Ntx+1]=5;
  int bounds[2]={3,7};int *device_masks,*device_bounds;
  CHECK(cudaMalloc(&device_masks,sizeof(masks)));CHECK(cudaMalloc(&device_bounds,sizeof(bounds)));
  CHECK(cudaMemcpy(device_masks,masks.data(),sizeof(masks),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(device_bounds,bounds,sizeof(bounds),cudaMemcpyHostToDevice));
  reference_step(fields,expected,hc,tables,view.inv);reference_step(fields,expected,hc,tables,view.inv);
  CHECK(cudaFuncSetAttribute(MstepPackedColumnCore<32,16,1,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,MstepTile<32,16>::packed_shared));
  // Direct kernel launch: deliberately no native host launch helper or graph.
  MstepPackedColumnCore<32,16,1,false><<<dim3(Ntx,Nty,segment?(Nz+segment-1)/segment:1),MstepTile<32,16>::threads,MstepTile<32,16>::packed_shared>>>(view,device_masks,device_bounds,Ntx,segment);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  size_t owned=0,mismatch=0,guard_writes=0,input_changes=0;
  for(int c=0;c<6;++c) {
    auto& f=fields[c];std::vector<uint32_t> actual(f.guarded.size()),input_after(f.guarded.size());
    CHECK(cudaMemcpy(actual.data(),f.out_raw,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(input_after.data(),f.in_raw,input_after.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<actual.size();++i) {
      input_changes+=input_after[i]!=f.guarded[i];
      if(i<Guard||i>=actual.size()-Guard){guard_writes+=actual[i]!=Sentinel;continue;}
      const int p=int(i)-Guard,z=p/(f.ny*f.nx),y=p/f.nx%f.ny,x=p%f.nx;
      const int quadrant=(y%16>=8?2:0)+(x%32>=16?1:0);
      const bool is_owned=z>=bounds[0]&&z<bounds[1]&&(masks[(y/16)*Ntx+x/32]&(1<<quadrant));
      if(!is_owned){guard_writes+=actual[i]!=Sentinel;continue;}
      ++owned;uint32_t reference;std::memcpy(&reference,&expected[c][p],4);
      if(actual[i]!=reference) {
        if(mismatch<8)std::fprintf(stderr,"field%d z%d y%d x%d actual0x%08x reference0x%08x\n",c,z,y,x,actual[i],reference);
        ++mismatch;
      }
    }
  }
  std::printf("{\"scope\":\"direct frozen production packed core only, no graphs/JAX/CPML/DFT\",\"segment\":%d,\"owned_values\":%zu,\"bit_mismatches\":%zu,\"guard_writes\":%zu,\"input_changes\":%zu}\n",segment,owned,mismatch,guard_writes,input_changes);
  for(auto& f:fields){CHECK(cudaFree(f.in_raw));CHECK(cudaFree(f.out_raw));}
  for(int c=0;c<3;++c){CHECK(cudaFree(device_codes[c]));CHECK(cudaFree(device_tables[c]));}
  CHECK(cudaFree(device_hc));CHECK(cudaFree(device_masks));CHECK(cudaFree(device_bounds));
  return mismatch||guard_writes||input_changes||owned!=9216?3:0;
}
""".replace("@UPDATE@", json.dumps(str(update_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build",
        type=Path,
        required=True,
        help="Frozen candidate directory containing source/ and manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nvcc", type=Path, default=Path(shutil.which("nvcc") or "/nonexistent/nvcc")
    )
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    build = args.build.resolve()
    metadata = json.loads((build / "manifest.json").read_text())
    source_root = build / "source"
    digest = digest_tree(source_root)
    if digest != metadata["source_sha256"]:
        parser.error("Frozen candidate source hash differs from its build manifest")
    # Configure arguments alone omit defaults selected by the frozen CMake
    # source. Recover effective options, so a default-enabled native build
    # cannot silently become an all-off probe through header fallbacks.
    cache = build / "build/CMakeCache.txt"
    if not cache.is_file():
        parser.error("Frozen build lacks CMakeCache.txt with effective options")
    false_values = {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"}
    options = {}
    for line in cache.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.startswith("BEAMZ_CUDA_MSTEP_") and key.endswith(":BOOL"):
            normalized = value.upper()
            options[key.removesuffix(":BOOL")] = int(
                normalized not in false_values and not normalized.endswith("-NOTFOUND")
            )
    if not options:
        parser.error("Frozen build has no effective mstep compile options")
    definitions = [f"-D{name}={value}" for name, value in sorted(options.items())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "direct_mstep.cu"
    binary = args.output_dir / "direct_mstep"
    source.write_text(harness_source(source_root / "src/update.cu"))
    report = {
        "schema": "beamz.direct-frozen-packed-core.v1",
        "frozen_source_sha256": digest,
        "native_binary_sha256": metadata.get("binary_sha256"),
        "scope": "actual production packed core; direct launch; no graph/JAX or edge/CPML/DFT validation",
        "source": str(source.resolve()),
        "harness_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cmake_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "effective_mstep_options": options,
        "definitions": definitions,
    }
    if args.compile:
        command = [
            str(args.nvcc),
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "--expt-relaxed-constexpr",
            "--relocatable-device-code=true",
            "-lineinfo",
            "-arch=sm_86",
            "-Xcompiler=-ffp-contract=off,-ffunction-sections,-fdata-sections",
            "-Xlinker=--gc-sections",
            *definitions,
            str(source),
            "-o",
            str(binary),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (args.output_dir / "build.log").write_text(result.stdout + result.stderr)
        report.update(compile_command=command, exit_code=result.returncode)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        result.check_returncode()
        report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
