#!/usr/bin/env python3
"""Generate and run a temporally blocked CUDA kernel from traced cell equations.

Synthetic periodic, equal-extent, scalar-coefficient fields ONLY. This is a
compiler correctness/resource probe, not the BeamZ application benchmark.
The intentionally simple full-box shared layout is a baseline for a future
liveness-driven rolling-plane generator. No runtime dispatch is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

try:
    from scripts.cuda_stencil_trace import emit_cuda, trace_yee_step
except ModuleNotFoundError:
    from cuda_stencil_trace import emit_cuda, trace_yee_step


def resource_estimate(tile, depth):
    if len(tile) != 3 or any(type(v) is not int or v < 1 for v in tile):
        raise ValueError("tile must contain three positive integers (z,y,x)")
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    extents = tuple(t + 2 * depth for t in tile)
    cells = extents[0] * extents[1] * extents[2]
    useful = tile[0] * tile[1] * tile[2]
    updates = sum(
        (extents[0] - 2 * k - a) * (extents[1] - 2 * k - a) * (extents[2] - 2 * k - a)
        for k in range(depth)
        for a in (1, 2)
    )
    return {
        "tile_zyx": tile,
        "depth": depth,
        "shared_bytes": 6 * cells * 4,
        "redundant_phase_update_ratio": updates / (2 * depth * useful),
    }


def generate_cuda(tile=(4, 4, 16), depths=(1, 2, 4), *, layout="box"):
    for depth in depths:
        resource_estimate(tile, depth)
    if 1 not in depths:
        raise ValueError("depth one is required as the GPU staging reference")
    if layout not in {"box", "rolling"}:
        raise ValueError("layout must be box or rolling")
    field_ids = {
        kind + c: i for i, (kind, c) in enumerate((k, c) for k in "HE" for c in "xyz")
    }

    def read_name(read):
        z, y, x = read.offset
        return (
            f"s[{field_ids[read.field]} * volume + p + ({z})*sx*sy + ({y})*sx + ({x})]"
        )

    assignments = []
    for update in trace_yee_step():
        assignments.append(
            f"s[{field_ids[update.output.field]} * volume + p] = {emit_cuda(update.value, read_name)};"
        )
    h_updates = "\n".join(assignments[:3])
    e_updates = "\n".join(assignments[3:])
    scalar_parameters = (
        "\n".join(
            f"const float {kind}_{field}={value}f;"
            for field in field_ids
            for kind, value in (
                ("source", ".07" if field.startswith("H") else ".09"),
                ("decay", ".999" if field.startswith("H") else ".998"),
            )
        )
        + "\nconst float inv_spacing=1.f;"
    )

    def global_read(read):
        z, y, x = read.offset
        return f"a[{field_ids[read.field]} * n + (wrap(z+({z}),nz)*ny+wrap(y+({y}),ny))*nx+wrap(x+({x}),nx)]"

    global_kernels = []
    for phase in range(2):
        updates = trace_yee_step()[phase * 3 : (phase + 1) * 3]
        statements = "\n".join(
            f"a[{field_ids[u.output.field]}*n+p]={emit_cuda(u.value, global_read)};"
            for u in updates
        )
        global_kernels.append(
            f"__global__ void sweep_{phase}(float* a,int nx,int ny,int nz) {{"
            "int n=nx*ny*nz,p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=n)return;"
            "int x=p%nx,y=p/nx%ny,z=p/(nx*ny);" + scalar_parameters + statements + "}"
        )
    launch_cases = "\n".join(
        f"case {d}: launch<{d}>(a,b,nx,ny,nz); break;" for d in depths
    )
    run_cases = "run(0, initial, reference, nx,ny,nz,steps,samples);\n" + "\n".join(
        f"run({d}, initial, reference, nx,ny,nz,steps,samples);" for d in depths
    )
    tz, ty, tx = tile
    source = (
        r"""
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(2); } } while(0)
constexpr int TX=@TX@, TY=@TY@, TZ=@TZ@;
__host__ __device__ int wrap(int p,int n) { int q=p%n; return q<0?q+n:q; }
@GLOBAL_KERNELS@
template<int D> __global__ void box(const float* in,float* out,int nx,int ny,int nz) {
  constexpr int sx=TX+2*D, sy=TY+2*D, sz=TZ+2*D, volume=sx*sy*sz;
  extern __shared__ float s[];
  const int ox=blockIdx.x*TX,oy=blockIdx.y*TY,oz=blockIdx.z*TZ;
  const int n=nx*ny*nz;
  for(int q=threadIdx.x;q<6*volume;q+=blockDim.x) {
    int p=q%volume, x=p%sx,y=p/sx%sy,z=p/(sx*sy),c=q/volume;
    int g=(wrap(oz+z-D,nz)*ny+wrap(oy+y-D,ny))*nx+wrap(ox+x-D,nx);
    s[q]=in[c*n+g];
  }
  __syncthreads();
  const float inv_spacing=1.f;
  const float source_Hx=.07f, source_Hy=.07f, source_Hz=.07f;
  const float source_Ex=.09f, source_Ey=.09f, source_Ez=.09f;
  const float decay_Hx=.999f, decay_Hy=.999f, decay_Hz=.999f;
  const float decay_Ex=.998f, decay_Ey=.998f, decay_Ez=.998f;
  #pragma unroll
  for(int k=0;k<D;++k) {
    int wx=sx-2*k-1,wy=sy-2*k-1,wz=sz-2*k-1;
    for(int i=threadIdx.x;i<wx*wy*wz;i+=blockDim.x) {
      int x=i%wx+k,y=i/wx%wy+k,z=i/(wx*wy)+k,p=(z*sy+y)*sx+x;
      @H_UPDATES@
    }
    __syncthreads();
    wx=sx-2*k-2; wy=sy-2*k-2; wz=sz-2*k-2;
    for(int i=threadIdx.x;i<wx*wy*wz;i+=blockDim.x) {
      int x=i%wx+k+1,y=i/wx%wy+k+1,z=i/(wx*wy)+k+1,p=(z*sy+y)*sx+x;
      @E_UPDATES@
    }
    __syncthreads();
  }
  for(int i=threadIdx.x;i<TX*TY*TZ;i+=blockDim.x) {
    int x=i%TX,y=i/TX%TY,z=i/(TX*TY);
    if(ox+x<nx && oy+y<ny && oz+z<nz) {
      int p=((z+D)*sy+y+D)*sx+x+D,g=((oz+z)*ny+oy+y)*nx+ox+x;
      for(int c=0;c<6;++c) out[c*n+g]=s[c*volume+p];
    }
  }
}
template<int D> void launch(const float* a,float* b,int nx,int ny,int nz) {
  constexpr int bytes=24*(TX+2*D)*(TY+2*D)*(TZ+2*D);
  // This executable uses one device. Configure each specialization before
  // timing; repeated driver attribute calls would distort launch-heavy cases.
  static const bool configured=[] {
    int dev,limit; CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
    if(bytes>limit) throw std::runtime_error("box exceeds device shared memory limit");
    CHECK(cudaFuncSetAttribute(box<D>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    return true;
  }();
  (void)configured;
  box<D><<<dim3((nx+TX-1)/TX,(ny+TY-1)/TY,(nz+TZ-1)/TZ),256,bytes>>>(a,b,nx,ny,nz);
  CHECK(cudaGetLastError());
}
float* advance(int depth,float* a,float* b,int nx,int ny,int nz,int steps) {
  if(depth==0) {
    for(int step=0;step<steps;++step) {
      sweep_0<<<(nx*ny*nz+255)/256,256>>>(a,nx,ny,nz);
      sweep_1<<<(nx*ny*nz+255)/256,256>>>(a,nx,ny,nz);
      CHECK(cudaGetLastError());
    }
    return a;
  }
  for(int done=0;done<steps;) {
    int d=steps-done>=depth?depth:1;
    switch(d) { @LAUNCH_CASES@ }
    std::swap(a,b); done+=d;
  }
  return a;
}
// Independent scalar equations: intentionally do not call the traced function.
void cpu_step(std::vector<float>& a,int nx,int ny,int nz) {
  const int n=nx*ny*nz;
  auto at=[&](int c,int z,int y,int x) { return a[c*n+(wrap(z,nz)*ny+wrap(y,ny))*nx+wrap(x,nx)]; };
  auto result=a;
  for(int z=0;z<nz;++z) for(int y=0;y<ny;++y) for(int x=0;x<nx;++x) {
    int p=(z*ny+y)*nx+x;
    float cx=(at(5,z,y+1,x)-at(5,z,y,x))-(at(4,z+1,y,x)-at(4,z,y,x));
    float cy=(at(3,z+1,y,x)-at(3,z,y,x))-(at(5,z,y,x+1)-at(5,z,y,x));
    float cz=(at(4,z,y,x+1)-at(4,z,y,x))-(at(3,z,y+1,x)-at(3,z,y,x));
    result[p]=std::fma(-.07f,cx,.999f*a[p]);
    result[n+p]=std::fma(-.07f,cy,.999f*a[n+p]);
    result[2*n+p]=std::fma(-.07f,cz,.999f*a[2*n+p]);
  }
  a.swap(result); result=a;
  for(int z=0;z<nz;++z) for(int y=0;y<ny;++y) for(int x=0;x<nx;++x) {
    int p=(z*ny+y)*nx+x;
    float cx=(at(2,z,y,x)-at(2,z,y-1,x))-(at(1,z,y,x)-at(1,z-1,y,x));
    float cy=(at(0,z,y,x)-at(0,z-1,y,x))-(at(2,z,y,x)-at(2,z,y,x-1));
    float cz=(at(1,z,y,x)-at(1,z,y,x-1))-(at(0,z,y,x)-at(0,z,y-1,x));
    result[3*n+p]=std::fma(.09f,cx,.998f*a[3*n+p]);
    result[4*n+p]=std::fma(.09f,cy,.998f*a[4*n+p]);
    result[5*n+p]=std::fma(.09f,cz,.998f*a[5*n+p]);
  }
  a.swap(result);
}
void run(int depth,const std::vector<float>& initial,const std::vector<float>& reference,int nx,int ny,int nz,int steps,int samples) {
  size_t bytes=initial.size()*sizeof(float); float *a,*b;
  CHECK(cudaMalloc(&a,bytes)); CHECK(cudaMalloc(&b,bytes));
  std::vector<float> actual(initial.size()); std::vector<float> times;
  cudaEvent_t begin,end; CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
  float* final=nullptr;
  for(int sample=-2;sample<samples;++sample) {
    CHECK(cudaMemcpy(a,initial.data(),bytes,cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(begin));
    final=advance(depth,a,b,nx,ny,nz,steps);
    CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
    float ms; CHECK(cudaEventElapsedTime(&ms,begin,end)); if(sample>=0) times.push_back(ms);
  }
  CHECK(cudaMemcpy(actual.data(),final,bytes,cudaMemcpyDeviceToHost));
  float error=0; size_t mismatches=0;
  for(size_t i=0;i<actual.size();++i) {
    if(!std::isfinite(actual[i])) throw std::runtime_error("nonfinite generated field");
    error=std::max(error,std::abs(actual[i]-reference[i]));
    mismatches+=std::memcmp(&actual[i],&reference[i],4)!=0;
  }
  if(error>2e-6f) throw std::runtime_error("generated CUDA disagrees with independent CPU equations");
  std::sort(times.begin(),times.end()); float ms=times[times.size()/2];
  std::printf("{\"depth\":%d,\"median_ms\":%.9g,\"gcups\":%.9g,\"max_abs_error\":%.9g,\"cpu_bit_mismatches\":%zu}\n",depth,ms,double(nx)*ny*nz*steps/(ms*1e6),error,mismatches);
  CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end)); CHECK(cudaFree(a)); CHECK(cudaFree(b));
}
int main(int argc,char** argv) {
  try {
    int nx=argc>1?std::atoi(argv[1]):37,ny=argc>2?std::atoi(argv[2]):29,nz=argc>3?std::atoi(argv[3]):23;
    int steps=argc>4?std::atoi(argv[4]):9,samples=argc>5?std::atoi(argv[5]):5;
    if(nx<1||ny<1||nz<1||steps<1||samples<1||int64_t(nx)*ny*nz>INT32_MAX/6) return 2;
    std::vector<float> initial(size_t(nx)*ny*nz*6); uint32_t rng=42;
    for(auto& v:initial) { rng=rng*1664525u+1013904223u; v=(int(rng>>8)-8388608)*1e-8f; }
    auto reference=initial;
    for(int step=0;step<steps;++step) cpu_step(reference,nx,ny,nz);
    @RUN_CASES@
  } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 2; }
}
""".replace("@TX@", str(tx))
        .replace("@TY@", str(ty))
        .replace("@TZ@", str(tz))
        .replace("@H_UPDATES@", h_updates)
        .replace("@E_UPDATES@", e_updates)
        .replace("@LAUNCH_CASES@", launch_cases)
        .replace("@RUN_CASES@", run_cases)
        .replace("@GLOBAL_KERNELS@", "\n".join(global_kernels))
    )
    if layout == "rolling":
        try:
            from scripts.cuda_stencil_rolling import generate_rolling_kernel
        except ModuleNotFoundError:
            from cuda_stencil_rolling import generate_rolling_kernel
        declarations = "template<int D> __global__ void rolling_dispatch(const float*,float*,int,int,int);\n"
        kernels = "\n".join(
            generate_rolling_kernel(tile, d).replace(
                f"__global__ void rolling_{d}(",
                f"template<> __global__ void rolling_dispatch<{d}>(",
            )
            for d in depths
        )
        source = (
            source.replace(
                "template<int D> void launch(",
                declarations + kernels + "\n"
                "template<int D> constexpr int rolling_shared_bytes() {\n"
                "int total=0; for(int k=0;k<2*D-1;++k) total+=(TX+2*D-1-k)*(TY+2*D-1-k);\n"
                "return 24*total; }\n"
                "template<int D> void launch(",
            )
            .replace(
                "constexpr int bytes=24*(TX+2*D)*(TY+2*D)*(TZ+2*D);",
                "constexpr int bytes=rolling_shared_bytes<D>();",
            )
            .replace("box<D>", "rolling_dispatch<D>")
        )
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument(
        "--arch", default="native", help="NVCC architecture, e.g. native or sm_86"
    )
    parser.add_argument(
        "--tile", type=int, nargs=3, default=(4, 4, 16), metavar=("Z", "Y", "X")
    )
    parser.add_argument(
        "--shape", type=int, nargs=3, default=(23, 29, 37), metavar=("Z", "Y", "X")
    )
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--layout", choices=("box", "rolling"), default="box")
    args = parser.parse_args()
    if any(v < 1 for v in args.shape) or args.steps < 1 or args.samples < 1:
        parser.error("shape, steps, and samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "traced_box.cu"
    source.write_text(generate_cuda(tuple(args.tile), layout=args.layout))
    report = {
        "scope": "synthetic periodic equal-extent scalar-coefficient fields; not production FDTD throughput",
        "shape_zyx": args.shape,
        "steps": args.steps,
        "samples": args.samples,
        "layout": args.layout,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "depth_zero_control": "two global-memory kernels per timestep, same equations",
        "resources": [resource_estimate(tuple(args.tile), d) for d in (1, 2, 4)],
    }
    if args.layout == "rolling":
        try:
            from scripts.cuda_stencil_rolling import rolling_resource_estimate
        except ModuleNotFoundError:
            from cuda_stencil_rolling import rolling_resource_estimate
        report["resources"] = [
            rolling_resource_estimate(tuple(args.tile), d) for d in (1, 2, 4)
        ]
    if not args.generate_only:
        if not args.nvcc:
            parser.error("nvcc is required unless --generate-only is selected")
        binary = args.output_dir / "traced_box"
        command = [
            args.nvcc,
            "-std=c++17",
            "-O3",
            "--fmad=false",
            "-Xcompiler=-ffp-contract=off",
            f"-arch={args.arch}",
            str(source),
            "-o",
            str(binary),
        ]
        subprocess.run(command, check=True, timeout=180)
        nz, ny, nx = args.shape
        result = subprocess.run(
            [
                str(binary.resolve()),
                str(nx),
                str(ny),
                str(nz),
                str(args.steps),
                str(args.samples),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        report["results"] = [json.loads(line) for line in result.stdout.splitlines()]
        report["compile_command"] = command
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
