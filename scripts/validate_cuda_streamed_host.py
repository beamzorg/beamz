"""Run the ACTUAL generated device body in a CPU SIMT barrier/shuffle shim.

Independent scalar Yee/CPML equations check bitwise fields, dense logical psi,
physical walls, point sources and unique publication. This is not CUDA execution
or packed-CPML indexing validation. A small physical grid keeps CPU threads low.
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

try:
    from scripts.cuda_space_time_codegen import generate
    from scripts.cuda_space_time_ir import StreamShape, yee_operator
except ModuleNotFoundError:
    from cuda_space_time_codegen import generate
    from cuda_space_time_ir import StreamShape, yee_operator


HOST = r"""
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>
#include <algorithm>
#define __device__
#define __forceinline__ inline
struct Warp {std::barrier<> barrier{32};float values[32]{};};
struct Block {std::barrier<> barrier{@THREADS@};std::array<Warp,@WARPS@> warp;};
struct Index {int x=0;};thread_local Index threadIdx;thread_local Block*simt;
inline void __syncthreads(){simt->barrier.arrive_and_wait();}
inline float __shfl_sync(unsigned mask,float value,int source){
 if(mask!=0xffffffffu)std::abort();auto&w=simt->warp[threadIdx.x/32];
 w.values[threadIdx.x%32]=value;w.barrier.arrive_and_wait();float result=w.values[source];w.barrier.arrive_and_wait();return result;
}
inline float __fmul_rn(float a,float b){return a*b;}
inline float __fsub_rn(float a,float b){return a-b;}
inline float __fadd_rn(float a,float b){return a+b;}
inline float __fmaf_rn(float a,float b,float c){return std::fma(a,b,c);}
struct __nv_bfloat16{uint16_t bits;};
inline __nv_bfloat16 __float2bfloat16_rn(float x){uint32_t u=std::bit_cast<uint32_t>(x);u+=0x7fffu+((u>>16)&1u);return{uint16_t(u>>16)};}
inline float __bfloat162float(__nv_bfloat16 x){return std::bit_cast<float>(uint32_t(x.bits)<<16);}
@GENERATED@
constexpr int LocalBase=@LOCAL_BASE@;
constexpr int NZ=2,NY=3,NX=4,T=@DEPTH@,P=(NZ+1)*(NY+1)*(NX+1),Local=@LOCAL@,Metal=@METAL@;
constexpr bool Cpml=@CPML@,Half=@HALF@,Native=@NATIVE@,Inactive=@INACTIVE@,ZeroFields=@ZERO_FIELDS@,UniformH=@UNIFORM_H@;
constexpr int D[6][3]={{NZ,NY,NX+1},{NZ,NY+1,NX},{NZ+1,NY,NX},{NZ+1,NY+1,NX},{NZ+1,NY,NX+1},{NZ,NY+1,NX+1}};
constexpr int LocalField[12]={@LOCAL_FIELDS@};
constexpr int LocalTerm[12]={@LOCAL_TERMS@};
constexpr int TermAxis[6]={1,0,0,2,2,1};
int at(int z,int y,int x){return(z*(NY+1)+y)*(NX+1)+x;}
bool valid(int f,int z,int y,int x){return z>=0&&z<D[f][0]&&y>=0&&y<D[f][1]&&x>=0&&x<D[f][2];}
float coefficient(int kind,int f,int z,int y,int x){
 if(Inactive&&kind>=2)std::abort();
 if(UniformH&&kind<2&&f<3)return kind?0.11f:1.f;
 if(kind<2)return kind?(.03f+.002f*f+.0001f*((z+2*y+3*x)%5)):(.99f-.0003f*(f%3));
 int a=TermAxis[LocalTerm[f-LocalBase]],coord=a==0?z:a==1?y:x;
 return kind==2?(-.03f-.001f*(coord%3)):kind==3?(.91f+.001f*(coord%4)):(.8f+.01f*(coord%5));
}
bool pec(int f,int z,int y,int x){int phase=f/3,c=f%3,normal=2-c,p[]={z,y,x};
 for(int a=0;a<3;++a){if((!phase&&a!=normal)||(phase&&a==normal))continue;
  if((p[a]==0&&(Metal&(1<<(2*a))))||(p[a]==D[f][a]-1&&(Metal&(1<<(2*a+1)))))return true;}return false;
}
float source(int f,int v,int z,int y,int x,float value){
 if((f==1&&z==1&&y==1&&x==1)||(f==4&&z==0&&y==2&&x==1))return std::fma(.002f,float(v),value);return value;
}
struct Data{std::vector<float>field,psi;std::vector<std::atomic<int>>written,psi_written;
 Data():field((T+1)*6*P,0.f),psi(Local*P,0.f),written((T+1)*6*P),psi_written(Local*P){}
};
struct Env {Data&data;int oz,oy,ox;float inv_spacing=.73f;
 bool Valid(int f,int z,int y,int x)const{return valid(f,z,y,x);}
 float Read(int f,int v,int z,int y,int x)const{
  if(!valid(f,z,y,x))return 0.f;int i=(v*6+f)*P+at(z,y,x);
  if(v){bool own=z+v>=oz&&z+v<oz+@TZ@&&y+v>=oy&&y+v<oy+@TY@&&x+v>=ox&&x+v<ox+@TX@;
   if(own||data.written[i].load()!=1){std::fprintf(stderr,"invalid global source f%d v%d p%d,%d,%d own%d count%d\n",f,v,z,y,x,own,data.written[i].load());std::abort();}}
  return data.field[i];
 }
 void ReadPacket(unsigned mask,int family,int v,int z,int y,int x,float&a,float&b,float&c)const{
  if(mask&1)a=Read(3*family,v,z,y,x);if(mask&2)b=Read(3*family+1,v,z,y,x);if(mask&4)c=Read(3*family+2,v,z,y,x);
 }
 float ReadRelative(int f,int v,int rz,int ry,int rx)const{return Read(f,v,oz+rz-v,oy+ry-v,ox+rx-v);}
 void ReadPacketRelative(unsigned mask,int family,int v,int rz,int ry,int rx,float&a,float&b,float&c)const{ReadPacket(mask,family,v,oz+rz-v,oy+ry-v,ox+rx-v,a,b,c);}
 void PublishPacketRelative(int v,int rz,int ry,int rx,float hx,float hy,float hz,float ex,float ey,float ez)const{float values[]={hx,hy,hz,ex,ey,ez};for(int f=0;f<6;++f)Publish(f,v,oz+rz-v,oy+ry-v,ox+rx-v,values[f]);}
 float Coefficient(int kind,int f,int z,int y,int x)const{return coefficient(kind,f,z,y,x);}
 bool LocalActive(int s,int z,int y,int x)const{int axis=TermAxis[LocalTerm[s-LocalBase]],p[]={z,y,x};return !Inactive&&(p[axis]==0||p[axis]==D[LocalField[s-LocalBase]][axis]-1);}
 float Difference(int phase,int f,int axis,int z,int y,int x,float center,float adjacent)const{
  int p[]={z,y,x};if(!phase){if(p[axis]+1>=D[f][axis])return 0.f;return __fmul_rn(__fsub_rn(adjacent,center),inv_spacing);}
  if(p[axis]==0)return Metal&(1<<(2*axis))?__fmul_rn(center,inv_spacing):0.f;
  if(p[axis]==D[f][axis])return Metal&(1<<(2*axis+1))?__fmul_rn(-adjacent,inv_spacing):0.f;
  return __fmul_rn(__fsub_rn(center,adjacent),inv_spacing);
 }
 float LoadLocal(int s,int z,int y,int x)const{if(Inactive)std::abort();return data.psi[(s-LocalBase)*P+at(z,y,x)];}
 void StoreLocal(int s,int z,int y,int x,float v)const{if(Inactive)std::abort();int i=(s-LocalBase)*P+at(z,y,x);data.psi[i]=v;++data.psi_written[i];}
 float Finish(int f,int v,int z,int y,int x,float a)const{return source(f,v,z,y,x,Native&&pec(f,z,y,x)?0.f:a);}
 void Publish(int f,int v,int z,int y,int x,float a)const{
  if(!valid(f,z,y,x))return;int i=(v*6+f)*P+at(z,y,x);data.field[i]=a;
  if(data.written[i].fetch_add(1)!=0){std::fprintf(stderr,"duplicate output\n");std::abort();}
 }
};
float refread(const std::vector<float>&v,int f,int step,int z,int y,int x){return valid(f,z,y,x)?v[(step*6+f)*P+at(z,y,x)]:0.f;}
int main(){Data data;uint32_t rng=77;auto random=[&](){rng=rng*1664525u+1013904223u;return(float(int(rng>>8))-8388608.f)*1e-8f;};
 for(int f=0;f<6;++f)for(int z=0;z<D[f][0];++z)for(int y=0;y<D[f][1];++y)for(int x=0;x<D[f][2];++x)data.field[f*P+at(z,y,x)]=ZeroFields?std::bit_cast<float>(((f+z+y+x)&1)?0x80000000u:0u):random();
 for(int s=0;s<Local;++s)for(int z=0;z<D[LocalField[s]][0];++z)for(int y=0;y<D[LocalField[s]][1];++y)for(int x=0;x<D[LocalField[s]][2];++x){float a=random()*.01f;data.psi[s*P+at(z,y,x)]=Half?__bfloat162float(__float2bfloat16_rn(a)):a;}
 auto reference=data.field,psi_reference=data.psi;
 constexpr int sa[]={2,0,1},sb[]={1,2,0},aa[]={1,0,2},ab[]={0,2,1};
 for(int v=1;v<=T;++v)for(int phase=0;phase<2;++phase)for(int c=0;c<3;++c){int f=phase*3+c;
  for(int z=0;z<D[f][0];++z)for(int y=0;y<D[f][1];++y)for(int x=0;x<D[f][2];++x){
   auto derivative=[&](int component,int axis){int input=(1-phase)*3+component,p[]={z,y,x},neighbor[]={z,y,x};neighbor[axis]+=phase?-1:1;
    int input_step=phase?v:v-1;if(Native){if(!phase&&p[axis]+1>=D[input][axis])return 0.f;
     if(phase&&(p[axis]==0||p[axis]==D[input][axis])&&!(Metal&(1<<(2*axis+(p[axis]!=0)))))return 0.f;}
    float center=refread(reference,input,input_step,z,y,x),other=refread(reference,input,input_step,neighbor[0],neighbor[1],neighbor[2]);
    return(phase?center-other:other-center)*.73f;};
   float d[]={derivative(sa[c],aa[c]),derivative(sb[c],ab[c])};float curl;
   if(Cpml){float corrected[2];for(int t=0;t<2;++t){int term=2*c+t,s=0;while(s<Local&&(LocalField[s]!=f||LocalTerm[s]!=term))++s;if(s==Local)std::abort();
     int i=s*P+at(z,y,x),coord=TermAxis[term]==0?z:TermAxis[term]==1?y:x;
     bool active=!Native||(!Inactive&&(coord==0||coord==D[f][TermAxis[term]]-1));
     if(active){float next=std::fma(coefficient(3,s+LocalBase,z,y,x),psi_reference[i],coefficient(2,s+LocalBase,z,y,x)*d[t]);
      psi_reference[i]=Half?__bfloat162float(__float2bfloat16_rn(next)):next;
      corrected[t]=(t?-1.f:1.f)*std::fma(d[t],coefficient(4,s+LocalBase,z,y,x),next);
     }else corrected[t]=(t?-1.f:1.f)*d[t];}
    curl=corrected[0]+corrected[1];}else curl=d[0]-d[1];
   float result=std::fma(phase?coefficient(1,f,z,y,x):-coefficient(1,f,z,y,x),curl,coefficient(0,f,z,y,x)*refread(reference,f,v-1,z,y,x));
   if(Native&&pec(f,z,y,x))result=0.f;reference[(v*6+f)*P+at(z,y,x)]=source(f,v,z,y,x,result);
  }
 }
 // BEGIN GENERATED EXECUTION
 std::vector<std::array<int,3>>boxes;for(int z=0;z<(NZ+T+@TZ@)/@TZ@;++z)for(int y=0;y<(NY+T+@TY@)/@TY@;++y)for(int x=0;x<(NX+T+@TX@)/@TX@;++x)boxes.push_back({z,y,x});
 std::sort(boxes.begin(),boxes.end(),[](auto a,auto b){int x=a[0]+a[1]+a[2],y=b[0]+b[1]+b[2];return x!=y?x<y:a<b;});
 Block block;std::vector<float>shared(beamz::cuda::streamed_ir::SharedBytes/4);std::vector<std::thread>workers;
 for(int q=0;q<@THREADS@;++q)workers.emplace_back([&,q](){threadIdx.x=q;simt=&block;
  for(auto b:boxes){block.barrier.arrive_and_wait();Env env{data,b[0]*@TZ@,b[1]*@TY@,b[2]*@TX@};
   beamz::cuda::streamed_ir::RunStreamedTile(env,shared.data(),env.oz,env.oy,env.ox);block.barrier.arrive_and_wait();}});
 for(auto&t:workers)t.join();
 // END GENERATED EXECUTION
 int errors=0;uint64_t compared=0;
 for(int v=1;v<=T;++v)for(int f=0;f<6;++f)for(int z=0;z<D[f][0];++z)for(int y=0;y<D[f][1];++y)for(int x=0;x<D[f][2];++x){int i=(v*6+f)*P+at(z,y,x);++compared;
  if(data.written[i]!=1||std::bit_cast<uint32_t>(reference[i])!=std::bit_cast<uint32_t>(data.field[i])){if(errors++<5)std::fprintf(stderr,"field f%d v%d p%d,%d,%d got%.9g expected%.9g writes%d\n",f,v,z,y,x,data.field[i],reference[i],data.written[i].load());}}
 for(int s=0;s<Local;++s)for(int z=0;z<D[LocalField[s]][0];++z)for(int y=0;y<D[LocalField[s]][1];++y)for(int x=0;x<D[LocalField[s]][2];++x){int i=s*P+at(z,y,x);++compared;
  int axis=TermAxis[LocalTerm[s]],coord=axis==0?z:axis==1?y:x;
  int expected_writes=Native&&(Inactive||(coord!=0&&coord!=D[LocalField[s]][axis]-1))?0:T;
  if(data.psi_written[i]!=expected_writes||std::bit_cast<uint32_t>(psi_reference[i])!=std::bit_cast<uint32_t>(data.psi[i])){if(errors++<5)std::fprintf(stderr,"psi%d p%d,%d,%d got%.9g expected%.9g writes%d\n",s,z,y,x,data.psi[i],psi_reference[i],data.psi_written[i].load());}}
 std::printf("{\"result\":\"%s\",\"compared\":%llu,\"errors\":%d}\n",errors?"fail":"pass",(unsigned long long)compared,errors);return errors?1:0;
}
"""


def host_source(result, shape, *, cpml=False, half=False, native=False, metallic=0,
                inactive=False, zero_fields=False, native_cell=False, scalar_h_class=False):
    header = result["source"].replace("#include <cuda_bf16.h>", "")
    ids = result.get("tensor_ids", {})
    if native_cell:
        ids = result["field_ids"] | result["local_term_ids"]
    local = sorted((i, name) for name, i in ids.items() if name.startswith("psi_"))
    fields = [3 * int(name[4] == "E") + int(name[5]) // 2 for _, name in local]
    terms = [int(name[5]) for _, name in local]
    substitutions = {"GENERATED": header, "LOCAL_BASE": "0" if native_cell else "6", "THREADS": str(shape.threads), "WARPS": str(shape.threads // 32),
                     "DEPTH": str(shape.depth), "LOCAL": str(len(local)), "METAL": str(metallic),
                     "CPML": str(cpml).lower(), "HALF": str(half).lower(), "NATIVE": str(native).lower(),
                     "INACTIVE": str(inactive).lower(), "ZERO_FIELDS": str(zero_fields).lower(), "UNIFORM_H": str(scalar_h_class).lower(),
                     "LOCAL_FIELDS": ",".join(map(str, fields)) or "0",
                     "LOCAL_TERMS": ",".join(map(str, terms)) or "0",
                     "TZ": str(shape.tile_zyx[0]), "TY": str(shape.tile_zyx[1]), "TX": str(shape.tile_zyx[2])}
    source = HOST
    if native_cell:
        start = source.index(" // BEGIN GENERATED EXECUTION")
        end = source.index(" // END GENERATED EXECUTION", start) + len(" // END GENERATED EXECUTION")
        calls = "\n".join(f"case {f}:value=beamz::cuda::trace_native::AdvanceNativeCell<{f//3},{f%3}>(env,read,v,z,y,x);break;" for f in range(6))
        execution = r"""
 Env env{data,0,0,0};
 for(int v=1;v<=T;++v)for(int phase=0;phase<2;++phase)for(int c=0;c<3;++c){int f=phase*3+c;
  for(int z=0;z<D[f][0];++z)for(int y=0;y<D[f][1];++y)for(int x=0;x<D[f][2];++x){
   auto read=[&](int field,int version,int dz,int dy,int dx){int zz=z+dz,yy=y+dy,xx=x+dx;
    if(!valid(field,zz,yy,xx))return 0.f;int i=(version*6+field)*P+at(zz,yy,xx);
    if(version&&data.written[i]!=1)std::abort();return data.field[i];};
   float value=0.f;switch(f){@CALLS@}env.Publish(f,v,z,y,x,value);
  }
 }
""".replace("@CALLS@", calls)
        source = source[:start] + execution + source[end:]
    for key, value in substitutions.items():
        source = source.replace("@" + key + "@", value)
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--cpml", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--native-boundaries", action="store_true")
    p.add_argument("--pec-mask", type=int, default=0)
    p.add_argument("--materialize-providers", action="store_true")
    p.add_argument("--metadata-remat", action="store_true")
    p.add_argument("--relative-callbacks", action="store_true")
    p.add_argument("--inactive-cpml", action="store_true")
    p.add_argument("--zero-fields", action="store_true")
    p.add_argument("--loop-stages", action="store_true")
    p.add_argument("--native-cell", action="store_true")
    p.add_argument("--scalar-h-class", action="store_true")
    a = p.parse_args()
    if a.bf16 and not a.cpml:
        p.error("BF16 storage requires CPML")
    compiler = shutil.which("c++")
    if compiler is None:
        p.error("C++20 compiler required")
    shape = StreamShape((2, 2, 3), a.depth, 32 * (a.depth + 1))
    operator = yee_operator(cpml_axes=(0, 1, 2) if a.cpml else None,
                            precision="bfloat16" if a.bf16 else "float32")
    if a.scalar_h_class:
        import numpy as np

        from scripts.cuda_trace_specialize import (
            magnetic_scalar_class,
            specialize_scalar_class,
            verify_scalar_class,
        )
        packet = [None] * 6 + [np.float32(1.)] * 3 + [np.float32(.11)] * 3
        fact = magnetic_scalar_class(packet)
        verify_scalar_class(fact, packet)
        operator, _ = specialize_scalar_class(operator, fact)
    if a.native_cell:
        if a.materialize_providers or a.metadata_remat or a.loop_stages:
            p.error("Native-cell validation has no storage-schedule options")
        try:
            from scripts.generate_cuda_native_cell import generate_native_cell
        except ModuleNotFoundError:
            from generate_cuda_native_cell import generate_native_cell
        result = generate_native_cell(operator)
    else:
        result = generate(operator, shape, materialized=a.materialize_providers,
                          metadata_remat=a.metadata_remat, native_boundaries=a.native_boundaries,
                          relative_callbacks=a.relative_callbacks, loop_stages=a.loop_stages)
    source = host_source(result, shape, cpml=a.cpml, half=a.bf16,
                         native=a.native_boundaries or a.native_cell, metallic=a.pec_mask, inactive=a.inactive_cpml,
                         zero_fields=a.zero_fields, native_cell=a.native_cell, scalar_h_class=a.scalar_h_class)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    path = a.output_dir / "host.cc"
    binary = a.output_dir / "host"
    path.write_text(source)
    command = [compiler, "-std=c++20", "-O1", "-pthread", "-ffp-contract=off", str(path), "-o", str(binary)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (a.output_dir / "build.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    run = subprocess.run([str(binary.resolve())], capture_output=True, text=True, timeout=90)
    (a.output_dir / "run.log").write_text(run.stdout + run.stderr)
    report = {"scope": __doc__, "generated_sha256": result["source_sha256"],
              "host_sha256": hashlib.sha256(source.encode()).hexdigest(), "command": command,
              "options": vars(a) | {"output_dir": str(a.output_dir)}, "run_exit_code": run.returncode,
              "stdout": run.stdout, "stderr": run.stderr}
    (a.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    run.check_returncode()
    print(run.stdout.strip())


if __name__ == "__main__":
    main()
