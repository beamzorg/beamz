// Coalesced low-X imports for the two-Y-row Z-pair cache; no high staging.
#pragma once
namespace moving_register_packets {
using namespace moving_register_cache;
constexpr int Points=BZ*BY;
constexpr size_t SharedBytes=moving_register_cache::SharedBytes+6ULL*Points*sizeof(float);
__device__ __forceinline__ unsigned Support(Geometry g,int z,int y,int x){
 if(z<0||y<0||x<0||z>g.n||y>g.n||x>g.n)return 0;
 const bool iz=z<g.n,iy=y<g.n,ix=x<g.n;
 return (iz&&iy)|((iz&&ix)<<1)|((iy&&ix)<<2)|(ix<<3)|(iy<<4)|(iz<<5);
}
template<int Steps,bool Interior=false>
__device__ __forceinline__ void StageLow(float*stage,const float*trail,const size_t*bases,
 Geometry g,int version,int oz,int oy,int ox,int*bad){
 for(int i=threadIdx.x;i<Points;i+=THREADS){const int dz=1+i/BY,dy=1+i%BY;
  unsigned mask=63;if constexpr(!Interior)mask=Support(g,oz-1+dz-version,oy-1+dy-version,ox-1-version);
  const int surface=Surface(dz-1,dy-1,BX-1);const size_t base=bases[4];
  const bool usable=surface>=0&&base!=static_cast<size_t>(-1);
  if(mask&&!usable)atomicAdd(bad,1);
  #pragma unroll
  for(int c=0;c<6;++c)stage[c*Points+i]=!(mask&(1u<<c))?0.f:!usable?__int_as_float(0x7fffffff):trail[base+((version-1)*6+c)*S+surface];
 }
}
__device__ __forceinline__ void RetireLow(const float*stage,float(&h)[3][Slots],float(&e)[3][Slots],int version,int,int sx){
 const int hz=(-version-1)&15,hy=(-version-1)&15,hx=(-version-1)&31;
 if(sx==hx){
  #pragma unroll
  for(int j=0;j<Slots;++j){const int sy=YSlot(j),zs=ZSlot(j);
   if(sy!=hy&&zs!=hz){const int dz=(zs-hz)&15,dy=(sy-hy)&15,key=(dz-1)*BY+dy-1;
    #pragma unroll
    for(int c=0;c<3;++c){h[c][j]=stage[c*Points+key];e[c][j]=stage[(c+3)*Points+key];}
   }
  }
 }
}
}
