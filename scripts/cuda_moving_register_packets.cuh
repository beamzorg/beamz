// Isolated register-cache field-only probe: coalesced pure-X face transfers.
// Low and high staging are distinct. B1 publishes low imports; B3/B4 publish
// high exports. B1 of the next iteration retires old high readers before any
// new high writer (after that iteration's E stage). No new CTA barrier.
#pragma once
namespace moving_register_packets {
using namespace moving_register_cache;
constexpr int Points=BZ*BY;
constexpr size_t SharedBytes=moving_register_cache::SharedBytes+12ULL*Points*sizeof(float);
__device__ __forceinline__ unsigned Support(Geometry g,int z,int y,int x){
 if(z<0||y<0||x<0||z>g.n||y>g.n||x>g.n)return 0;
 const bool iz=z<g.n,iy=y<g.n,ix=x<g.n;
 return (iz&&iy)|((iz&&ix)<<1)|((iy&&ix)<<2)|(ix<<3)|(iy<<4)|(iz<<5);
}
template<int Steps>
__device__ __forceinline__ void StageLow(float*stage,const float*trail,const size_t*bases,
 Geometry g,int version,int oz,int oy,int ox,int*bad){
 for(int i=threadIdx.x;i<Points;i+=THREADS){const int dz=1+i/BY,dy=1+i%BY;
  const unsigned mask=Support(g,oz-1+dz-version,oy-1+dy-version,ox-1-version);
  const int surface=Surface(dz-1,dy-1,BX-1);const size_t base=bases[4];
  const bool usable=surface>=0&&base!=static_cast<size_t>(-1);
  if(mask&&!usable)atomicAdd(bad,1);
  #pragma unroll
  for(int c=0;c<6;++c)stage[c*Points+i]=!(mask&(1u<<c))?0.f:!usable?__int_as_float(0x7fffffff):trail[base+((version-1)*6+c)*S+surface];
 }
}
__device__ __forceinline__ void RetireLow(const float*stage,float(&h)[3][Slots],float(&e)[3][Slots],int version,int sy,int sx){
 const int hz=(-version-1)&(CZ-1),hy=(-version-1)&(CY-1),hx=(-version-1)&(CX-1),dy=(sy-hy)&(CY-1);
 if(sx==hx&&sy!=hy){
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/(CX*CY)+j*(THREADS/(CX*CY));if(zslot!=hz){const int dz=(zslot-hz)&(CZ-1),key=(dz-1)*BY+dy-1;
   #pragma unroll
   for(int c=0;c<3;++c){h[c][j]=stage[c*Points+key];e[c][j]=stage[(c+3)*Points+key];}
  }}
 }
}
__device__ __forceinline__ void StageHigh(float*stage,const float(&h)[3][Slots],const float(&e)[3][Slots],int version,int sy,int sx){
 const int hz=(-version-1)&(CZ-1),hy=(-version-1)&(CY-1),hx=(-version-1)&(CX-1),dy=(sy-hy)&(CY-1);
 if(sx==((hx+BX)&(CX-1))&&dy>0&&dy<BY){
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/(CX*CY)+j*(THREADS/(CX*CY)),dz=(zslot-hz)&(CZ-1);
   if(dz>0&&dz<BZ){const int key=(dz-1)*BY+dy-1;
    #pragma unroll
    for(int c=0;c<3;++c){stage[c*Points+key]=h[c][j];stage[(c+3)*Points+key]=e[c][j];}
   }
  }
 }
}
template<int Steps>
__device__ __forceinline__ void PublishHigh(const float*stage,float*trail,float*output,const size_t*bases,Geometry g,int version,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<(BZ-1)*(BY-1);i+=THREADS){const int lz=i/(BY-1),ly=i%(BY-1),key=lz*BY+ly;
  const int surface=BY*BX+(BZ-1)*BX+i,z=oz+lz-version,y=oy+ly-version,x=ox+BX-1-version;
  #pragma unroll
  for(int c=0;c<6;++c){const float value=stage[c*Points+key];trail[bases[0]+((version-1)*6+c)*S+surface]=value;
   if(version==Steps&&Valid(c,z,y,x,g.n))output[Offset(c,z,y,x,g.n)]=value;
  }
 }
}
}
