// Coalesced pure low-X imports. Shared fields scatter; private Hy/Ey retire
// after the existing B2 via a separate 2*BZ*BY staging region.
#pragma once
namespace moving_asymmetric_low_x {
constexpr int Points=BZ*BY;
constexpr size_t SharedBytes=moving_asymmetric_cache::SharedBytes+2ULL*Points*sizeof(float);
__device__ __forceinline__ void Stage(float*shared,float*stage,const float*trail,const size_t*bases,
 Geometry g,int version,int oz,int oy,int ox,int*bad){
 using namespace moving_asymmetric_cache;
 const int head=WrapZ(-version-1),yh=(-version-1)&15,xh=(-version-1)&31;
 for(int i=threadIdx.x;i<Points;i+=THREADS){
  const int dz=1+i/BY,dy=1+i%BY;
  const int z=oz-1+dz-version,y=oy-1+dy-version,x=ox-1-version;
  const bool bounded=z>=0&&y>=0&&x>=0&&z<=g.n&&y<=g.n&&x<=g.n;
  const bool iz=z<g.n,iy=y<g.n,ix=x<g.n;
  const unsigned mask=!bounded?0u:((iz&&iy?1u:0u)|(iz&&ix?2u:0u)|(iy&&ix?4u:0u)|(ix?8u:0u)|(iy?16u:0u)|(iz?32u:0u));
  const int surface=Surface(dz-1,dy-1,BX-1);const size_t base=bases[4];
  const bool usable=surface>=0&&base!=static_cast<size_t>(-1);
  if(mask&&!usable)atomicAdd(bad,1);
  int j=head+dz;if(j>=CZ)j-=CZ;const int sy=(yh+dy)&15;
  #pragma unroll
  for(int c=0;c<6;++c){const float value=!(mask&(1u<<c))?0.f:!usable?__int_as_float(0x7fffffff):trail[base+((version-1)*6+c)*S+surface];
   if(c==1)stage[i]=value;else if(c==4)stage[Points+i]=value;else At(shared,c,j,sy,xh)=value;
  }
 }
}
__device__ __forceinline__ void Retire(const float*stage,float(&hy)[moving_asymmetric_cache::Slots],
 float(&ey)[moving_asymmetric_cache::Slots],int version,int sy,int sx){
 using namespace moving_asymmetric_cache;
 const int head=WrapZ(-version-1),yh=(-version-1)&15,xh=(-version-1)&31;
 if(sx==xh&&sy!=yh){const int dy=(sy-yh)&15;
  #pragma unroll
  for(int j=0;j<Slots;++j)if(j!=head){int dz=j-head;if(dz<0)dz+=CZ;const int index=(dz-1)*BY+dy-1;hy[j]=stage[index];ey[j]=stage[Points+index];}
 }
}
}
namespace moving_asymmetric_low_x {
__device__ __forceinline__ void StageHigh(float*stage,const float(&hy)[moving_asymmetric_cache::Slots],
 const float(&ey)[moving_asymmetric_cache::Slots],int version,int sy,int sx){
 using namespace moving_asymmetric_cache;
 const int head=WrapZ(-version-1),yh=(-version-1)&15,xh=(-version-1)&31,dy=(sy-yh)&15;
 if(sx==((xh+BX)&31)&&dy>0&&dy<BY){
  #pragma unroll
  for(int j=0;j<Slots;++j){int dz=j-head;if(dz<0)dz+=CZ;if(dz>0&&dz<BZ){const int key=(dz-1)*BY+dy-1;stage[key]=hy[j];stage[Points+key]=ey[j];}}
 }
}
template<int Steps>
__device__ __forceinline__ void PublishHigh(const float*shared,const float*stage,float*trail,
 float*output,const size_t*bases,Geometry g,int version,int oz,int oy,int ox){
 using namespace moving_asymmetric_cache;
 const int head=WrapZ(-version-1),yh=(-version-1)&15,xh=(-version-1)&31;
 for(int i=threadIdx.x;i<(BZ-1)*(BY-1);i+=THREADS){const int lz=i/(BY-1),ly=i%(BY-1),key=lz*BY+ly;
  int j=head+lz+1;if(j>=CZ)j-=CZ;const int sy=(yh+ly+1)&15,sx=(xh+BX)&31;
  const int surface=BY*BX+(BZ-1)*BX+i;
  const int z=oz+lz-version,y=oy+ly-version,x=ox+BX-1-version;
  #pragma unroll
  for(int c=0;c<6;++c){float value;
   if(c==1)value=stage[key];else if(c==4)value=stage[Points+key];
   else value=At(const_cast<float*>(shared),c,j,sy,sx);
   trail[bases[0]+((version-1)*6+c)*S+surface]=value;
   if(version==Steps&&Valid(c,z,y,x,g.n))output[Offset(c,z,y,x,g.n)]=value;
  }
 }
}
}
