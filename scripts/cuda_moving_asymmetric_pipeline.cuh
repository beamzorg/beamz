// Family-split asymmetric cache. Two existing data barriers per version.
#pragma once
#include <cuda_pipeline.h>
#ifndef MOVING_ASYMMETRIC_ASYNC
#define MOVING_ASYMMETRIC_ASYNC 0
#endif
namespace moving_asymmetric_pipeline {
using namespace moving_asymmetric_cache;
constexpr size_t SharedBytes=moving_asymmetric_cache::SharedBytes+sizeof(int);
// Batch all three components: transform owner, surface and validity once.
template<int Family,bool Async>
__device__ __forceinline__ void ImportFamily(float*shared,float&yvalue,const float*trail,
 const size_t*bases,Geometry g,int version,int oz,int oy,int ox,int j,int sy,int sx,
 int dz,int dy,int dx,int*bad){
 const int pz=oz-1+dz-version,py=oy-1+dy-version,px=ox-1+dx-version;
 const bool bounded=pz>=0&&py>=0&&px>=0&&pz<=g.n&&py<=g.n&&px<=g.n;
 const bool iz=pz<g.n,iy=py<g.n,ix=px<g.n;
 const unsigned mask=!bounded?0u:Family?((ix?1u:0u)|(iy?2u:0u)|(iz?4u:0u)):
 ((iz&&iy?1u:0u)|(iz&&ix?2u:0u)|(iy&&ix?4u:0u));
 const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);
 const int surface=Surface(dz?dz-1:BZ-1,dy?dy-1:BY-1,dx?dx-1:BX-1);
 const size_t base=bases[neighbor];
 const bool usable=neighbor&&surface>=0&&base!=static_cast<size_t>(-1);
 if(mask&&!usable)atomicAdd(bad,1);
 #pragma unroll
 for(int c=0;c<3;++c){
  const bool valid=(mask&(1u<<c))!=0;
  if(c==1){yvalue=!valid?0.f:!usable?__int_as_float(0x7fffffff):trail[base+((version-1)*6+Family*3+c)*S+surface];}
  else {float*destination=&At(shared,Family*3+c,j,sy,sx);
   if(!valid)*destination=0.f;
   else if(!usable)*destination=__int_as_float(0x7fffffff);
   else {const float*source=trail+base+((version-1)*6+Family*3+c)*S+surface;
    if constexpr(Async)__pipeline_memcpy_async(destination,source,sizeof(float));else *destination=*source;
   }
  }
 }
}
}
template<int Steps> __global__ __launch_bounds__(THREADS,MOVING_MIN_CTAS)
void AsymmetricPipelinedMoving(const int*tiles,Geometry g,const float*initial,
 float*output,float*trail,int*bad){
 using namespace moving_asymmetric_cache;
 using moving_asymmetric_pipeline::ImportFamily;
 MOVING_PROFILE_DECLARE;MOVING_PROFILE_BEGIN();
 extern __shared__ float shared[];
 size_t*bases=reinterpret_cast<size_t*>(shared+4*PAD);
 volatile int*h_counter=reinterpret_cast<volatile int*>(bases+8);
 volatile int*e_counter=h_counter+1;
 const int tile=tiles[blockIdx.x],bz=tile/(g.nx*g.ny),by=(tile/g.nx)%g.ny,bx=tile%g.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX;
#if MOVING_TRAIL_RING
 if(!threadIdx.x)g.owner_tags[TrailOwner(g,tile,bz+by+bx)]=tile;
 __syncthreads();
#endif
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
  size_t base=static_cast<size_t>(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*g.nx*g.ny-dy*g.nx-dx;
   const int owner=TrailOwner(g,producer,bz+by+bx-dz-dy-dx);
#if MOVING_TRAIL_RING
   if(bits&&g.owner_tags[owner]!=producer)atomicAdd(bad,1);else
#endif
   base=static_cast<size_t>(owner)*Steps*6*S;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=threadIdx.x/32;
 float hy[Slots],ey[Slots];
 #pragma unroll
 for(int j=0;j<Slots;++j){const int z=j==CZ-1?-1:j,y=sy==15?-1:sy,x=sx==31?-1:sx;
  hy[j]=GlobalAt(initial,1,oz+z,oy+y,ox+x,g.n);
  ey[j]=GlobalAt(initial,4,oz+z,oy+y,ox+x,g.n);
  #pragma unroll
  for(int family=0;family<2;++family){At(shared,family*3,j,sy,sx)=GlobalAt(initial,family*3,oz+z,oy+y,ox+x,g.n);At(shared,family*3+2,j,sy,sx)=GlobalAt(initial,family*3+2,oz+z,oy+y,ox+x,g.n);}
 }

 if(!threadIdx.x){*h_counter=1;*e_counter=1;}
 __syncthreads();MOVING_PROFILE_MARK(Init);
 #pragma unroll 1
 for(int n=1;n<=Steps;++n){
  MOVING_PROFILE_MARK(Loop);
  {const int head=WrapZ(-n-1),yh=(-n-1)&15,xh=(-n-1)&31;
   const int dy=(sy-yh)&15,dx=(sx-xh)&31;
   #pragma unroll
   for(int j=0;j<Slots;++j)if(j==head||sy==yh||sx==xh){const int dz=WrapZ(j-head);
    ImportFamily<0,MOVING_ASYMMETRIC_ASYNC>(shared,hy[j],trail,bases,g,n,oz,oy,ox,j,sy,sx,dz,dy,dx,bad);
   }
   if constexpr(MOVING_ASYMMETRIC_ASYNC)__pipeline_commit();
  }
  MOVING_PROFILE_MARK(Import);
  {const int phase=*h_counter,head=WrapZ(-phase-1),yh=(-phase-1)&15,xh=(-phase-1)&31,low=-phase-1;
   const int py=oy+low+((sy-yh)&15),px=ox+low+((sx-xh)&31);
   MOVING_PROFILE_MARK(HSetup);
   #pragma unroll
   for(int j=0;j<Slots;++j){
    // Full-mask shuffle precedes every owned/physical predicate.
    const float next_ey=__shfl_sync(0xffffffffu,ey[j],(sx+1)&31);
    if(j!=head&&sy!=yh&&sx!=xh){const int pz=oz+low+WrapZ(j-head);
     const auto read=[&] __device__(int field,int,int dz,int dy,int dx){
      if(field<3)return field==1?hy[j]:At(shared,field,j,sy,sx);
      if(field==4){if(dz)return ey[(j+1)%Slots];if(dx)return next_ey;return ey[j];}
      return At(shared,field,j+dz,sy+dy,sx+dx);
     };
     #pragma unroll
     for(int c=0;c<3;++c){const float value=Valid(c,pz,py,px,g.n)?Generated<0>(c,read):0.f;if(c==1)hy[j]=value;else At(shared,c,j,sy,sx)=value;}
    }
   }
  }

  MOVING_PROFILE_MARK(HCompute);
  if constexpr(MOVING_ASYMMETRIC_ASYNC)__pipeline_wait_prior(0);
  __syncthreads(); // B1 retires H old-E reads and previous E publication.
  MOVING_PROFILE_MARK(HWait);
  if(!threadIdx.x)*h_counter=n+1;
  {const int head=WrapZ(-n-1),yh=(-n-1)&15,xh=(-n-1)&31;
   const int dy=(sy-yh)&15,dx=(sx-xh)&31;
   #pragma unroll
   for(int j=0;j<Slots;++j)if(j==head||sy==yh||sx==xh){const int dz=WrapZ(j-head);
    ImportFamily<1,MOVING_ASYMMETRIC_ASYNC>(shared,ey[j],trail,bases,g,n,oz,oy,ox,j,sy,sx,dz,dy,dx,bad);
   }
   if constexpr(MOVING_ASYMMETRIC_ASYNC)__pipeline_commit();
  }
  MOVING_PROFILE_MARK(Import);
  {const int head=WrapZ(-n-1),yh=(-n-1)&15,xh=(-n-1)&31,low=-n-1;
   const int dy=(sy-yh)&15,dx=(sx-xh)&31,py=oy+low+dy,px=ox+low+dx;
   #pragma unroll
   for(int j=0;j<Slots;++j)if(j!=head&&sy!=yh&&sx!=xh){const int dz=WrapZ(j-head),pz=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
    if(surface>=0||n==Steps){
     #pragma unroll
     for(int c=0;c<3;++c){const float value=c==1?hy[j]:At(shared,c+0,j,sy,sx);
      if(surface>=0)trail[bases[0]+((n-1)*6+c+0)*S+surface]=value;
      if(n==Steps&&Valid(c+0,pz,py,px,g.n))output[Offset(c+0,pz,py,px,g.n)]=value;
     }
    }
   }
  }
  MOVING_PROFILE_MARK(Publish);
  {const int phase=*e_counter,head=WrapZ(-phase-1),yh=(-phase-1)&15,xh=(-phase-1)&31,low=-phase-1;
   const int py=oy+low+((sy-yh)&15),px=ox+low+((sx-xh)&31);
   MOVING_PROFILE_MARK(ESetup);
   #pragma unroll
   for(int j=0;j<Slots;++j){
    const float previous_hy=__shfl_sync(0xffffffffu,hy[j],(sx-1)&31);
    if(j!=head&&sy!=yh&&sx!=xh){const int pz=oz+low+WrapZ(j-head);
     const auto read=[&] __device__(int field,int,int dz,int dy,int dx){
      if(field>=3)return field==4?ey[j]:At(shared,field,j,sy,sx);
      if(field==1){if(dz)return hy[(j+Slots-1)%Slots];if(dx)return previous_hy;return hy[j];}
      return At(shared,field,j+dz,sy+dy,sx+dx);
     };
     #pragma unroll
     for(int c=0;c<3;++c){const float value=Valid(c+3,pz,py,px,g.n)?Generated<1>(c,read):0.f;if(c==1)ey[j]=value;else At(shared,c+3,j,sy,sx)=value;}
    }
   }
  }

  MOVING_PROFILE_MARK(ECompute);
  if constexpr(MOVING_ASYMMETRIC_ASYNC)__pipeline_wait_prior(0);
  __syncthreads(); // B2 publishes E and retires H publication / E readers.
  MOVING_PROFILE_MARK(EWait);
  if(!threadIdx.x)*e_counter=n+1;
  {const int head=WrapZ(-n-1),yh=(-n-1)&15,xh=(-n-1)&31,low=-n-1;
   const int dy=(sy-yh)&15,dx=(sx-xh)&31,py=oy+low+dy,px=ox+low+dx;
   #pragma unroll
   for(int j=0;j<Slots;++j)if(j!=head&&sy!=yh&&sx!=xh){const int dz=WrapZ(j-head),pz=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
    if(surface>=0||n==Steps){
     #pragma unroll
     for(int c=0;c<3;++c){const float value=c==1?ey[j]:At(shared,c+3,j,sy,sx);
      if(surface>=0)trail[bases[0]+((n-1)*6+c+3)*S+surface]=value;
      if(n==Steps&&Valid(c+3,pz,py,px,g.n))output[Offset(c+3,pz,py,px,g.n)]=value;
     }
    }
   }
  }
  MOVING_PROFILE_MARK(Publish);

 }
 MOVING_PROFILE_MARK(Loop);MOVING_PROFILE_END(g,tile,bz+by+bx,Steps);
}
