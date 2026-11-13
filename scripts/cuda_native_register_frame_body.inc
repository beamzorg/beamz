template<int Depth,bool Safe,int T>
__device__ __forceinline__ void StageRegisterLow(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ImportPacket<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy,0);
  stage[i]=v.hx;stage[LowXPoints+i]=v.hy;stage[2*LowXPoints+i]=v.hz;
  stage[3*LowXPoints+i]=v.ex;stage[4*LowXPoints+i]=v.ey;stage[5*LowXPoints+i]=v.ez;
 }
}
template<int T,int Slots>
__device__ __forceinline__ void RetireRegisterLow(const float*stage,float(&h)[3][Slots],float(&e)[3][Slots],int n,int sy,int sx){
 const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,dy=(sy-hy)&15;
 if(sx==hx&&sy!=hy){
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/512+j*(T/512);if(zslot!=hz){const int dz=(zslot-hz)&7,key=(dz-1)*BY+dy-1;
   #pragma unroll
   for(int c=0;c<3;++c){h[c][j]=stage[c*LowXPoints+key];e[c][j]=stage[(c+3)*LowXPoints+key];}
  }}
 }
}

template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>
__global__ __launch_bounds__((Fast?1024:512),1) void Asymmetric(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 constexpr int T=Fast?1024:512,Slots=PAD/T,ZStep=T/512;
 constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
 float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+1;
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/512;
 float h[3][Slots],e[3][Slots];
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+(zs==7?-1:zs),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
  #pragma unroll
  for(int c=0;c<3;++c){h[c][j]=Initial(p,c,z,y,x);e[c][j]=Initial(p,c+3,z,y,x);frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];}
 }
 if(!threadIdx.x)*phase_step=1;__syncthreads();
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy>env{p,flags};
 #pragma unroll 1
 for(int n=1;n<=Depth;++n){
  {const int step=*phase_step,hz=(-step-1)&7,hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;
   const int dy=(sy-hy)&15,dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);
    if(zs!=hz&&sy!=hy&&sx!=hx){
     const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
      if(field<3)return h[field][j];const int c=field-3;
      if(!dz&&!ddy&&!ddx)return e[c][j];return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];
     };
     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);
     }h[c][j]=value;}
    }
   }
  }
  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;
  const int dy=(sy-hy)&15,dx=(sx-hx)&31;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs==hz||sy==hy){const int dz=(zs-hz)&7;
   const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,dz,dy,dx);
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
  }}
  StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // B1 retires all old-E frame readers and publishes low imports.
  RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]=h[c][j];
  }
  __syncthreads(); // B2 publishes all current H values.
  {const int step=*phase_step,ehz=(-step-1)&7,ehy=(-step-1)&15,ehx=(-step-1)&31,elow=-step-1;
   const int y=oy+elow+((sy-ehy)&15),x=ox+elow+((sx-ehx)&31);
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);
    if(zs!=ehz&&sy!=ehy&&sx!=ehx){
     const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
      if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];
     };
     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c+3,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x);
     }e[c][j]=value;}
    }
   }
  }
  __syncthreads(); // B3 retires current-H frame readers before E overwrite.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];
  }
  if(!threadIdx.x)*phase_step=n+1;
  __syncthreads(); // B4 publishes E and the next phase counter.
  const int y=oy+low+dy,x=ox+low+dx;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs!=hz&&sy!=hy&&sx!=hx){const int dz=(zs-hz)&7,z=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
   #pragma unroll
   for(int c=0;c<3;++c){
    if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][j];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=e[c][j];}
    if(n==Depth){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);
     if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
     if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=e[c][j];
    }
   }
  }}
 }
}
