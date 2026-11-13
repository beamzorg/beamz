// Small cache: permanent shared Ey, private Ex/Ez pending, separate Hx/Hy snapshots.
// Canonical point math and full-warp Hy(-X) participation are unchanged.
template<int Depth,bool Safe,int T>
__device__ __forceinline__ void StagePartialH(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ImportPacket<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy,0);
  stage[i]=v.hx;stage[LowXPoints+i]=v.hy;stage[2*LowXPoints+i]=v.hz;
 }
}
template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0>
__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){
 constexpr int T=BlockT?BlockT:((Layout==1&&Policy==1&&IsotropicH&&UniformE)?1024:512);
 constexpr int XY=CY*CX,ZGroups=T>XY?T/XY:1,YGroups=XY>T?XY/T:1;
 constexpr int LocalZ=CZ/ZGroups,Slots=LocalZ*YGroups;
 static_assert(CZ==8&&CY==16&&CX==32&&(T==512||T==1024),"Small shared-Ey mapping only");
 constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 extern __shared__ float frame[];
 float*ey_shared=frame+2*PAD;
 float*snapshots=frame+3*PAD;
 size_t*bases=reinterpret_cast<size_t*>(snapshots+2048);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
 volatile int*window=phase_step+1;
 float*low_stage=reinterpret_cast<float*>(bases+8)+3;
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if constexpr(!Fast){
  if(!threadIdx.x){const int mz=p.phase[0].own[0].z+1,my=p.phase[0].own[0].y+1,mx=p.phase[0].own[1].x+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
  __syncthreads();if(window[0]>window[1])return;
 }
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&(CY-1),zbase=(threadIdx.x/XY)*LocalZ;
 float h[3][Slots],pending_ex[YGroups],pending_ez[YGroups];
 const auto get_ey=[&] __device__(int zs,int y,int x,int){return ey_shared[Slot(zs,y,x)];};
 const auto set_ey=[&] __device__(int zs,int y,int,float value){ey_shared[Slot(zs,y,sx)]=value;};
 __syncthreads();
 const int first=Fast?1:window[0],prev=first-1;
 #pragma unroll
 for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX);
  #pragma unroll
  for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j;FieldPacket v{};
   if(first==1){const int z=oz+(zs==CZ-1?-1:zs),y=oy+(ys==CY-1?-1:ys),x=ox+(sx==31?-1:sx);
    v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
   }else{const int hz=(-prev-1)&(CZ-1),hy=(-prev-1)&(CY-1),hx=(-prev-1)&31;
    if(zs==hz||ys==hy||sx==hx)v=ImportPacket<Depth,false>(p,bases,prev,oz,oy,ox,(zs-hz)&(CZ-1),(ys-hy)&(CY-1),(sx-hx)&31);
   }
   h[0][si]=v.hx;h[1][si]=v.hy;h[2][si]=v.hz;set_ey(zs,ys,si,v.ey);
   frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;
  }
 }
 if(!threadIdx.x)*phase_step=first;__syncthreads();
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};
 #pragma unroll 1
 for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){
  {const int step=*phase_step,hz=(-step-1)&(CZ-1),hy=(-step-1)&(CY-1),hx=(-step-1)&31,low=-step-1;
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),y=oy+low+((ys-hy)&(CY-1)),x=ox+low+((sx-hx)&31);
    #pragma unroll
    for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j,z=oz+low+((zs-hz)&(CZ-1));
     const float ey_x=get_ey(zs,ys,sx+1,si);
     if(zs!=hz&&ys!=hy&&sx!=hx){
      const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
       if(field<3)return h[field][si];
       if(field==4){if(ddx)return ey_x;if(dz)return get_ey((zs+1)&(CZ-1),ys,sx,si+1);return get_ey(zs,ys,sx,si);}
       return frame[(field==3?0:1)*PAD+Slot(zs+dz,ys+ddy,sx+ddx)];
      };
      #pragma unroll
      for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){
       if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);
       else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);
       else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);
      }h[c][si]=value;}
     }
    }
   }
  }
  const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&31,low=-n-1;
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31;
   #pragma unroll
   for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j;if(zs==hz||ys==hy){
    const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,(zs-hz)&(CZ-1),dy,dx);h[0][si]=v.hx;h[1][si]=v.hy;h[2][si]=v.hz;
   }}
  }
  StagePartialH<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // R0: retire all old E readers; publish low-X H imports.
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31;
   #pragma unroll
   for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j,dz=(zs-hz)&(CZ-1);
    if(sx==hx&&ys!=hy&&zs!=hz){const int key=(dz-1)*BY+dy-1;h[0][si]=low_stage[key];h[1][si]=low_stage[LowXPoints+key];h[2][si]=low_stage[2*LowXPoints+key];}
    if(zs==hz||ys==hy||sx==hx){const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,dz,dy,dx);frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;set_ey(zs,ys,si,v.ey);}
    if constexpr(ZGroups==2)if(j==LocalZ-1){const int key=(zbase/LocalZ)*XY+ys*CX+sx;snapshots[key]=h[0][si];snapshots[1024+key]=h[1][si];}
   }
  }
  #pragma unroll
  for(int j=0;j<LocalZ;++j){const int zs=zbase+j;
   float ex[YGroups],ez[YGroups];
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),si=gy*LocalZ+j;ex[gy]=frame[Slot(zs,ys,sx)];ez[gy]=frame[PAD+Slot(zs,ys,sx)];frame[Slot(zs,ys,sx)]=h[0][si];frame[PAD+Slot(zs,ys,sx)]=h[2][si];}
   __syncthreads(); // Pj: publish current H XY, retire all previous-plane H XY readers.
   if(j){
    #pragma unroll
    for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX);frame[Slot(zs-1,ys,sx)]=pending_ex[gy];frame[PAD+Slot(zs-1,ys,sx)]=pending_ez[gy];}
   }
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),si=gy*LocalZ+j,snapshot_key=((zbase/LocalZ)^1)*XY+ys*CX+sx;
    const float hy_x=__shfl_sync(0xffffffffu,h[1][si],(sx+31)&31);
    const int dz=(zs-hz)&(CZ-1),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31,z=oz+low+dz,y=oy+low+dy,x=ox+low+dx;
    if(zs!=hz&&ys!=hy&&sx!=hx){
     const auto read=[&] __device__(int field,int,int ddz,int ddy,int ddx){
      if(field>=3)return field==3?ex[gy]:field==4?get_ey(zs,ys,sx,si):ez[gy];
      if(!ddz&&!ddy&&!ddx)return h[field][si];
      if(ddz){if(j)return h[field][si-1];if constexpr(ZGroups==2)return snapshots[field*1024+snapshot_key];else return h[field][si+LocalZ-1];}
      if(field==1)return hy_x;
      return frame[(field==0?0:1)*PAD+Slot(zs,ys+ddy,sx+ddx)];
     };
     ex[gy]=env.Valid(3,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x):0.f;
     const float new_ey=env.Valid(4,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x):0.f;set_ey(zs,ys,si,new_ey);
     ez[gy]=env.Valid(5,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x):0.f;
    }
    // Private pending belongs only to this thread/Y group. Separate boundary
    // snapshots stay immutable throughout E and are published by P0.
    pending_ex[gy]=ex[gy];pending_ez[gy]=ez[gy];
    if(zs!=hz&&ys!=hy&&sx!=hx){const int surface=Surface(dz-1,dy-1,dx-1);
     #pragma unroll
     for(int c=0;c<3;++c){const float ev=c==0?ex[gy]:c==1?get_ey(zs,ys,sx,si):ez[gy];
      if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][si];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;}
      if(n==Depth){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][si];if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=ev;}
     }
    }
   }
  }
  __syncthreads(); // RL: retire last-plane H XY readers before final pending restore.
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX);frame[Slot(zbase+LocalZ-1,ys,sx)]=pending_ex[gy];frame[PAD+Slot(zbase+LocalZ-1,ys,sx)]=pending_ez[gy];}
  if(!threadIdx.x)*phase_step=n+1;
  __syncthreads(); // RE: publish all current E and retire scratch readers for next H.
 }
}
