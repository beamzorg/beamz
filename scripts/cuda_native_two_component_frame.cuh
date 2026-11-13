// Private research storage schedule: Ex/Ez shared; H and Ey fixed scalars.
// The canonical traced point operator is included unchanged by the consumer.
template<int Depth,bool Safe,int T>
__device__ __forceinline__ void StageTwoLow(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ImportPacket<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy,0);
  stage[i]=v.hx;stage[LowXPoints+i]=v.hy;stage[2*LowXPoints+i]=v.hz;
  stage[3*LowXPoints+i]=v.ex;stage[4*LowXPoints+i]=v.ey;stage[5*LowXPoints+i]=v.ez;
 }
}
template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>
__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){
 constexpr int T=(Fast||(Layout==1&&Policy==1&&IsotropicH&&UniformE))?1024:512;
 constexpr int Slots=PAD/T;
 constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 extern __shared__ float frame[];
 float*ey_cross=frame+2*PAD;
 float*h_cross=ey_cross+1024;
 size_t*bases=reinterpret_cast<size_t*>(h_cross+2048);
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
 const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=T==1024?(threadIdx.x/512)*8:0,xy=sy*32+sx;
 float h[3][Slots],ey[Slots];
 __syncthreads();
 const int first=Fast?1:window[0],prev=first-1;
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j;FieldPacket v{};
  if(first==1){const int z=oz+(zs==15?-1:zs),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
   v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
  }else{const int hz=(-prev-1)&15,hy=(-prev-1)&15,hx=(-prev-1)&31;
   if(zs==hz||sy==hy||sx==hx)v=ImportPacket<Depth,false>(p,bases,prev,oz,oy,ox,(zs-hz)&15,(sy-hy)&15,(sx-hx)&31);
  }
  h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;ey[j]=v.ey;
  frame[Slot(zs,sy,sx)]=v.ex;frame[PAD+Slot(zs,sy,sx)]=v.ez;
  if((zs&7)==0)ey_cross[(zs/8)*512+xy]=v.ey;
 }
 if(!threadIdx.x)*phase_step=first;__syncthreads();
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};
 #pragma unroll 1
 for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){
  {const int step=*phase_step,hz=(-step-1)&15,hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;
   const int y=oy+low+((sy-hy)&15),x=ox+low+((sx-hx)&31);
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j,z=oz+low+((zs-hz)&15);
    // Full-mask collectives precede every owned/support predicate.
    const float ey_x=__shfl_sync(0xffffffffu,ey[j],(sx+1)&31);
    if(zs!=hz&&sy!=hy&&sx!=hx){
     const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
      if(field<3)return h[field][j];
      if(field==4){
       if(ddx)return ey_x;
       if(dz){if constexpr(T==1024){if(j==Slots-1)return ey_cross[((zs/8)^1)*512+xy];else return ey[j+1];}
              else{return j==Slots-1?ey[0]:ey[j+1];}}
       return ey[j];
      }
      return frame[(field==3?0:1)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];
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
  const int hz=(-n-1)&15,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;
  const int dy=(sy-hy)&15,dx=(sx-hx)&31;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs==hz||sy==hy){
   const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,(zs-hz)&15,dy,dx);
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;ey[j]=v.ey;
  }}
  StageTwoLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // T1: all old Ex/Ez H readers retire; low imports publish.
  if(sx==hx&&sy!=hy){
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs!=hz){const int key=(((zs-hz)&15)-1)*BY+dy-1;
    h[0][j]=low_stage[key];h[1][j]=low_stage[LowXPoints+key];h[2][j]=low_stage[2*LowXPoints+key];ey[j]=low_stage[4*LowXPoints+key];
   }}
  }
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if((zs&7)==7){const int k=(zs/8)*512+xy;
   h_cross[k]=h[0][j];h_cross[1024+k]=h[1][j];
  }}
  __syncthreads(); // T2: both current H chunk-boundary planes publish.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;
   float ex=frame[Slot(zs,sy,sx)],ez=frame[PAD+Slot(zs,sy,sx)];
   frame[Slot(zs,sy,sx)]=h[0][j];frame[PAD+Slot(zs,sy,sx)]=h[2][j];
   __syncthreads(); // Pj: publish Hx/Hz XY plane; old Ex/Ez center is saved.
   const float hy_x=__shfl_sync(0xffffffffu,h[1][j],(sx+31)&31);
   const int dz=(zs-hz)&15,z=oz+low+dz,y=oy+low+dy,x=ox+low+dx;
   if(zs!=hz&&sy!=hy&&sx!=hx){
    const auto read=[&] __device__(int field,int,int ddz,int ddy,int ddx){
     if(field>=3)return field==3?ex:field==4?ey[j]:ez;
     if(!ddz&&!ddy&&!ddx)return h[field][j];
     if(ddz){if constexpr(T==1024){if(j==0)return h_cross[field*1024+((zs/8)^1)*512+xy];else return h[field][j-1];}
             else{return j==0?h[field][Slots-1]:h[field][j-1];}}
     if(field==1)return hy_x;
     return frame[(field==0?0:1)*PAD+Slot(zs,sy+ddy,sx+ddx)];
    };
    float next_ex=0.f,next_ey=0.f,next_ez=0.f;
    if(env.Valid(3,z,y,x))next_ex=beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x);
    if(env.Valid(4,z,y,x))next_ey=beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x);
    if(env.Valid(5,z,y,x))next_ez=beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x);
    ex=next_ex;ey[j]=next_ey;ez=next_ez;
   }else{
    const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,dz,dy,dx);ex=v.ex;ez=v.ez;
   }
   __syncthreads(); // Qj: all Hx/Hz XY plane readers retire before Ex/Ez restore.
   frame[Slot(zs,sy,sx)]=ex;frame[PAD+Slot(zs,sy,sx)]=ez;
   if(zs!=hz&&sy!=hy&&sx!=hx){const int surface=Surface(dz-1,dy-1,dx-1);
    #pragma unroll
    for(int c=0;c<3;++c){const float evalue=c==0?ex:c==1?ey[j]:ez;
     if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][j];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=evalue;}
     if(n==Depth){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);
      if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
      if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=evalue;
     }
    }
   }
  }
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if((zs&7)==0)ey_cross[(zs/8)*512+xy]=ey[j];}
  if(!threadIdx.x)*phase_step=n+1;
  __syncthreads(); // T3: publish complete Ex/Ez, Ey boundaries and next phase.
 }
}
