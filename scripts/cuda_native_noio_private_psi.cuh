// Native rank scheduler; one phase register bank plus four CPML history colors.
constexpr size_t NativeNoioPsiSharedBytes=3ULL*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)+864*sizeof(float);
static_assert(NativeNoioPsiSharedBytes<=ProfileCachedSharedBytes,"Retain original NoIO shared allocation and carveout");
template<int Depth,int T,bool Electric>
__device__ __forceinline__ void NativeNoioPsiGhost(const Parameters&p,float*frame,const size_t*bases,int n,int oz,int oy,int ox){
 constexpr int GhostPoints=PAD-BZ*BY*BX;
 for(int i=threadIdx.x;i<GhostPoints;i+=T){int index=i,z,y,x;
  if(index<CY*CX){z=0;y=index/CX;x=index%CX;}
  else{index-=CY*CX;if(index<BZ*CX){z=1+index/CX;y=0;x=index%CX;}else{index-=BZ*CX;z=1+index/BY;y=1+index%BY;x=0;}}
  const auto v=ImportPacket<Depth,false,true>(p,bases,n,oz,oy,ox,z,y,x);const int slot=Slot(z-n-1,y-n-1,x-n-1);
  if constexpr(Electric){frame[slot]=v.ex;frame[PAD+slot]=v.ey;frame[2*PAD+slot]=v.ez;}
  else{frame[slot]=v.hx;frame[PAD+slot]=v.hy;frame[2*PAD+slot]=v.hz;}
 }
}
// BEAMZ_PRIVATE_PSI_POINT

template<int Depth,int CubeN,int ActiveAxes>
__device__ __forceinline__ void NativeNoioPrivatePsiBody(const Parameters&p,const int32_t*tiles){
 static_assert(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4,"Only single-axis independent local recurrences");
 constexpr int T=1024,Slots=4,ZStep=2;
 extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);volatile int*window=reinterpret_cast<volatile int*>(bases+8);float*profiles=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+2;
 StageIndependentCpmlProfiles(p,profiles);
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx,oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);
  window[0]=max(1,max(oz-p.phase[0].own[0].z,max(oy-p.phase[0].own[0].y,ox-p.phase[0].own[1].x)));
  window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
 }
 __syncthreads();const int first=window[0],last=window[1];if(first>last)return;
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx,owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 __syncthreads();const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/512,prev=first-1;
 float state[3][Slots];PrivatePsiHistory<true,Slots>history;
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;FieldPacket v{};
  if(first==1){const int z=oz+(zs==CZ-1?-1:zs),y=oy+(sy==CY-1?-1:sy),x=ox+(sx==CX-1?-1:sx);
   v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
  }else{const int hz=(-prev-1)&7,hy=(-prev-1)&15,hx=(-prev-1)&31;if(zs==hz||sy==hy||sx==hx)v=ImportPacket<Depth,false,true>(p,bases,prev,oz,oy,ox,(zs-hz)&7,(sy-hy)&15,(sx-hx)&31);}
  state[0][j]=v.hx;state[1][j]=v.hy;state[2][j]=v.hz;const int slot=Slot(zs,sy,sx);frame[slot]=v.ex;frame[PAD+slot]=v.ey;frame[2*PAD+slot]=v.ez;
 }
 __syncthreads();const uint8_t flags=p.flags[tile];
 auto env=[&] __device__(){if constexpr(CubeN>0)return FixedCubeCachedEnvironment<CubeN,ActiveAxes>{p,flags,profiles};else return Environment<1,false,true,1,true,-1,false,ActiveAxes,true>{p,flags,profiles};}();
 #pragma unroll 1
 for(int n=first;n<=last;++n){
  // BEAMZ_NOIO_PSI_H_REGION
  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1,dy=(sy-hy)&15,dx=(sx-hx)&31;
  __syncthreads();
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs!=hz&&sy!=hy&&sx!=hx){const int dz=(zs-hz)&7,z=oz+low+dz,y=oy+low+dy,x=ox+low+dx,surface=Surface(dz-1,dy-1,dx-1);
   #pragma unroll
   for(int c=0;c<3;++c){const int slot=c*PAD+Slot(zs,sy,sx);const float old_e=frame[slot],new_h=state[c][j];frame[slot]=new_h;state[c][j]=old_e;
    if(surface>=0&&n<Depth)p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=new_h;
    if(n==Depth){const auto&f=FinalField(p,c);if(Contains(f,z,y,x))f.data[z*f.plane+y*f.row+x]=new_h;}
   }
  }}
  NativeNoioPsiGhost<Depth,T,false>(p,frame,bases,n,oz,oy,ox);__syncthreads();
  // BEAMZ_NOIO_PSI_E_REGION
  if(n<last){
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs==hz||sy==hy||sx==hx){
    #pragma unroll
    for(int c=0;c<3;++c)state[c][j]=frame[c*PAD+Slot(zs,sy,sx)];
   }}
  }
  __syncthreads();
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs!=hz&&sy!=hy&&sx!=hx){const int dz=(zs-hz)&7,z=oz+low+dz,y=oy+low+dy,x=ox+low+dx,surface=Surface(dz-1,dy-1,dx-1);
   #pragma unroll
   for(int c=0;c<3;++c){const int slot=c*PAD+Slot(zs,sy,sx);const float new_e=state[c][j];
    if(n<last){const float new_h=frame[slot];frame[slot]=new_e;state[c][j]=new_h;}
    if(surface>=0&&n<Depth)p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=new_e;
    if(n==Depth){const auto&f=FinalField(p,c+3);if(Contains(f,z,y,x))f.data[z*f.plane+y*f.row+x]=new_e;}
   }
  }}
  if(n<last)NativeNoioPsiGhost<Depth,T,true>(p,frame,bases,n,oz,oy,ox);__syncthreads();
 }
}
