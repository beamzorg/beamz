// Phase-local register ownership with direct shared-family ghost imports.
__host__ __device__ __forceinline__ int PhaseSlot(int z,int y,int x){
 z&=(CZ-1);y&=(CY-1);x&=(CX-1);x^=y|((z&1)<<4);return (z*CY+y)*CX+x;
}
struct PhasePoint{int z,y,x;};
constexpr int PhaseGhostPoints=PAD-BZ*BY*BX;
__host__ __device__ __forceinline__ PhasePoint PhaseGhostLogical(int i){
 if(i<CY*CX)return {0,i/CX,i%CX};i-=CY*CX;
 if(i<BZ*CX)return {1+i/CX,0,i%CX};i-=BZ*CX;
 return {1+i/BY,1+i%BY,0};
}
template<int T>__host__ __device__ __forceinline__ PhasePoint PhasePrivateLogical(int thread,int j){
 return {thread/(CY*CX)+j*(T/(CY*CX)),(thread/CX)&(CY-1),thread&(CX-1)};
}
__host__ __device__ __forceinline__ PhasePoint PhaseGhostPhysicalSlot(int i,int n){
 const auto p=PhaseGhostLogical(i);return {(p.z-n-1)&(CZ-1),(p.y-n-1)&(CY-1),(p.x-n-1)&(CX-1)};
}
template<int Depth,int T,bool Electric>
__device__ __forceinline__ void PhaseScatterGhost(const Parameters&p,float*frame,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<PhaseGhostPoints;i+=T){const auto d=PhaseGhostLogical(i),s=PhaseGhostPhysicalSlot(i,n);
  const auto v=ImportPacket<Depth,false,true>(p,bases,n,oz,oy,ox,d.z,d.y,d.x);const int slot=Slot(s.z,s.y,s.x);
  if constexpr(Electric){frame[slot]=v.ex;frame[PAD+slot]=v.ey;frame[2*PAD+slot]=v.ez;}
  else{frame[slot]=v.hx;frame[PAD+slot]=v.hy;frame[2*PAD+slot]=v.hz;}
 }
}
__device__ __forceinline__ float PhaseReloadShared(const float*pointer){
 float value;asm volatile("ld.volatile.shared.f32 %0, [%1];":"=f"(value):"r"(unsigned(__cvta_generic_to_shared(pointer))):"memory");return value;
}

template<int Depth,int CubeN,int T=512>
__device__ __forceinline__ void PhaseRebindingBody(const Parameters&p){
 static_assert(CZ==8&&CY==16&&CX==32&&T==512,"First prototype preserves native geometry and doubles resident CTA opportunities");
 constexpr int Slots=PAD/T,ZStep=T/(CY*CX);
 extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);
 volatile int*window=reinterpret_cast<volatile int*>(bases+8);
 const int tile=closed_pipeline::Tile(),bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX;const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=closed_pipeline::Owner(p,tile);p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();if(!threadIdx.x)closed_pipeline::Started(p);
 if(!threadIdx.x){
  window[0]=max(1,max(oz-p.phase[0].own[0].z,max(oy-p.phase[0].own[0].y,ox-p.phase[0].own[1].x)));
  window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
 }
 __syncthreads();if(window[0]>window[1])return;
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx,owner=closed_pipeline::Owner(p,producer);
   if(bits&&__ldcg(p.owner_tags+owner)!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/(CY*CX),first=window[0],last=window[1],prev=first-1;
 float state[3][Slots];__syncthreads();closed_pipeline::Skipped(prev);closed_pipeline::WaitParents(p,prev);if(closed_pipeline::Aborted())return;
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;FieldPacket v{};
  if(first==1){const int z=oz+(zs==CZ-1?-1:zs),y=oy+(sy==CY-1?-1:sy),x=ox+(sx==CX-1?-1:sx);
   v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
  }else{const int hz=(-prev-1)&(CZ-1),hy=(-prev-1)&(CY-1),hx=(-prev-1)&(CX-1);
   if(zs==hz||sy==hy||sx==hx)v=ImportPacket<Depth,false,true>(p,bases,prev,oz,oy,ox,(zs-hz)&(CZ-1),(sy-hy)&(CY-1),(sx-hx)&(CX-1));
  }
  state[0][j]=v.hx;state[1][j]=v.hy;state[2][j]=v.hz;const int slot=Slot(zs,sy,sx);
  frame[slot]=v.ex;frame[PAD+slot]=v.ey;frame[2*PAD+slot]=v.ez;
 }
 __syncthreads();const uint8_t flags=p.flags[tile];
 #pragma unroll 1
 for(int n=first;n<=last;++n){
  // BEAMZ_PHASE_H_REGION
  const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&(CX-1),low=-n-1;
  closed_pipeline::WaitParents(p,n);if(closed_pipeline::Aborted())return;
  __syncthreads(); // R0: all old-E stencil readers have retired. Centers remain thread-owned.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs!=hz&&sy!=hy&&sx!=hx){
   const int dz=(zs-hz)&(CZ-1),dy=(sy-hy)&(CY-1),dx=(sx-hx)&(CX-1),z=oz+low+dz,y=oy+low+dy,x=ox+low+dx,surface=Surface(dz-1,dy-1,dx-1);
   #pragma unroll
   for(int c=0;c<3;++c){const int slot=c*PAD+Slot(zs,sy,sx);const float old_e=frame[slot],new_h=state[c][j];frame[slot]=new_h;
    if(surface>=0&&n<Depth)p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=new_h;
    if(n==Depth){const auto&f=FinalField(p,c);if(Contains(f,z,y,x))f.data[z*f.plane+y*f.row+x]=new_h;}
    state[c][j]=old_e; // The same register now holds old E, never both families.
   }
  }}
  PhaseScatterGhost<Depth,T,false>(p,frame,bases,n,oz,oy,ox);
  __syncthreads(); // B2: disjoint owned/ghost stores publish the H family.
  // H register definitions are now dead; E math reads H from shared.
  // BEAMZ_PHASE_E_REGION
  if(n<last){
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs==hz||sy==hy||sx==hx){
    #pragma unroll
    for(int c=0;c<3;++c)state[c][j]=PhaseReloadShared(frame+c*PAD+Slot(zs,sy,sx));
   }}
   __syncthreads(); // B3: E math and ghost-H reloads release the H frame.
  }
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs!=hz&&sy!=hy&&sx!=hx){
   const int dz=(zs-hz)&(CZ-1),dy=(sy-hy)&(CY-1),dx=(sx-hx)&(CX-1),z=oz+low+dz,y=oy+low+dy,x=ox+low+dx,surface=Surface(dz-1,dy-1,dx-1);
   #pragma unroll
   for(int c=0;c<3;++c){
    const int slot=c*PAD+Slot(zs,sy,sx);const float new_e=state[c][j];
    if(n<last){const float new_h=PhaseReloadShared(frame+slot);frame[slot]=new_e;state[c][j]=new_h;}
    if(surface>=0&&n<Depth)p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=new_e;
    if(n==Depth){const auto&f=FinalField(p,c+3);if(Contains(f,z,y,x))f.data[z*f.plane+y*f.row+x]=new_e;}
   }
  }}
  if(n<last){PhaseScatterGhost<Depth,T,true>(p,frame,bases,n,oz,oy,ox);__syncthreads();} // B4 publishes E; its register lifetime ends.
  closed_pipeline::Publish(n);
 }
}
