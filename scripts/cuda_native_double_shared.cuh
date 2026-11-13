// Both families fit in shared memory after eliminating permanent low-X scratch.
// Same-family Yee reads are thread-owned centers; opposite-family reads may be
// neighbors. In-place own-family writes therefore need only one phase join.
template<int T>
__device__ __forceinline__ void DoubleStageProfiles(const Parameters&p,float*ab){
 for(int i=threadIdx.x;i<576;i+=T){const int slot=i/48,kind=(i/24)&1,packed=i%24;const auto&psi=p.phase[slot/6].psi[slot%6];ab[i]=__ldg((kind?psi.b:psi.a)+packed);}
}
template<int Depth,int T,bool Electric>
__device__ __forceinline__ void DoubleScatterGhost(const Parameters&p,float*frame,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<PhaseGhostPoints;i+=T){const auto d=PhaseGhostLogical(i),s=PhaseGhostPhysicalSlot(i,n);
  const auto v=ImportPacket<Depth,false,true>(p,bases,n,oz,oy,ox,d.z,d.y,d.x);const int slot=Slot(s.z,s.y,s.x);
  if constexpr(Electric){frame[3*PAD+slot]=v.ex;frame[4*PAD+slot]=v.ey;frame[5*PAD+slot]=v.ez;}
  else{frame[slot]=v.hx;frame[PAD+slot]=v.hy;frame[2*PAD+slot]=v.hz;}
 }
}
template<int Depth,int T,int Family>
__device__ __forceinline__ void DoublePublishFamily(const Parameters&p,const float*frame,const size_t*bases,int n,int oz,int oy,int ox){
 constexpr int Slots=PAD/T,ZStep=T/(CY*CX);const int sx=threadIdx.x&(CX-1),sy=(threadIdx.x/CX)&(CY-1),zbase=threadIdx.x/(CY*CX);
 const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&(CX-1),low=-n-1;
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs!=hz&&sy!=hy&&sx!=hx){
  const int dz=(zs-hz)&(CZ-1),dy=(sy-hy)&(CY-1),dx=(sx-hx)&(CX-1),z=oz+low+dz,y=oy+low+dy,x=ox+low+dx,surface=Surface(dz-1,dy-1,dx-1);
  if(surface>=0&&n<Depth){
   #pragma unroll
   for(int c=0;c<3;++c)p.trails[bases[0]+((n-1)*6+Family*3+c)*SurfacePoints+surface]=PhaseReloadShared(frame+(Family*3+c)*PAD+Slot(zs,sy,sx));
  }
  if(n==Depth){
   #pragma unroll
   for(int c=0;c<3;++c){const auto&f=FinalField(p,Family*3+c);if(Contains(f,z,y,x))f.data[z*f.plane+y*f.row+x]=PhaseReloadShared(frame+(Family*3+c)*PAD+Slot(zs,sy,sx));}
  }
 }}
}
template<int Depth,int CubeN,int T=1024>
__device__ __forceinline__ void DoubleSharedBody(const Parameters&p){
 constexpr int Slots=PAD/T,ZStep=T/(CY*CX);
 extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+6*PAD);
 volatile int*window=reinterpret_cast<volatile int*>(bases+8);float*profiles=reinterpret_cast<float*>(bases+8)+2;
 const int tile=closed_pipeline::Tile(),bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX;const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=closed_pipeline::Owner(p,tile);p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();if(!threadIdx.x)closed_pipeline::Started(p);
 if(!threadIdx.x){window[0]=max(1,max(oz-p.phase[0].own[0].z,max(oy-p.phase[0].own[0].y,ox-p.phase[0].own[1].x)));window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));}
 __syncthreads();if(window[0]>window[1])return;
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx,owner=closed_pipeline::Owner(p,producer);
   if(bits&&__ldcg(p.owner_tags+owner)!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&(CX-1),sy=(threadIdx.x/CX)&(CY-1),zbase=threadIdx.x/(CY*CX),first=window[0],last=window[1],prev=first-1;const uint8_t flags=p.flags[tile];
 if(flags&14u)DoubleStageProfiles<T>(p,profiles);
 __syncthreads();closed_pipeline::Skipped(prev);closed_pipeline::WaitParents(p,prev);if(closed_pipeline::Aborted())return;
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;FieldPacket v{};
  if(first==1){const int z=oz+(zs==CZ-1?-1:zs),y=oy+(sy==CY-1?-1:sy),x=ox+(sx==CX-1?-1:sx);
   v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
  }else{const int hz=(-prev-1)&(CZ-1),hy=(-prev-1)&(CY-1),hx=(-prev-1)&(CX-1);
   if(zs==hz||sy==hy||sx==hx)v=ImportPacket<Depth,false,true>(p,bases,prev,oz,oy,ox,(zs-hz)&(CZ-1),(sy-hy)&(CY-1),(sx-hx)&(CX-1));
  }
  const int slot=Slot(zs,sy,sx);frame[slot]=v.hx;frame[PAD+slot]=v.hy;frame[2*PAD+slot]=v.hz;frame[3*PAD+slot]=v.ex;frame[4*PAD+slot]=v.ey;frame[5*PAD+slot]=v.ez;
 }
 __syncthreads();
 #pragma unroll 1
 for(int n=first;n<=last;++n){
  // BEAMZ_DOUBLE_H_REGION
  DoublePublishFamily<Depth,T,0>(p,frame,bases,n,oz,oy,ox);
  closed_pipeline::WaitParents(p,n);if(closed_pipeline::Aborted())return;
  DoubleScatterGhost<Depth,T,false>(p,frame,bases,n,oz,oy,ox);
  __syncthreads(); // H publication and retirement of every old-E stencil reader.
  // BEAMZ_DOUBLE_E_REGION
  DoublePublishFamily<Depth,T,1>(p,frame,bases,n,oz,oy,ox);
  if(n<last)DoubleScatterGhost<Depth,T,true>(p,frame,bases,n,oz,oy,ox);
  // This writer-fence/CTAjoin also publishes shared E and retires all H readers.
  closed_pipeline::Publish(n);
 }
}
