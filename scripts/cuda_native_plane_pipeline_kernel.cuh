// Included inside moving_native after all native point bodies and launchers.
template<int Depth,int Axes>
__device__ __forceinline__ void PlaneNoIO(const Parameters&p,bool uniform){
 if(uniform){
  if(p.physical_n[0]==512&&p.physical_n[1]==512&&p.physical_n[2]==512)FixedCubeCachedBody<Depth,512,Axes,true>(p,nullptr);
  else if(p.physical_n[0]==564&&p.physical_n[1]==564&&p.physical_n[2]==564)FixedCubeCachedBody<Depth,564,Axes,true>(p,nullptr);
  else UniformBody<Depth,1,false,true,1,true,Axes,true,true,1024>(p,nullptr);
 }else UniformBody<Depth,1,false,true,1,false,-1,false,true,1024>(p,nullptr);
}
template<int Depth,int Axes>
__device__ __forceinline__ void PlaneSourceCpml(const Parameters&p){
 SourcePendingPairedBody<Depth,1,false,true,5,true,1024,20,true,Axes,true,true>(p,nullptr);
}
template<int Depth,int Policy>
__device__ __forceinline__ void PlaneSource(const Parameters&p,bool uniform,int axes){
 if(uniform){
  if constexpr(Policy==5){
   switch(axes){case 2:PlaneSourceCpml<Depth,1>(p);return;case 4:PlaneSourceCpml<Depth,2>(p);return;case 8:PlaneSourceCpml<Depth,4>(p);return;default:PlaneSourceCpml<Depth,-1>(p);return;}
  }else SourcePendingPairedBody<Depth,1,false,true,Policy,true,1024,20,true,-1,false,true>(p,nullptr);
 }else SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20,true,-1,false,true>(p,nullptr);
}
template<int Depth>
__device__ __forceinline__ void PlaneDispatch(const Parameters&p){
 const int tile=plane_pipeline::Tile();const unsigned flags=p.flags[tile];const int policy=beamz::cuda::moving::Policy(flags);const bool uniform=flags&64u;
 switch(policy){
  case 0:if(uniform)UniformBody<Depth,1,true,true,0,true,-1,false,true,1024>(p,nullptr);
         else UniformBody<Depth,1,true,true,0,false,-1,false,true,1024>(p,nullptr);return;
  case 1:switch(flags&14u){case 2:PlaneNoIO<Depth,1>(p,uniform);return;case 4:PlaneNoIO<Depth,2>(p,uniform);return;case 8:PlaneNoIO<Depth,4>(p,uniform);return;default:PlaneNoIO<Depth,-1>(p,uniform);return;}
  case 2:if(uniform)UniformBody<Depth,1,false,true,2,true,-1,false,true,1024>(p,nullptr);
         else UniformBody<Depth,1,false,true,2,false,-1,false,true,1024>(p,nullptr);return;
  case 3:PlaneSource<Depth,3>(p,uniform,flags&14u);return;
  case 4:PlaneSource<Depth,4>(p,uniform,flags&14u);return;
  case 5:PlaneSource<Depth,5>(p,uniform,flags&14u);return;
 }
 if(!threadIdx.x)plane_pipeline::Abort(2048);
}
__global__ void AsymmetricPlaneReset(BeamzNativePlanePipelineWorkspace q,const int32_t*bad){
 const int tile=blockIdx.x*blockDim.x+threadIdx.x;
 if(tile<q.tile_count){const int z=tile/(q.ny*q.nx),y=(tile/q.nx)%q.ny,x=tile%q.nx;
  q.pending[tile]=(z>0)+(y>0)+(x>0);q.claimed[tile]=0;q.ready[tile]=0;q.entry_ready[tile]=0;
 }
 if(tile<q.nz)q.plane_done[tile]=0;
 if(!tile){*q.head=0;*q.tail=1;*q.abort=(*bad!=0);*q.completed=0;*q.started=0;*q.exited=0;
  q.queue[0]=0;q.entry_ready[0]=1;q.claimed[0]=1;
 }
}
template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricPlaneStep(const __grid_constant__ Parameters p,
 const __grid_constant__ BeamzNativePlanePipelineWorkspace q){
 using namespace plane_pipeline;
 if(!threadIdx.x){auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=p.bad;cold.tile=-1;cold.depth=Depth;cold.abort_local=0;cold.parent_ready=0;}
 __syncthreads();
 for(;;){
  if(!threadIdx.x)Pop();__syncthreads();if(C().tile<0)break;
  PlaneDispatch<Depth>(p);
  __syncthreads();
  if(!threadIdx.x&&!Aborted())Complete();
  __syncthreads();
 }
 __syncthreads();
 if(!threadIdx.x){
  ::cuda::atomic_ref<int32_t,::cuda::thread_scope_device>exited(*q.exited);
  const int previous=exited.fetch_add(1,::cuda::memory_order_acq_rel);
  if(previous==q.worker_count-1&&!Acquire(q.abort)){
   if(Acquire(q.completed)!=q.tile_count||Acquire(q.started)!=q.tile_count)Abort(512);
  }
 }
}
template<int Depth>
inline cudaError_t LaunchPlanePipeline(cudaStream_t stream,const Parameters&p,const BeamzNativePlanePipelineWorkspace&q){
 static_assert(plane_pipeline::ColdOffset>=ResearchSourceSharedBytes<5>,"Cold queue must not alias native profiles or source scratch");
 auto status=cudaFuncSetAttribute(AsymmetricPlaneStep<Depth>,cudaFuncAttributeMaxDynamicSharedMemorySize,plane_pipeline::SharedBytes);if(status!=cudaSuccess)return status;
 AsymmetricPlaneStep<Depth><<<q.worker_count,1024,plane_pipeline::SharedBytes,stream>>>(p,q);return cudaGetLastError();
}
