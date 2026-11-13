// Included after the original native bodies and launchers. Physics policy is
// a kernel template, so the compiler never allocates an all-policy worker.
template<int Depth,int Axes,int CubeN>
__device__ __forceinline__ void ClosedNoIO(const Parameters&p){
 // A captured program certificate covers every Policy1/5 E trajectory,
 // including vacuous empty windows that deliberately lack the old flag64.
 if constexpr(CubeN>0)FixedCubeCachedBody<Depth,CubeN,Axes,true>(p,nullptr);
 else UniformBody<Depth,1,false,true,1,true,Axes,true,true,1024>(p,nullptr);
}
template<int Depth,bool Uniform,int Axes>
__device__ __forceinline__ void ClosedSourceCpml(const Parameters&p){
 SourcePendingPairedBody<Depth,1,false,true,5,Uniform,1024,20,true,Axes,true,true>(p,nullptr);
}
template<int Depth,bool Uniform>
__device__ __forceinline__ void ClosedSourceCpmlAxes(const Parameters&p,int axes){
 switch(axes){
  case 0:ClosedSourceCpml<Depth,Uniform,0>(p);return;
  case 2:ClosedSourceCpml<Depth,Uniform,1>(p);return;
  case 4:ClosedSourceCpml<Depth,Uniform,2>(p);return;
  case 8:ClosedSourceCpml<Depth,Uniform,4>(p);return;
  default:ClosedSourceCpml<Depth,Uniform,-1>(p);return;
 }
}
template<int Depth,int Policy,int CubeN=0>
__device__ __forceinline__ void ClosedDispatch(const Parameters&p){
 const unsigned flags=p.flags[closed_pipeline::Tile()];const bool uniform=flags&64u;
 if constexpr(Policy==6){
  switch(beamz::cuda::moving::Policy(flags)){
   case 0:ClosedDispatch<Depth,0,CubeN>(p);return;
   case 1:ClosedDispatch<Depth,1,CubeN>(p);return;
   case 2:ClosedDispatch<Depth,2,CubeN>(p);return;
   case 3:ClosedDispatch<Depth,3,CubeN>(p);return;
   case 4:ClosedDispatch<Depth,4,CubeN>(p);return;
   case 5:ClosedDispatch<Depth,5,CubeN>(p);return;
  }
  if(!threadIdx.x)closed_pipeline::Abort(2048);
 }else if constexpr(Policy==0){
  if(uniform)UniformBody<Depth,1,true,true,0,true,-1,false,true,1024>(p,nullptr);
  else UniformBody<Depth,1,true,true,0,false,-1,false,true,1024>(p,nullptr);
 }else if constexpr(Policy==1){
  switch(flags&14u){case 2:ClosedNoIO<Depth,1,CubeN>(p);return;case 4:ClosedNoIO<Depth,2,CubeN>(p);return;case 8:ClosedNoIO<Depth,4,CubeN>(p);return;default:ClosedNoIO<Depth,-1,CubeN>(p);return;}
 }else if constexpr(Policy==2){
  if(uniform)UniformBody<Depth,1,false,true,2,true,-1,false,true,1024>(p,nullptr);
  else UniformBody<Depth,1,false,true,2,false,-1,false,true,1024>(p,nullptr);
 }else if constexpr(Policy==3){
  if(uniform)SourcePendingPairedBody<Depth,1,false,true,3,true,1024,20,true,-1,false,true>(p,nullptr);
  else SourcePendingPairedBody<Depth,1,false,true,3,false,1024,20,true,-1,false,true>(p,nullptr);
 }else if constexpr(Policy==4){
  SourcePendingPairedBody<Depth,1,false,true,4,false,1024,20,true,-1,false,true>(p,nullptr);
 }else if constexpr(Policy==5){
  ClosedSourceCpmlAxes<Depth,true>(p,flags&14u);
 }
}
__global__ void AsymmetricClosedReset(BeamzNativeClosedPipelineWorkspace q,const int32_t*bad){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<q.tile_count){q.next[i]=q.initial_next[i];q.pending[i]=q.initial_pending[i];q.claimed[i]=(q.initial_pending[i]==0);q.ready[i]=0;}
 if(i<q.cohort_count){q.heads[i]=q.initial_heads[i];q.enqueued[i]=q.root_counts[i];q.started[i]=0;q.completed[i]=0;q.exited[i]=0;}
 if(!i)*q.abort=(*bad!=0);
}
template<int Depth,int Policy,int CubeN=0>
__global__ __launch_bounds__(1024,1) void AsymmetricClosedStep(const __grid_constant__ Parameters p,
 const __grid_constant__ BeamzNativeClosedPipelineWorkspace q){
 using namespace closed_pipeline;
 if(!threadIdx.x){auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=p.bad;cold.tile=-1;cold.depth=Depth;cold.abort_local=0;cold.parent_ready=0;}
 __syncthreads();
 if(!q.cpml_uniform){if(!threadIdx.x)Abort(2048);return;}
 for(;;){
  if(!threadIdx.x)Pop();__syncthreads();if(C().tile<0)break;
  if(Policy!=6&&beamz::cuda::moving::Policy(p.flags[Tile()])!=Policy){if(!threadIdx.x)Abort(2048);}
  else ClosedDispatch<Depth,Policy,CubeN>(p);
  __syncthreads();
  if(!threadIdx.x&&!Aborted())Complete();
  __syncthreads();
 }
 __syncthreads();
 if(!threadIdx.x){
  const int previous=AddAcqRel(q.exited+q.cohort,1);
  if(previous==q.worker_count-1&&!Acquire(q.abort)){
   if(Acquire(q.completed+q.cohort)!=q.cohort_tasks||Acquire(q.started+q.cohort)!=q.cohort_tasks||Acquire(q.enqueued+q.cohort)!=q.cohort_tasks)Abort(512);
  }
 }
}
template<int Depth,int Policy,int CubeN=0>
inline cudaError_t LaunchClosedPolicy(cudaStream_t stream,const Parameters&p,const BeamzNativeClosedPipelineWorkspace&q){
 static_assert(closed_pipeline::ColdOffset>=ResearchSourceSharedBytes<5>,"Queue cold state aliases native profiles");
 auto status=cudaFuncSetAttribute(AsymmetricClosedStep<Depth,Policy,CubeN>,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes);if(status!=cudaSuccess)return status;
 AsymmetricClosedStep<Depth,Policy,CubeN><<<q.worker_count,1024,closed_pipeline::SharedBytes,stream>>>(p,q);return cudaGetLastError();
}
template<int Depth>
inline cudaError_t LaunchClosedCohorts(cudaStream_t stream,const Parameters&p,BeamzNativeClosedPipelineWorkspace q){
 for(int c=0;c<q.cohort_count;++c){
  q.cohort=c;q.cohort_tasks=q.cohort_offsets_host[c+1]-q.cohort_offsets_host[c];q.worker_count=std::min(q.max_workers,q.cohort_tasks);
  cudaError_t status=cudaErrorInvalidValue;
  switch(q.cohort_policies_host[c]){
   case 0:status=LaunchClosedPolicy<Depth,0>(stream,p,q);break;
   case 1:
    if(FixedNoIOCubeEligible<512>(p))status=LaunchClosedPolicy<Depth,1,512>(stream,p,q);
    else if(FixedNoIOCubeEligible<564>(p))status=LaunchClosedPolicy<Depth,1,564>(stream,p,q);
    else status=LaunchClosedPolicy<Depth,1,0>(stream,p,q);
    break;
   case 2:status=LaunchClosedPolicy<Depth,2>(stream,p,q);break;
   case 3:status=LaunchClosedPolicy<Depth,3>(stream,p,q);break;
   case 4:status=LaunchClosedPolicy<Depth,4>(stream,p,q);break;
   case 5:status=LaunchClosedPolicy<Depth,5>(stream,p,q);break;
  }
  if(status!=cudaSuccess)return status;
 }
 return cudaSuccess;
}
