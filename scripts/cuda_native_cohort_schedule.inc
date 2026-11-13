// Exact host-only dependency lowering; included after existing Launch helpers.
// Per-cohort events are unique within a macro, reused only after final join.
template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchCohortRanks(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){
 if(!w.cohort_offsets_host||!w.cohort_last_policy_host||!w.cohort_events_host||
    (w.cohort_dependency_count&&!w.cohort_dependencies_host))return cudaErrorInvalidValue;
 auto status=cudaEventRecord(static_cast<cudaEvent_t>(w.policy_fork_event),stream);if(status!=cudaSuccess)return status;
 for(int policy=0;policy<6;++policy)if(w.cohort_last_policy_host[policy]>=0){
  status=cudaStreamWaitEvent(static_cast<cudaStream_t>(w.policy_streams[policy]),static_cast<cudaEvent_t>(w.policy_fork_event),0);if(status!=cudaSuccess)return status;
 }
 for(int rank=0;rank<w.rank_count;++rank){const auto*offsets=w.policy_offsets_host+rank*7;
  for(int policy=0;policy<6;++policy){const int start=offsets[policy],count=offsets[policy+1]-start;if(!count)continue;
   const int cohort=rank*6+policy;const auto branch=static_cast<cudaStream_t>(w.policy_streams[policy]);
   for(int i=w.cohort_offsets_host[cohort];i<w.cohort_offsets_host[cohort+1];++i){
    const int producer=w.cohort_dependencies_host[i];
    status=cudaStreamWaitEvent(branch,static_cast<cudaEvent_t>(w.cohort_events_host[producer]),0);if(status!=cudaSuccess)return status;
   }
   switch(policy){
    case 0:status=Launch<Depth,Layout,true,IsotropicH,0>(branch,p,w.tile_ids_device+start,count);break;
    case 1:status=LaunchNoIO<Depth,Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_noio_counts_host[rank]);break;
    case 2:status=LaunchObservers<Depth,Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_observer_counts_host[rank]);break;
    case 3:status=Launch<Depth,Layout,false,IsotropicH,3>(branch,p,w.tile_ids_device+start,count);break;
    case 4:status=Launch<Depth,Layout,false,IsotropicH,4>(branch,p,w.tile_ids_device+start,count);break;
    case 5:status=Launch<Depth,Layout,false,IsotropicH,5>(branch,p,w.tile_ids_device+start,count);break;
   }
   if(status!=cudaSuccess)return status;
   status=cudaEventRecord(static_cast<cudaEvent_t>(w.cohort_events_host[cohort]),branch);if(status!=cudaSuccess)return status;
  }
 }
 // Every forked stream rejoins the origin before DFT, clock advance, another
 // macro, reset, readback, graph-capture end or any event-handle reuse.
 for(int policy=0;policy<6;++policy){const int last=w.cohort_last_policy_host[policy];if(last<0)continue;
  status=cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(w.cohort_events_host[last]),0);if(status!=cudaSuccess)return status;
 }
 return cudaSuccess;
}
