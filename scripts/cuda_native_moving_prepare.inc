// Included in NativeSession's private section by the private candidate patcher.
#if BEAMZ_CUDA_NATIVE_MOVING
  void EnsureMovingResources(int depth){
    if(moving_ready_){if(depth!=moving_workspace_.depth)throw std::invalid_argument("moving depth is fixed per session");return;}
    // A failed upload must not leave a half-built plan on a retry.
    if(!moving_allocations_.empty()){
      Check(cudaStreamSynchronize(stream_.value),"retire incomplete moving upload");
      moving_allocations_.clear();
    }
    moving_workspace_={};moving_fact_key_.clear();
    if(depth!=4&&depth!=16)throw std::invalid_argument("compiled native moving depth must be4 or16");
    if(!mstep_||workspace_count_!=12||!attributes_.cpml_enabled||attributes_.metric_kind!=0||attributes_.boundary_code!=(12<<8))throw std::invalid_argument("moving requires Cartesian FP32 CPML12 Mstep without PEC");
    int minimum[3]{INT32_MAX,INT32_MAX,INT32_MAX};
    constexpr int dz[6]={0,0,1,1,1,0},dy[6]={0,1,0,1,0,1},dx[6]={1,0,0,0,1,1};
    for(int c=0;c<6;++c)for(int a=0;a<3;++a)minimum[a]=std::min(minimum[a],int(base_inputs_[c].dims[a]));
    for(int c=0;c<6;++c){const auto&field=base_inputs_[c];const auto&final=other_[c];
      if(field.rank!=3||field.element_type!=kBeamzF32||final.rank!=3||final.element_type!=kBeamzF32||field.data==final.data)throw std::invalid_argument("moving requires separate FP32 initial/final field banks");
      const int add[3]={dz[c],dy[c],dx[c]};
      for(int a=0;a<3;++a)if(field.dims[a]!=minimum[a]+add[a]||final.dims[a]!=field.dims[a])throw std::invalid_argument("moving requires canonical staggered supports");
      for(const auto*f:{&field,&final})if(f->row_stride!=f->dims[2]||f->plane_stride!=f->dims[1]*f->dims[2]||f->compact_slots)throw std::invalid_argument("moving requires dense canonical field pitches");
    }
    bool lossless=true;uint32_t source_bits[3]{};
    for(int c=0;c<3;++c){
      for(int index:{6+c,9+c})if(base_inputs_[index].rank!=0||base_inputs_[index].element_type!=kBeamzF32)throw std::invalid_argument("moving requires scalar FP32 H coefficients");
      std::memcpy(&moving_workspace_.h_decay[c],host_[6+c].data,4);std::memcpy(&moving_workspace_.h_source[c],host_[9+c].data,4);
      std::memcpy(&source_bits[c],host_[9+c].data,4);
      lossless&=moving_workspace_.h_decay[c]==1.f;
      if(!std::isfinite(moving_workspace_.h_decay[c])||!std::isfinite(moving_workspace_.h_source[c]))throw std::invalid_argument("moving requires finite H coefficients");
    }
    if(lossless&&source_bits[0]==source_bits[1]&&source_bits[1]==source_bits[2])moving_workspace_.data_flags|=beamz::cuda::moving::kLosslessIsotropicH;
    const int layout=base_inputs_[40].rank;
    for(int c=0;c<3;++c){const auto&table=base_inputs_[37+c];const auto&codes=base_inputs_[40+c];
      if(table.rank!=1||table.element_type!=kBeamzF32||table.dims[0]<1||table.dims[0]>INT32_MAX||codes.element_type!=kBeamzS32||codes.rank!=layout||(layout!=1&&layout!=2))throw std::invalid_argument("moving requires homogeneous byte/block packed E coefficients");
    }
    moving_workspace_.data_flags|=layout==1?beamz::cuda::moving::kBytePackedE:beamz::cuda::moving::kBlockPackedE;
    for(int g=0;g<9;++g){const int index=source_offset_+3*g;const auto&coeff=base_inputs_[index];
      if(g<3&&coeff.dims[0])throw std::invalid_argument("moving pre-E source is unsupported");
      if(!coeff.dims[0])continue;
      if(coeff.dims[0]<=1||!((attributes_.coincident_source_group_mask>>g)&1))throw std::invalid_argument("moving initially admits only multi-source coincident arithmetic");
      const auto*starts=static_cast<const int32_t*>(host_[index+2].data);
      for(int s=1;s<coeff.dims[0];++s)for(int a=0;a<3;++a)if(starts[3*s+a]!=starts[a])throw std::invalid_argument("moving coincident source starts differ");
    }
    BindOutputs();auto a=attributes_;a.nsteps=std::min(depth,attributes_.nsteps);BeamzNativeLaunchStorage validate;
    BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&validate);
    Check(static_cast<cudaError_t>(BeamzValidateNativeProgram(validate.program)),"validate moving packet");
    constexpr int axes[6]={1,0,0,2,2,1};
    for(int family=0;family<2;++family){const auto&p=family?validate.program.e_ab:validate.program.h_ab;
      if(p.uniform_cpml_thickness!=12)throw std::invalid_argument("moving requires uniform CPML12");
      for(int term=0;term<6;++term){const int axis=axes[term];const auto&target=p.outputs[term/2];const auto&psi=p.inputs[31+term];
        if(psi.rank!=3||psi.element_type!=kBeamzF32||p.outputs[3+term].data!=psi.data||psi.compact_slots)throw std::invalid_argument("moving requires in-place FP32 packed psi");
        for(int d=0;d<3;++d)if(psi.dims[d]!=(d==axis?24:target.dims[d]))throw std::invalid_argument("moving packed psi shape differs");
        if(psi.row_stride!=psi.dims[2]||psi.plane_stride!=psi.dims[1]*psi.dims[2])throw std::invalid_argument("moving psi must be canonical packed slabs");
        for(int k=0;k<3;++k){const auto&profile=p.inputs[13+3*term+k];
          if(profile.rank!=3||profile.element_type!=kBeamzF32)throw std::invalid_argument("moving profile type/rank differs");
          for(int d=0;d<3;++d)if(profile.dims[d]!=(d==axis?24:1))throw std::invalid_argument("moving profile dimensions differ");
        }
      }
    }
    BeamzNativeDeepPlan observation;
    if(attributes_.monitor_count)observation=BeamzBuildNativeDeepPlan(base_inputs_.data(),base_inputs_[monitor_offset_],static_cast<const int32_t*>(host_[monitor_offset_].data));
    if(attributes_.monitor_count&&observation.keys.empty())throw std::invalid_argument("moving monitor has no valid raw samples");
    std::vector<std::array<int,3>>points;points.reserve(observation.keys.size());
    int component=0;
    for(int64_t key:observation.keys){while(component<5&&key>=observation.field_base[component+1])++component;const int64_t local=key-observation.field_base[component];const auto&f=base_inputs_[component];
      points.push_back({int(local/(f.dims[1]*f.dims[2])),int((local/f.dims[2])%f.dims[1]),int(local%f.dims[2])});
    }
    const int32_t*starts[9]{};for(int g=0;g<9;++g)starts[g]=static_cast<const int32_t*>(host_[source_offset_+3*g+2].data);
    moving_plan_=BeamzBuildNativeMovingPlan(base_inputs_.data(),depth,points,validate.groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));
    BeamzNativeMovingSparseObservation sparse;
    moving_lookup_bytes_=0;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
    sparse=BeamzBuildNativeMovingSparseObservation(base_inputs_.data(),observation.keys);
    moving_lookup_bytes_=(sparse.coarse.size()+sparse.leaf.size())*sizeof(int32_t);
#endif
    moving_observation_bytes_=observation.keys.size()*sizeof(int64_t)+observation.publication_words.size()*sizeof(uint32_t)+observation.remapped_indices.size()*sizeof(int32_t)+size_t(depth)*observation.keys.size()*sizeof(float)+moving_lookup_bytes_;
    moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
    if(!BEAMZ_CUDA_NATIVE_MOVING_POLICIES)throw std::invalid_argument("parallel moving requires policy partition");
    while(moving_policy_streams_.size()<6)moving_policy_streams_.push_back(std::make_unique<Stream>());
    while(moving_policy_done_.size()<6)moving_policy_done_.push_back(std::make_unique<MovingDependencyEvent>());
    if(!moving_policy_fork_)moving_policy_fork_=std::make_unique<MovingDependencyEvent>();
    for(int policy=0;policy<6;++policy){moving_workspace_.policy_streams[policy]=moving_policy_streams_[policy]->value;moving_workspace_.policy_done_events[policy]=moving_policy_done_[policy]->value;}
    moving_workspace_.policy_fork_event=moving_policy_fork_->value;moving_workspace_.parallel_policies=true;
#endif
    size_t free,total;Check(cudaMemGetInfo(&free,&total),"query moving admission");
    if(moving_extra_bytes_+(256ULL<<20)>free)throw std::invalid_argument("moving workspace plus256MiB reserve exceeds actual free GPU memory");
    auto upload=[&](const void*data,size_t bytes,bool zero=false){auto memory=std::make_unique<DeviceMemory>(bytes);
      if(bytes){if(zero)Check(cudaMemsetAsync(memory->data,0,bytes,stream_.value),"initialize moving metadata");else if(data)Check(cudaMemcpyAsync(memory->data,data,bytes,cudaMemcpyHostToDevice,stream_.value),"upload moving metadata");}
      void*pointer=memory->data;moving_allocations_.push_back(std::move(memory));return pointer;
    };
    moving_workspace_.tile_ids_device=static_cast<int32_t*>(upload(moving_plan_.tile_ids.data(),moving_plan_.tile_ids.size()*4));
    moving_workspace_.slots_device=static_cast<int32_t*>(upload(moving_plan_.slots.data(),moving_plan_.slots.size()*4));
    moving_workspace_.tile_flags_device=static_cast<uint8_t*>(upload(moving_plan_.tile_flags.data(),moving_plan_.tile_flags.size()));
    moving_workspace_.trails=static_cast<float*>(upload(nullptr,moving_plan_.trail_floats*sizeof(float)));
    moving_workspace_.owner_tags=static_cast<uint64_t*>(upload(nullptr,size_t(moving_plan_.epochs)*moving_plan_.max_frontier*sizeof(uint64_t),true));
    moving_workspace_.epoch=static_cast<uint32_t*>(upload(nullptr,4,true));moving_workspace_.bad=static_cast<int32_t*>(upload(nullptr,4,true));
    moving_workspace_.policy_offsets_host=moving_plan_.policy_offsets.empty()?nullptr:moving_plan_.policy_offsets.data();
    moving_workspace_.rank_offsets_host=moving_plan_.rank_offsets.data();moving_workspace_.bulk_counts_host=moving_plan_.bulk_counts.data();
    std::copy(moving_plan_.counts.begin(),moving_plan_.counts.end(),moving_workspace_.counts);std::copy(moving_plan_.owned.begin(),moving_plan_.owned.end(),moving_workspace_.owned);
    moving_workspace_.max_frontier=moving_plan_.max_frontier;moving_workspace_.rank_count=moving_plan_.bulk_counts.size();moving_workspace_.tile_count=moving_plan_.tile_ids.size();moving_workspace_.depth=depth;moving_workspace_.epochs=moving_plan_.epochs;moving_workspace_.trail_floats=moving_plan_.trail_floats;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
    moving_workspace_.observation_tiles=static_cast<const int32_t*>(upload(sparse.coarse.data(),sparse.coarse.size()*4));
    moving_workspace_.observation_leaf=static_cast<const int32_t*>(upload(sparse.leaf.data(),sparse.leaf.size()*4));
    std::copy(sparse.grid.begin(),sparse.grid.end(),moving_workspace_.observation_grid);
#endif
    auto&obs=moving_workspace_.observation;
    obs.keys=static_cast<int64_t*>(upload(observation.keys.data(),observation.keys.size()*8));
    obs.publication_words=static_cast<uint32_t*>(upload(observation.publication_words.data(),observation.publication_words.size()*4));
    moving_remapped_=base_inputs_[monitor_offset_];moving_remapped_.data=upload(observation.remapped_indices.data(),observation.remapped_indices.size()*4);
    obs.values=static_cast<float*>(upload(nullptr,size_t(depth)*observation.keys.size()*4));obs.key_count=observation.keys.size();
    std::copy(observation.field_base.begin(),observation.field_base.end(),obs.field_base);obs.mask_nz=observation.mask_shape[0];obs.mask_ny=observation.mask_shape[1];obs.mask_nx=observation.mask_shape[2];
    // Serialize every host-side launch decision rather than relying on device
    // pointer identity or a probabilistic plan hash across fresh sessions.
    moving_fact_key_.clear();
    const auto append_bytes=[&](const void*p,size_t n){if(n)moving_fact_key_.append(static_cast<const char*>(p),n);};
    append_bytes(moving_plan_.policy_offsets.data(),moving_plan_.policy_offsets.size()*4);append_bytes(moving_plan_.tile_ids.data(),moving_plan_.tile_ids.size()*4);append_bytes(moving_plan_.rank_offsets.data(),moving_plan_.rank_offsets.size()*4);append_bytes(moving_plan_.slots.data(),moving_plan_.slots.size()*4);append_bytes(moving_plan_.bulk_counts.data(),moving_plan_.bulk_counts.size()*4);append_bytes(moving_plan_.tile_flags.data(),moving_plan_.tile_flags.size());
    append_bytes(moving_workspace_.counts,sizeof(moving_workspace_.counts));append_bytes(moving_workspace_.owned,sizeof(moving_workspace_.owned));append_bytes(&moving_workspace_.depth,sizeof(int));append_bytes(&moving_workspace_.max_frontier,sizeof(int));append_bytes(&moving_workspace_.epochs,sizeof(int));
    append_bytes(moving_workspace_.observation_grid,sizeof(moving_workspace_.observation_grid));
    append_bytes(&moving_workspace_.parallel_policies,sizeof(bool));
    append_bytes(moving_workspace_.h_decay,sizeof(moving_workspace_.h_decay));append_bytes(moving_workspace_.h_source,sizeof(moving_workspace_.h_source));append_bytes(&moving_workspace_.data_flags,sizeof(uint32_t));
    append_bytes(&obs.key_count,sizeof(int));append_bytes(obs.field_base,sizeof(obs.field_base));append_bytes(&obs.mask_nz,sizeof(int));append_bytes(&obs.mask_ny,sizeof(int));append_bytes(&obs.mask_nx,sizeof(int));
    Check(cudaStreamSynchronize(stream_.value),"finish moving workspace preparation");
    resident_bytes_+=moving_extra_bytes_;moving_ready_=true;
  }
  int EnqueueMovingDft(const BeamzProgramLaunch&program,int offset){
    const auto&obs=moving_workspace_.observation;
    if(!program.monitors||!program.monitors->monitor_count)return cudaSuccess;
    auto h=program.h_ab,e=program.e_ab;auto monitors=*program.monitors;monitors.indices=moving_remapped_;
    BeamzBuffer field{};field.rank=3;field.element_type=kBeamzF32;field.dims[0]=field.dims[1]=1;field.dims[2]=obs.key_count;field.row_stride=field.plane_stride=obs.key_count;
    for(int version=0;version<moving_workspace_.depth;++version){field.data=obs.values+size_t(version)*obs.key_count;
      for(int c=0;c<3;++c)h.outputs[c]=e.outputs[c]=field;
      const auto error=BeamzEnqueueDftGroups(stream_.value,h,e,monitors,offset+version);if(error!=cudaSuccess)return error;
    }
    return cudaSuccess;
  }
#endif
