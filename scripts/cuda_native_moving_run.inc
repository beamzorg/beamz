// Included in NativeSession's public section by the private candidate patcher.
  nb::dict RunMoving(const std::vector<int>&chunks,int depth){
#if !BEAMZ_CUDA_NATIVE_MOVING
    (void)chunks;(void)depth;throw std::invalid_argument("native moving support not compiled");
#else
    auto guard=Guard();RequireOpen();
    if(chunks.empty())throw std::invalid_argument("moving schedule cannot be empty");
    int64_t total=0;
    for(int count:chunks){if(count<1)throw std::invalid_argument("moving chunks must be positive");total+=count;if(total>INT32_MAX)throw std::invalid_argument("moving total exceeds int32");}
    const int first_step=preserve_?initial_step_:step_,first_elapsed=preserve_?initial_elapsed_:elapsed_;
    if(int64_t(first_step)+total>INT32_MAX||int64_t(first_elapsed)+total>INT32_MAX)throw std::invalid_argument("moving continuation exceeds int32 clocks");
    for(int g=0;g<9;++g)if(base_inputs_[source_offset_+3*g].dims[0]&&int64_t(first_step)+total>base_inputs_[source_offset_+3*g+1].dims[1])throw std::invalid_argument("moving exceeds prepared source waveforms");
    EnsureMovingResources(depth);
    const auto started=std::chrono::steady_clock::now();
    float cuda_ms=0,physics_ms=0,prepare_ms=0;
    {
      nb::gil_scoped_release release;
      while(chunk_starts_.size()<chunks.size()){chunk_starts_.push_back(std::make_unique<Event>());chunk_ready_.push_back(std::make_unique<Event>());chunk_finished_.push_back(std::make_unique<Event>());}
      Check(cudaEventRecord(start_.value,stream_.value),"record moving start");
      if(preserve_){
        BindInitial();
        for(const auto&snapshot:snapshots_)if(host_[snapshot.first].bytes)Check(cudaMemcpyAsync(base_inputs_[snapshot.first].data,snapshot.second->data,host_[snapshot.first].bytes,cudaMemcpyDeviceToDevice,stream_.value),"restore moving initial state");
        step_=initial_step_;elapsed_=initial_elapsed_;
      }
      // Operational epoch/tags are deliberately NOT restored with physical
      // state, so replay/reset cannot accept a stale previous-macro producer.
      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");
      for(size_t chunk=0;chunk<chunks.size();++chunk){
        if(chunk)Check(cudaEventRecord(chunk_starts_[chunk]->value,stream_.value),"record moving chunk preparation");
        Check(cudaEventRecord(chunk_ready_[chunk]->value,stream_.value),"record moving physics start");
        const int count=chunks[chunk];const auto initial_fields=current_,initial_other=other_;
        BindOutputs();auto a=attributes_;a.nsteps=count;
        BeamzNativeLaunchStorage identity;
        BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&identity);
        std::string key=BeamzGraphKey("native-asymmetric-moving-v1",stream_.value,identity.program);
        for(const auto&field:other_)BeamzAppendBufferKey(&key,field);
        key.append(moving_fact_key_);
        BeamzAppendBufferKey(&key,moving_remapped_);
        const auto append_pointer=[&](const void*pointer){key.append(reinterpret_cast<const char*>(&pointer),sizeof(pointer));};
        append_pointer(moving_workspace_.tile_ids_device);append_pointer(moving_workspace_.slots_device);append_pointer(moving_workspace_.tile_flags_device);
        append_pointer(moving_workspace_.trails);append_pointer(moving_workspace_.owner_tags);append_pointer(moving_workspace_.epoch);append_pointer(moving_workspace_.bad);
        append_pointer(moving_workspace_.observation.keys);append_pointer(moving_workspace_.observation.publication_words);append_pointer(moving_workspace_.observation.values);
        append_pointer(moving_workspace_.observation_tiles);append_pointer(moving_workspace_.observation_leaf);
        for(int policy=0;policy<6;++policy){append_pointer(moving_workspace_.policy_streams[policy]);append_pointer(moving_workspace_.policy_done_events[policy]);}
        append_pointer(moving_workspace_.policy_fork_event);
        append_pointer(clock_time_->data);
        const auto enqueue=[&]()->cudaError_t{
          for(int offset=0;offset+depth<=count;offset+=depth){
            BindOutputs();BeamzNativeLaunchStorage launch;
            BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&launch);
            int error=BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);
            if(error)return static_cast<cudaError_t>(error);
            error=EnqueueMovingDft(launch.program,offset);
            if(error)return static_cast<cudaError_t>(error);
            std::swap(current_,other_);
          }
          BindOutputs();BeamzNativeLaunchStorage tail;
          BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&tail);
          for(int offset=(count/depth)*depth;offset<count;++offset){const int error=BeamzEnqueueProgramStepAt(stream_.value,tail.program,offset);if(error)return static_cast<cudaError_t>(error);}
          return BeamzNativeAdvanceClocks(stream_.value,static_cast<int*>(base_inputs_[monitor_offset_+14].data),static_cast<int*>(base_inputs_[monitor_offset_+15].data),static_cast<const float*>(base_inputs_[monitor_offset_+13].data),static_cast<float*>(clock_time_->data),attributes_.dt,count);
        };
        Check(BeamzLaunchGraph(stream_.value,key,(a.cuda_flags&kBeamzGraphCache)!=0,a.graph_cache_capacity,enqueue),"execute native moving graph");
        // Capture closures are skipped on cache hits. Commit field parity
        // independently of capture/replay; packed psi always remains in place.
        current_=initial_fields;other_=initial_other;
        if((count/depth)&1)std::swap(current_,other_);
        step_+=count;elapsed_+=count;
        Check(cudaEventRecord(chunk_finished_[chunk]->value,stream_.value),"record moving chunk end");
      }
      BindOutputs();last_bank_=0;last_nsteps_=static_cast<int>(total);last_chunk_count_=chunks.size();++runs_;
      Check(cudaEventRecord(end_.value,stream_.value),"record moving end");Check(cudaEventSynchronize(end_.value),"finish moving solve");
      Check(cudaEventElapsedTime(&cuda_ms,start_.value,end_.value),"measure moving solve");
      for(size_t chunk=0;chunk<chunks.size();++chunk){float physics=0,prepare=0;
        Check(cudaEventElapsedTime(&physics,chunk_ready_[chunk]->value,chunk_finished_[chunk]->value),"measure moving chunk physics");
        Check(cudaEventElapsedTime(&prepare,chunk?chunk_starts_[chunk]->value:start_.value,chunk_ready_[chunk]->value),"measure moving preparation");physics_ms+=physics;prepare_ms+=prepare;
      }
      int32_t bad=0;Check(cudaMemcpy(&bad,moving_workspace_.bad,sizeof(bad),cudaMemcpyDeviceToHost),"read moving ownership diagnostic");
      if(bad)throw std::runtime_error("native moving producer/staging diagnostic failed: "+std::to_string(bad));
    }
    nb::dict result;result["cuda_ms"]=cuda_ms;result["physics_cuda_ms"]=physics_ms;result["prepare_cuda_ms"]=prepare_ms;
    result["host_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    result["nsteps"]=static_cast<int>(total);result["chunks"]=nb::cast(chunks);result["output_bank"]=0;
    result["physical_field_bank"]=current_[0].data==base_inputs_[0].data?0:1;
    result["preserve_inputs"]=preserve_;result["initialize_scratch"]=false;result["scratch_zero_bytes"]=0;
    result["preserving_copy_bytes"]=preserve_?preserve_bytes_:0;result["algorithm"]="asymmetric_moving";result["moving_depth"]=depth;
    int macros=0,tails=0;for(int count:chunks){macros+=count/depth;tails+=count%depth;}
    result["moving_macro_count"]=macros;result["ordinary_tail_steps"]=tails;
    result["physics_timing_scope"]="moving_rank_kernels_sources_raw_observations_DFT_tails_and_clock_advance";
    result["diagnostic_read_after_cuda_timer"]=true;return result;
#endif
  }
