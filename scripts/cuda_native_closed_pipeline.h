// Private research ABI for schedule-first physics closures and unique-node LIFO.
#pragma once
#include "native_moving.h"
#include <cstdint>

struct BeamzNativeClosedPipelineWorkspace {
 int32_t*next=nullptr,*pending=nullptr,*claimed=nullptr,*ready=nullptr;
 const int32_t*initial_next=nullptr,*initial_pending=nullptr,*initial_heads=nullptr,*root_counts=nullptr;
 const int32_t*tile_cohort=nullptr,*owner_slots=nullptr,*finish_offsets=nullptr,*finish_children=nullptr;
 int32_t*heads=nullptr,*enqueued=nullptr,*started=nullptr,*completed=nullptr,*exited=nullptr,*abort=nullptr;
 const int32_t*cohort_offsets_host=nullptr,*cohort_policies_host=nullptr;
 int tile_count=0,cohort_count=0,max_workers=0,finish_count=0,cpml_uniform=0;
 int cohort=0,cohort_tasks=0,worker_count=0;
 uint64_t timeout_ns=10000000000ULL;
};

int BeamzEnqueueNativeClosedPipeline(void*stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&workspace,
 const BeamzNativeClosedPipelineWorkspace&pipeline);
