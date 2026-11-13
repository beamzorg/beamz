// Private research ABI for the started-parent, partial-step native pipeline.
#pragma once
#include "native_moving.h"
#include <cstdint>

struct BeamzNativePlanePipelineWorkspace {
 int32_t*queue=nullptr,*entry_ready=nullptr,*pending=nullptr,*claimed=nullptr,*ready=nullptr,*plane_done=nullptr;
 int32_t*head=nullptr,*tail=nullptr,*abort=nullptr,*completed=nullptr,*started=nullptr,*exited=nullptr;
 int tile_count=0,worker_count=0,nz=0,ny=0,nx=0;
 uint64_t timeout_ns=10000000000ULL;
};

int BeamzEnqueueNativePlanePipeline(void*stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&workspace,
 const BeamzNativePlanePipelineWorkspace&pipeline);
