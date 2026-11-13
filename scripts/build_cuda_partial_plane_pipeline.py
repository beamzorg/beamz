"""Bind partial-step resident workers to the original native scalar loops."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from scripts.cuda_trace_phase_cycle import block


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique partial-plane anchor required:'+old[:110])
    return text.replace(old,new,1)


ENQUEUE='''
int BeamzEnqueueNativePlanePipeline(void*opaque,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&w,
 const BeamzNativePlanePipelineWorkspace&q){
 using namespace beamz::cuda::moving_native;
 const auto validation=BeamzValidateNativeProgram(program);if(validation)return validation;
 if(!(w.data_flags&beamz::cuda::moving::kBytePackedE)||!(w.data_flags&beamz::cuda::moving::kLosslessIsotropicH)||
    !(w.data_flags&8u)||w.source_presence_mask!=20u||q.tile_count!=w.tile_count||q.worker_count<1||q.worker_count>q.tile_count||
    q.tile_count>INT32_MAX-q.worker_count||q.nz!=w.counts[0]||q.ny!=w.counts[1]||q.nx!=w.counts[2]||q.timeout_ns<1000000ULL||
    !q.queue||!q.entry_ready||!q.pending||!q.claimed||!q.ready||!q.plane_done||!q.head||!q.tail||!q.abort||!q.completed||!q.started||!q.exited)return cudaErrorInvalidValue;
 const size_t owners=3ULL*w.counts[1]*w.counts[2];
 if(w.trail_floats<owners*w.depth*6*SurfacePoints)return cudaErrorInvalidValue;
 Parameters p;int layout=0;if(!Prepare(program,final_fields,w,macro_step,&p,&layout)||layout!=1)return cudaErrorInvalidValue;
 const auto stream=static_cast<cudaStream_t>(opaque);
 BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
 AsymmetricPlaneReset<<<(q.tile_count+255)/256,256,0,stream>>>(q,w.bad);status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if(w.depth==4)return LaunchPlanePipeline<4>(stream,p,q);
 if(w.depth==16)return LaunchPlanePipeline<16>(stream,p,q);
 return cudaErrorInvalidValue;
}
'''


def stage(source,output,expected,*,outlined=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:raise ValueError('New disjoint pipeline source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=expected:raise ValueError('Explicit frozen native parent required')
    c=original
    c=once(c,'#include "native_moving.h"','#include "native_moving.h"\n#include "native_plane_pipeline.h"\n#include <cuda/atomic>')
    c=once(c,'__device__ __forceinline__ int WrapZ(int z)', '#include "native_plane_pipeline_control.cuh"\n\n__device__ __forceinline__ int WrapZ(int z)')
    c=once(c,'bool CachedProfiles=false> struct Environment {','bool CachedProfiles=false,bool Pipeline=false> struct Environment {')
    load='return p.phase[slot/6].psi[slot%6].input[PsiOffset(slot,z,y,x)];'
    if c.count(load)!=2:raise ValueError('Both generic and fixed mutable psi readers required')
    c=c.replace(load,'if constexpr(Pipeline)return __ldcg(p.phase[slot/6].psi[slot%6].input+PsiOffset(slot,z,y,x));else '+load)
    c=once(c,'template<int N,int ActiveAxes=-1>struct FixedCubeCachedEnvironment:',
           'template<int N,int ActiveAxes=-1,bool Pipeline=false>struct FixedCubeCachedEnvironment:')
    base='Environment<1,false,true,1,true,-1,false,ActiveAxes,true>'
    if c.count(base)!=2:raise ValueError('Both fixed environment bases required')
    c=c.replace(base,'Environment<1,false,true,1,true,-1,false,ActiveAxes,true,Pipeline>')
    signatures={
        'void UniformBody(':('template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>',
                             'template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false,bool Pipeline=false,int BlockT=0>'),
        'void FixedCubeCachedBody(':('template<int Depth,int N,int ActiveAxes=-1>', 'template<int Depth,int N,int ActiveAxes=-1,bool Pipeline=false>'),
        'void SourcePendingPairedBody(':('template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1,bool CachedProfiles=false>',
                                        'template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1,bool CachedProfiles=false,bool Pipeline=false>')}
    for old,new in signatures.values():c=once(c,old,new)
    edits=[]
    for signature in signatures:
        begin,end,body=block(c,signature);before=body
        if signature=='void UniformBody(':
            body=once(body,'constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,',
                      'constexpr int T=BlockT?BlockT:((Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512),')
        body=once(body,'const int tile=tiles[blockIdx.x],','const int tile=Pipeline?plane_pipeline::Tile():tiles[blockIdx.x],')
        body=body.replace('const int owner=(rank&3)*p.frontier+p.slots[tile];',
                          'const int owner=Pipeline?plane_pipeline::Owner(p,tile):(rank&3)*p.frontier+p.slots[tile];')
        body=body.replace('const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];',
                          'const int owner=Pipeline?plane_pipeline::Owner(p,producer):((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];')
        body=once(body,'if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))',
                  'if(bits&&(Pipeline?__ldcg(p.owner_tags+owner):p.owner_tags[owner])!=(epoch|uint32_t(producer)))')
        _,tag_end,_=block(body,'if(!threadIdx.x){const int owner=')
        position=body.index('__syncthreads();',tag_end)+len('__syncthreads();')
        body=body[:position]+'\n if constexpr(Pipeline){if(!threadIdx.x)plane_pipeline::Started();}'+body[position:]
        first='const int first=Fast?1:window[0],prev=first-1;' if 'Paired' in signature else 'const int first=window[0],prev=first-1;'
        body=once(body,first,first+'\n  if constexpr(Pipeline){plane_pipeline::Skipped(prev);plane_pipeline::WaitParents(prev);if(plane_pipeline::Aborted())return;}')
        current='  const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&31,low=-n-1;' if 'Paired' in signature else '  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;'
        body=once(body,current,'  if constexpr(Pipeline){plane_pipeline::WaitParents(n);if(plane_pipeline::Aborted())return;}\n'+current)
        body=body.replace('Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env',
                          'Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles,Pipeline>env')
        body=body.replace('Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>env',
                          'Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles,Pipeline>env')
        body=body.replace('FixedCubeCachedEnvironment<N,ActiveAxes>env','FixedCubeCachedEnvironment<N,ActiveAxes,Pipeline>env')
        a,b,loop=block(body,'for(int n=(Fast?1:window[0])')
        body=body[:a]+loop[:-1]+' if constexpr(Pipeline)plane_pipeline::Publish(n);\n }'+body[b:]
        edits.append(dict(signature=signature,before=before,after_sha256=hashlib.sha256(body.encode()).hexdigest()))
        c=c[:begin]+body+c[end:]
    marker='template<int Depth,int Layout>\ninline cudaError_t DispatchClass('
    outlined_bodies=''
    if outlined:
        for signature in signatures:
            begin,end,body=block(c,signature);start=c.rfind('template<',0,begin)
            clone=c[start:end].replace('__forceinline__','__noinline__',1)
            name=signature.split()[1].split('(')[0]
            clone=clone.replace('void '+name+'(','void PlaneOutlined'+name+'(',1)
            outlined_bodies+=clone+'\n\n'
    c=once(c,marker,outlined_bodies+'#include "native_plane_pipeline_kernel.cuh"\n\n'+marker)
    c+='\n'+ENQUEUE
    cmake=once((source/'CMakeLists.txt').read_text(),
               '  target_include_directories(beamz_cuda_moving PRIVATE src)',
               '''  target_include_directories(beamz_cuda_moving PRIVATE src)
  find_path(BEAMZ_CUDA_CCCL_INCLUDE cuda/atomic
    HINTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include"
          "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/../../usr/include/cccl"
    REQUIRED)
  target_include_directories(beamz_cuda_moving PRIVATE "${BEAMZ_CUDA_CCCL_INCLUDE}")''')
    e=(source/'src/native_extension.cc').read_text()
    e=once(e,'#include "native_moving.h"','#include "native_moving.h"\n#include "native_plane_pipeline.h"')
    e=once(e,'  BeamzNativeMovingPlan moving_plan_;','  BeamzNativePlanePipelineWorkspace moving_plane_pipeline_;bool moving_plane_pipeline_active_=false;\n  BeamzNativeMovingPlan moving_plan_;')
    static='  const auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));'
    e=once(e,static,'''  auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));
  const size_t old_owners=size_t(p.epochs)*p.max_frontier,owners=std::max(old_owners,size_t(3)*p.counts[1]*p.counts[2]);
  p.metadata_bytes+=(owners-old_owners)*sizeof(uint64_t);
  p.trail_floats=owners*depth*6*p.surface_points;
  const size_t pipeline_bytes=p.tile_ids.size()*5*sizeof(int32_t)+p.counts[0]*sizeof(int32_t)+6*sizeof(int32_t);''')
    e=once(e,'r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes;',
           'r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes+pipeline_bytes;r["plane_pipeline_bytes"]=pipeline_bytes;')
    metadata='    moving_observation_bytes_=observation.keys.size()*sizeof(int64_t)'
    e=once(e,metadata,'''    const size_t original_owners=size_t(moving_plan_.epochs)*moving_plan_.max_frontier;
    const size_t allocated_owners=std::max(original_owners,size_t(3)*moving_plan_.counts[1]*moving_plan_.counts[2]);
    moving_plan_.metadata_bytes+=(allocated_owners-original_owners)*sizeof(uint64_t);
    moving_plan_.trail_floats=allocated_owners*depth*6*moving_plan_.surface_points;
    const size_t pipeline_bytes=moving_plan_.tile_ids.size()*5*sizeof(int32_t)+moving_plan_.counts[0]*sizeof(int32_t)+6*sizeof(int32_t);
'''+metadata)
    e=once(e,'moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_;',
           'moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_+pipeline_bytes;')
    e=once(e,'size_t(moving_plan_.epochs)*moving_plan_.max_frontier*sizeof(uint64_t),true)', 'allocated_owners*sizeof(uint64_t),true)')
    allocation='    moving_workspace_.epoch=static_cast<uint32_t*>(upload(nullptr,4,true));moving_workspace_.bad=static_cast<int32_t*>(upload(nullptr,4,true));'
    e=once(e,allocation,allocation+'''
    auto&pipe=moving_plane_pipeline_;const size_t task_bytes=moving_plan_.tile_ids.size()*sizeof(int32_t);
    pipe.queue=static_cast<int32_t*>(upload(nullptr,task_bytes));pipe.entry_ready=static_cast<int32_t*>(upload(nullptr,task_bytes));
    pipe.pending=static_cast<int32_t*>(upload(nullptr,task_bytes));pipe.claimed=static_cast<int32_t*>(upload(nullptr,task_bytes));pipe.ready=static_cast<int32_t*>(upload(nullptr,task_bytes));
    pipe.plane_done=static_cast<int32_t*>(upload(nullptr,moving_plan_.counts[0]*sizeof(int32_t)));
    pipe.head=static_cast<int32_t*>(upload(nullptr,4));pipe.tail=static_cast<int32_t*>(upload(nullptr,4));pipe.abort=static_cast<int32_t*>(upload(nullptr,4));
    pipe.completed=static_cast<int32_t*>(upload(nullptr,4));pipe.started=static_cast<int32_t*>(upload(nullptr,4));pipe.exited=static_cast<int32_t*>(upload(nullptr,4));
    int pipeline_device=0,pipeline_sms=0;Check(cudaGetDevice(&pipeline_device),"pipeline current device");Check(cudaDeviceGetAttribute(&pipeline_sms,cudaDevAttrMultiProcessorCount,pipeline_device),"pipeline SM count");
    pipe.tile_count=moving_plan_.tile_ids.size();pipe.worker_count=std::min(pipeline_sms,pipe.tile_count);
    pipe.nz=moving_plan_.counts[0];pipe.ny=moving_plan_.counts[1];pipe.nx=moving_plan_.counts[2];
    moving_plane_pipeline_active_=(moving_workspace_.data_flags&beamz::cuda::moving::kBytePackedE)&&(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH)&&(moving_workspace_.data_flags&8u)&&moving_workspace_.source_presence_mask==20u;
''')
    key='    append_bytes(&moving_workspace_.parallel_policies,sizeof(bool));'
    e=once(e,key,key+'''
    const int pipeline_planes=3;append_bytes(&pipeline_planes,sizeof(int));append_bytes(&moving_plane_pipeline_active_,sizeof(bool));
    append_bytes(&pipe.worker_count,sizeof(int));append_bytes(&pipe.timeout_ns,sizeof(uint64_t));
    const void*pipeline_pointers[]={pipe.queue,pipe.entry_ready,pipe.pending,pipe.claimed,pipe.ready,pipe.plane_done,pipe.head,pipe.tail,pipe.abort,pipe.completed,pipe.started,pipe.exited};
    for(const void*pointer:pipeline_pointers)append_bytes(&pointer,sizeof(pointer));''')
    call='int error=BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);'
    e=once(e,call,'int error=moving_plane_pipeline_active_?BeamzEnqueueNativePlanePipeline(stream_.value,launch.program,other_.data(),offset,moving_workspace_,moving_plane_pipeline_):BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);')
    e=e.replace('"native-asymmetric-moving-v1"','"native-asymmetric-partial-plane-v1"')
    e=once(e,'    result["moving_observation_bytes"]=moving_observation_bytes_;',
           '    result["moving_plane_pipeline_active"]=moving_plane_pipeline_active_;result["moving_plane_pipeline_workers"]=moving_plane_pipeline_.worker_count;\n    result["moving_observation_bytes"]=moving_observation_bytes_;')
    e=once(e,'NB_MODULE(_cuda_native, module) {','NB_MODULE(_cuda_native, module) {\n module.attr("moving_plane_pipeline_compiled")=true;module.attr("moving_plane_pipeline_planes")=3;')
    shutil.copytree(source,output)
    (output/'CMakeLists.txt').write_text(cmake)
    for name,value in [('asymmetric_native.cuh',c),('native_extension.cc',e)]: (output/'src'/name).write_text(value)
    helpers={}
    for script,name in [('cuda_native_plane_pipeline.h','native_plane_pipeline.h'),('cuda_native_plane_pipeline_control.cuh','native_plane_pipeline_control.cuh'),('cuda_native_plane_pipeline_kernel.cuh','native_plane_pipeline_kernel.cuh')]:
        p=Path('scripts')/script;content=p.read_text()
        if outlined and script=='cuda_native_plane_pipeline_kernel.cuh':
            for signature in signatures:
                body_name=signature.split()[1].split('(')[0]
                content=content.replace(body_name+'<','PlaneOutlined'+body_name+'<')
        (output/'src'/name).write_text(content);helpers[name]=hashlib.sha256(content.encode()).hexdigest()
    proof=dict(schema='beamz.native-partial-plane-pipeline.v1',parent_consumer_sha256=expected,consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),
               extension_sha256=hashlib.sha256(e.encode()).hexdigest(),helpers=helpers,body_bindings=edits,expected_kernels=71,
               history_planes=3,outlined=outlined,initial_admission='losslessisotropicH bytepackedE and sourcePlanePair20fact; originalcompletefallback forothers',
               field_math='Original point loops and canonical scalar-cell helper retained; mutable psi loads coherentcg only in pipeline instantiations',
               synchronization='Started afterTagCTAbarrier; lateprevwait; currentghostwait; perwriterfence+CTAjoin+leaderrelease perstep; callerjoin beforeplanecomplete',
               allocation='maxoldrank/threeplaneowners; queues+counterbytes included in CPUandactualfreeguards',GPU_executed=False)
    (output.parent/'partial-plane-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--outlined',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,outlined=a.outlined)))
