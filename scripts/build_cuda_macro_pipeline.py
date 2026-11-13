"""Lower proved cross-macro ownership into encoded CUDA tasks and raw slabs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from scripts.cuda_trace_phase_cycle import block


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique macro pipeline anchor required:'+old[:120])
    return text.replace(old,new,1)


def stage(source,output,expected,macros=2,native_banked=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents or macros not in (2,4,8):raise ValueError('New disjoint2/4/8macro research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=expected or (not native_banked and 'PhysicalFrontierIndex' not in original):raise ValueError('Explicit immutable frontier parent required')
    c=original
    helpers='''__device__ __forceinline__ int PhysicalTileID(const Parameters&p,int encoded){return encoded%(p.nz*p.ny*p.nx);}
template<int Depth>__device__ __forceinline__ int MacroStepOffset(const Parameters&p,int encoded){return (encoded/(p.nz*p.ny*p.nx))*Depth;}
'''
    c=once(c,'__device__ __forceinline__ int WrapZ(int z)',helpers+'\n__device__ __forceinline__ int WrapZ(int z)')
    c=once(c,'const Parameters&p;uint8_t flags;const float*profiles=nullptr;',
           'const Parameters&p;uint8_t flags;const float*profiles=nullptr;int step_offset=0;')
    if c.count('*s.current_step+p.macro_step+version-1')!=2:raise ValueError('Both exact source time bindings required')
    c=c.replace('*s.current_step+p.macro_step+version-1','*s.current_step+p.macro_step+step_offset+version-1')
    c=once(c,'Observe(p,field,version,z,y,x,value);','Observe(p,field,version+step_offset,z,y,x,value);')
    if native_banked:
        c=once(c,'int nz,ny,nx,frontier,macro_step;','int nz,ny,nx,frontier,macro_step,macro_level_stride;')
        c=once(c,'out->macro_step=macro;','out->macro_step=macro;out->macro_level_stride=w.slab_macro_stride;')
        c=once(c,'float Initial(const Parameters&p,int field,int z,int y,int x){',
               'float Initial(const Parameters&p,int field,int z,int y,int x,int bank=0){')
        c=once(c,'const auto&f=InitialField(p,field);return Contains(f,z,y,x)?f.data[z*f.plane+y*f.row+x]:0.f;',
               'const auto&f=(bank&1)?FinalField(p,field):InitialField(p,field);return Contains(f,z,y,x)?f.data[z*f.plane+y*f.row+x]:0.f;')
        c=once(c,'__device__ __forceinline__ float Initial(','''__device__ __forceinline__ const MstepCompactField& MacroFinalField(const Parameters&p,int field,int bank){return (bank&1)?InitialField(p,field):FinalField(p,field);}
__device__ __forceinline__ float Initial(''')
        terminal=c.index(' if constexpr(ReuseTerminal){',c.index('FieldPacket ImportPacket('))
        end_terminal=c.index(' const float*point=p.trails',terminal)
        c=c[:terminal]+' if constexpr(ReuseTerminal){if(version==Depth)mask&=7u;}\n'+c[end_terminal:]
    edits=[]
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
        begin,end,body=block(c,signature);before=body
        if not native_banked:
            for anchor in ('if(!threadIdx.x){const int owner=','if(threadIdx.x<8){const int bits='):
                a,b,_=block(body,anchor);body=body[:a]+body[b:]
            body=once(body,' const uint64_t epoch=uint64_t(*p.epoch)<<32;','')
            body=once(body,'const int tile=tiles[blockIdx.x],','const int tile=PhysicalTileID(p,tiles[blockIdx.x]),')
        else:
            body=once(body,'const int tile=tiles[blockIdx.x],','const int encoded=tiles[blockIdx.x],macro_index=MacroStepOffset<1>(p,encoded),tile=PhysicalTileID(p,encoded),')
            body=once(body,'rank=bz+by+bx;','rank=bz+by+bx+macro_index*p.macro_level_stride;')
            body=body.replace('p.slots[tile]','p.slots[encoded]').replace('uint32_t(tile)','uint32_t(encoded)')
            body=body.replace('p.slots[producer]','p.slots[producer+macro_index*p.nz*p.ny*p.nx]').replace('uint32_t(producer)','uint32_t(producer+macro_index*p.nz*p.ny*p.nx)')
            body=re.sub(r'Initial\(p,([^,]+),z,y,x\)',r'Initial(p,\1,z,y,x,macro_index)',body)
            body=body.replace('FinalField(p,c)','MacroFinalField(p,c,macro_index)').replace('FinalField(p,c+3)','MacroFinalField(p,c+3,macro_index)')
            count=body.count('if(surface>=0&&(Layout!=1||n<Depth))')
            if count!=(2 if 'Paired' in signature else 1):raise ValueError('Native macro publication sites changed')
            body=body.replace('if(surface>=0&&(Layout!=1||n<Depth))','if(surface>=0)')
            body=body.replace('p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=','if(Layout!=1||n<Depth)p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=')
        body,count=re.subn(r'(env\{p,flags,profiles\};)',r'\1env.step_offset=MacroStepOffset<Depth>(p,tiles[blockIdx.x]);',body)
        if count!=1 or (not native_banked and ('owner_tags' in body or 'bases[' in body)):raise ValueError('Complete per-task macro binding required')
        edits.append(dict(signature=signature,before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(body.encode()).hexdigest()))
        c=c[:begin]+body+c[end:]
    for array in ('tiles','ids'):
        c=c.replace(f'p.flags[{array}[blockIdx.x]]',f'p.flags[PhysicalTileID(p,{array}[blockIdx.x])]')
    # Prepare uses the original physical plan. Only the actual host launch view
    # switches to merged affine levels and encoded task IDs after admission.
    entry=block(c,'int BeamzEnqueueNativeAsymmetric(')[2]
    entry=once(entry,'int BeamzEnqueueNativeAsymmetric(','int BeamzEnqueueNativeMacroSlab(')
    entry=once(entry,'const BeamzNativeMovingWorkspace&w){','const BeamzNativeMovingWorkspace&base){')
    entry=once(entry,'if(!Prepare(program,final_fields,w,macro_step,&p,&layout))',
               'if(!Prepare(program,final_fields,base,macro_step,&p,&layout))')
    anchor='#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES\n if(!w.uniform_noio_counts_host'
    setup=''' if(!base.slab_tile_ids_device||!base.slab_policy_offsets_host||!base.slab_noio_counts_host||!base.slab_observer_counts_host||base.slab_level_count<1)return cudaErrorInvalidValue;
 BeamzNativeMovingWorkspace slab=base;
 slab.tile_ids_device=base.slab_tile_ids_device;slab.policy_offsets_host=base.slab_policy_offsets_host;
 slab.uniform_noio_counts_host=base.slab_noio_counts_host;slab.uniform_observer_counts_host=base.slab_observer_counts_host;
 slab.rank_count=base.slab_level_count;const auto&w=slab;
'''
    if native_banked:
        setup+=' if(!base.slab_slots_device||base.slab_max_frontier<1)return cudaErrorInvalidValue;\n p.slots=base.slab_slots_device;p.frontier=base.slab_max_frontier;\n'
    entry=once(entry,anchor,setup+anchor)
    c+='\n'+entry+'\n'
    h=(source/'src/native_moving.h').read_text()
    h=once(h,'  bool parallel_policies=false;','''  const int32_t*slab_tile_ids_device=nullptr;
  const int32_t*slab_policy_offsets_host=nullptr;
  const int32_t*slab_noio_counts_host=nullptr,*slab_observer_counts_host=nullptr;
  const int32_t*slab_slots_device=nullptr;
  int slab_level_count=0,slab_max_frontier=0,slab_macro_stride=0;
  bool parallel_policies=false;''')
    declaration='''int BeamzEnqueueNativeMacroSlab(void*stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&workspace);
'''
    h=once(h,'#endif',declaration+'\n#endif')
    e=(source/'src/native_extension.cc').read_text()
    e=once(e,'#include "native_moving.h"','#include "native_moving.h"\n#include "native_macro_slab.h"')
    e=once(e,'  BeamzNativeMovingPlan moving_plan_;','  BeamzNativeMovingMacroSlab moving_macro_slab_;\n  BeamzNativeMovingPlan moving_plan_;')
    # Public CPU planning includes the full raw slab and encoded GPU ID vector.
    e=once(e,'+size_t(depth)*obs.keys.size()*4+lookup_bytes;',f'+size_t(depth)*{macros}*obs.keys.size()*4+lookup_bytes;')
    e=once(e,'r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes;',
           f'r["extra_gpu_bytes"]=p.metadata_bytes+p.trail_floats*4+observation_bytes+p.tile_ids.size()*{macros}*4;r["macro_slab_count"]={macros};')
    e=once(e,'+size_t(depth)*observation.keys.size()*sizeof(float)+moving_lookup_bytes_;',f'+size_t(depth)*{macros}*observation.keys.size()*sizeof(float)+moving_lookup_bytes_;')
    e=once(e,'    moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_;',
           f'''    moving_macro_slab_=BeamzBuildNativeMovingMacroSlab(moving_plan_,{macros});
    moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_+moving_macro_slab_.device_bytes();''')
    e=once(e,'    moving_workspace_.tile_ids_device=static_cast<int32_t*>(upload(moving_plan_.tile_ids.data(),moving_plan_.tile_ids.size()*4));',
           '''    moving_workspace_.slab_tile_ids_device=static_cast<int32_t*>(upload(moving_macro_slab_.tile_ids.data(),moving_macro_slab_.device_bytes()));
    moving_workspace_.slab_policy_offsets_host=moving_macro_slab_.policy_offsets.data();
    moving_workspace_.slab_noio_counts_host=moving_macro_slab_.uniform_noio_counts.data();moving_workspace_.slab_observer_counts_host=moving_macro_slab_.uniform_observer_counts.data();
    moving_workspace_.slab_level_count=moving_macro_slab_.level_count;
    moving_workspace_.tile_ids_device=static_cast<int32_t*>(upload(moving_plan_.tile_ids.data(),moving_plan_.tile_ids.size()*4));''')
    e=once(e,'upload(nullptr,size_t(depth)*observation.keys.size()*4)',f'upload(nullptr,size_t(depth)*{macros}*observation.keys.size()*4)')
    anchor='    append_bytes(&moving_workspace_.parallel_policies,sizeof(bool));'
    e=once(e,anchor,anchor+f'''
    const int slab_macros={macros};append_bytes(&slab_macros,sizeof(int));
    append_bytes(moving_macro_slab_.tile_ids.data(),moving_macro_slab_.tile_ids.size()*4);
    append_bytes(moving_macro_slab_.policy_offsets.data(),moving_macro_slab_.policy_offsets.size()*4);
    append_bytes(moving_macro_slab_.uniform_noio_counts.data(),moving_macro_slab_.uniform_noio_counts.size()*4);
    append_bytes(moving_macro_slab_.uniform_observer_counts.data(),moving_macro_slab_.uniform_observer_counts.size()*4);''')
    e=once(e,'        append_pointer(moving_workspace_.tile_ids_device);',
           '        append_pointer(moving_workspace_.slab_tile_ids_device);append_pointer(moving_workspace_.tile_ids_device);')
    e=once(e,'int EnqueueMovingDft(const BeamzProgramLaunch&program,int offset){',
           'int EnqueueMovingDft(const BeamzProgramLaunch&program,int offset,int raw_step_offset=0){')
    e=once(e,'field.data=obs.values+size_t(version)*obs.key_count;',
           'field.data=obs.values+size_t(raw_step_offset+version)*obs.key_count;')
    begin,end,run=block(e,'nb::dict RunMoving(')
    a,b,loop=block(run,'for(int offset=0;offset+depth<=count;offset+=depth)')
    # Each complete slab rejoins before original chronological DFT kernels.
    loop='''for(int offset=0;offset+depth<=count;){
            BindOutputs();BeamzNativeLaunchStorage launch;
            BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&launch);
            if(offset+SLAB_MACROS*depth<=count){
              int error=BeamzEnqueueNativeMacroSlab(stream_.value,launch.program,other_.data(),offset,moving_workspace_);if(error)return static_cast<cudaError_t>(error);
              for(int m=0;m<SLAB_MACROS;++m){error=EnqueueMovingDft(launch.program,offset+m*depth,m*depth);if(error)return static_cast<cudaError_t>(error);}
              offset+=SLAB_MACROS*depth;
            }else{
              int error=BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);if(error)return static_cast<cudaError_t>(error);
              error=EnqueueMovingDft(launch.program,offset);if(error)return static_cast<cudaError_t>(error);offset+=depth;
            }
          }'''.replace('SLAB_MACROS',str(macros))
    if native_banked:
        loop=once(loop,'if(error)return static_cast<cudaError_t>(error);offset+=depth;',
                  'if(error)return static_cast<cudaError_t>(error);std::swap(current_,other_);offset+=depth;')
    run=run[:a]+loop+run[b:]
    run=run.replace('native-asymmetric-physical-frontier-v1',f'native-asymmetric-macro-slab{macros}-v1')
    if native_banked:run=run.replace('native-asymmetric-moving-v1',f'native-banked-macro-slab{macros}-v1')
    run=run.replace('packed_frontier_conversion_moving_sources_raw_observations_DFT_tails_and_clock_advance',
                    'packed_frontier_macro_pipeline_sources_raw_observations_chronological_DFT_tails_and_clock_advance')
    e=e[:begin]+run+e[end:]
    e=once(e,'NB_MODULE(_cuda_native, module) {',f'NB_MODULE(_cuda_native, module) {{\n module.attr("moving_macro_slab_count")={macros};')
    e=once(e,'    result["moving_observation_bytes"]=moving_observation_bytes_;',
           f'    result["moving_macro_slab_count"]={macros};result["moving_macro_slab_levels"]=moving_macro_slab_.level_count;\n    result["moving_observation_bytes"]=moving_observation_bytes_;')
    if native_banked:
        planner='  const auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));'
        e=once(e,planner,'''  auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));
  BeamzCompactNativeMovingUniformNoIO(p,false);BeamzCompactNativeMovingUniformObservers(p,false);
  const auto slab=BeamzBuildNativeMovingMacroSlab(p,MACROS);
  p.metadata_bytes+=size_t(p.epochs)*(slab.max_frontier-p.max_frontier)*sizeof(uint64_t);
  p.trail_floats=size_t(p.epochs)*slab.max_frontier*depth*6*p.surface_points;'''.replace('MACROS',str(macros)))
        e=once(e,f'+observation_bytes+p.tile_ids.size()*{macros}*4;',f'+observation_bytes+p.tile_ids.size()*{macros}*8;')
        e=once(e,f'    moving_macro_slab_=BeamzBuildNativeMovingMacroSlab(moving_plan_,{macros});',
               f'''    moving_macro_slab_=BeamzBuildNativeMovingMacroSlab(moving_plan_,{macros});
    moving_plan_.metadata_bytes+=size_t(moving_plan_.epochs)*(moving_macro_slab_.max_frontier-moving_plan_.max_frontier)*sizeof(uint64_t);
    moving_plan_.trail_floats=size_t(moving_plan_.epochs)*moving_macro_slab_.max_frontier*depth*6*moving_plan_.surface_points;''')
        e=once(e,'+moving_macro_slab_.device_bytes();','+moving_macro_slab_.device_bytes()+moving_macro_slab_.slots.size()*4;')
        e=once(e,'    moving_workspace_.slab_level_count=moving_macro_slab_.level_count;',
               '''    moving_workspace_.slab_level_count=moving_macro_slab_.level_count;
    moving_workspace_.slab_slots_device=static_cast<int32_t*>(upload(moving_macro_slab_.slots.data(),moving_macro_slab_.slots.size()*4));
    moving_workspace_.slab_max_frontier=moving_macro_slab_.max_frontier;moving_workspace_.slab_macro_stride=moving_macro_slab_.macro_level_stride;''')
        e=once(e,'size_t(moving_plan_.epochs)*moving_plan_.max_frontier*sizeof(uint64_t),true)',
               'size_t(moving_plan_.epochs)*moving_macro_slab_.max_frontier*sizeof(uint64_t),true)')
        e=once(e,'        append_pointer(moving_workspace_.slab_tile_ids_device);',
               '        append_pointer(moving_workspace_.slab_slots_device);append_pointer(moving_workspace_.slab_tile_ids_device);')
        anchor='    append_bytes(moving_macro_slab_.tile_ids.data(),moving_macro_slab_.tile_ids.size()*4);'
        e=once(e,anchor,anchor+'''
    append_bytes(moving_macro_slab_.slots.data(),moving_macro_slab_.slots.size()*4);
    append_bytes(&moving_macro_slab_.max_frontier,sizeof(int));append_bytes(&moving_macro_slab_.macro_level_stride,sizeof(int));''')
    shutil.copytree(source,output)
    for name,value in [('asymmetric_native.cuh',c),('native_moving.h',h),('native_extension.cc',e)]:
        (output/'src'/name).write_text(value)
    planner=Path('scripts/cuda_native_macro_slab.h');shutil.copy2(planner,output/'src/native_macro_slab.h')
    proof=dict(schema='beamz.cross-macro-temporal-pipeline.v1',parent_consumer_sha256=expected,consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),
               extension_sha256=hashlib.sha256(e.encode()).hexdigest(),header_sha256=hashlib.sha256(h.encode()).hexdigest(),planner_sha256=hashlib.sha256(planner.read_bytes()).hexdigest(),
               macro_count=macros,native_banked=native_banked,expected_kernels=68 if native_banked else 69,body_bindings=edits,
               physics='Canonical FP32 cell helper and all floating-point rounding operations unchanged; source and raw times have one integer macro offset',
               observations='Independent raw rows per macro, unchanged DFT kernels run in original chronological order after complete slab join',
               allocation='Raw slab and encoded tile GPU bytes included in CPU plan and actual-free admission before allocation'+('; merged compact frontier, native bank parity, encoded slots and larger owner tags' if native_banked else ''),GPU_executed=False)
    (output.parent/'macro-pipeline-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--macros',type=int,default=2);p.add_argument('--native-banked',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,a.macros,a.native_banked)))
