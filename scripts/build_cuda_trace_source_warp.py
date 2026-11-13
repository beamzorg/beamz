"""Stage a source-only architectural warp/storage lowering on the frozen control."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.build_cuda_trace_tile import replace_body,replace_last_bounds
from scripts.cuda_trace_phase_cycle import block
from scripts.cuda_trace_source_warp_codegen import generate
from scripts.cuda_trace_source_warp_topology import SourceWarpTopology


def stage(source,output,expected,x_slots=8,*,shared_e=False,region_clock=False,uniform_only=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:raise ValueError('New disjoint research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=expected:raise ValueError('Explicit frozen consumer identity required')
    topology=SourceWarpTopology(x_slots=x_slots);generated,proof=generate(topology,shared_e=shared_e,region_clock=region_clock)
    marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1,bool CachedProfiles=false>'
    if original.count(marker)!=1:raise ValueError('Stable source callback/compute boundary required')
    text=original.replace(marker,'#include "trace_source_warp_generated.cuh"\n'+marker)
    wrapper='''
 RunSourceWarp<Depth,Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>(p,tiles);
'''
    if uniform_only:
        start,end,_=block(original,'void SourcePendingPairedBody(')
        start=original.rfind('template<',0,start)
        control=original[start:end].replace('void SourcePendingPairedBody(', 'void ControlSourcePendingPairedBody(')
        text=text.replace('#include "trace_source_warp_generated.cuh"',control+'\n#include "trace_source_warp_generated.cuh"')
        wrapper='''
 if constexpr(UniformE)RunSourceWarp<Depth,Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>(p,tiles);
 else ControlSourcePendingPairedBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,BlockT,SourceMask,PlanePair,ActiveAxes,CachedProfiles>(p,tiles);
'''
        if topology.threads==512:
            wrapper=wrapper.replace('ControlSourcePendingPairedBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,BlockT,SourceMask,PlanePair,ActiveAxes,CachedProfiles>(p,tiles)',
                                    'UniformBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,false>(p,tiles)')
        elif topology.threads!=1024:raise ValueError('Matching control fallback thread map required')
        if proof['shared_frame_bytes']<58680:raise ValueError('Control source fallback must fit the superset frame')
    text=replace_body(text,'void SourcePendingPairedBody(',wrapper)
    start=text.index('constexpr size_t SourcePendingSharedBytes=');end=text.index('template<int Depth,int Policy>',start)
    text=text[:start]+'''constexpr size_t SourcePendingSharedBytes=SourceWarpFrameBytes;
template<int Policy>constexpr size_t ResearchSourceSharedBytes=SourcePendingSharedBytes+(Policy==5?864ULL*sizeof(float):0);
'''+text[end:]
    for signature in ('void AsymmetricSourcePending20(','void AsymmetricSourcePlanePair20('):
        start,_,_=block(text,signature);text=replace_last_bounds(text[:start],'SourceWarpThreads')+text[start:]
    for name in ('AsymmetricSourcePending20<Depth,Policy>','AsymmetricSourcePlanePair20<Depth,Policy>'):
        old=name+'<<<count,1024,ResearchSourceSharedBytes<Policy>,'
        if text.count(old)!=1:raise ValueError('Exact source launch mapping required')
        text=text.replace(old,name+'<<<count,SourceWarpThreads,ResearchSourceSharedBytes<Policy>,')
    # Independent NoIO, bulk and all physical support/source/DFT callbacks are
    # unchanged; source-only shape is a thread/storage map, not a scene rotation.
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','float Finish(','void Observe('):
        if block(text,signature)[2]!=block(original,signature)[2]:raise ValueError('Unrelated physics or compute body changed')
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    (output/'src/trace_source_warp_generated.cuh').write_text(generated)
    extension=(output/'src/native_extension.cc').read_text()
    extension=extension.replace('result["moving_source_pending_barriers"]=5;', 'result["moving_source_pending_barriers"]=2;')
    extension=extension.replace('result["moving_source_pending_shared_bytes"]=58680;',f'result["moving_source_pending_shared_bytes"]={proof["shared_frame_bytes"]};')
    extension=extension.replace('module.attr("moving_source_pending_barriers")=5;', 'module.attr("moving_source_pending_barriers")=2;')
    extension=extension.replace('module.attr("moving_source_pending_mask20_1024")=true;', f'module.attr("moving_source_pending_mask20_1024")={str(topology.threads==1024).lower()};')
    extension=extension.replace('NB_MODULE(_cuda_native, module) {',f'''NB_MODULE(_cuda_native, module) {{
 module.attr("moving_source_warp_generated")=true;
 module.attr("moving_source_pending_threads")={topology.threads};
 module.attr("moving_source_plane_pair_threads")={topology.threads};
''')
    (output/'src/native_extension.cc').write_text(extension)
    proof.update(parent_consumer_sha256=expected,consumer_sha256=hashlib.sha256(text.encode()).hexdigest(),
                 expected_kernels=68,source_only=True,uniform_only=uniform_only,control_fallback='Original paired1024' if topology.threads==1024 else 'Original generic512',physics_callbacks_unchanged=True,
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output.parent/'source-warp-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True)
    p.add_argument('--x-slots',type=int,default=8)
    p.add_argument('--shared-e',action='store_true')
    p.add_argument('--region-clock',action='store_true')
    p.add_argument('--uniform-only',action='store_true')
    a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,a.x_slots,shared_e=a.shared_e,region_clock=a.region_clock,uniform_only=a.uniform_only)))
