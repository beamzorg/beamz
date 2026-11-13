"""Generate phase-wide CPML guard scheduling from the typed cell trace.

Pairs are grouped by their geometric active-axis predicate. Each term keeps
its own a/b/inv profile pointers and recurrence: no coefficient equality is
assumed. A native byte-range certificate protects reordered mutable stores.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,block,sha
from scripts.generate_cuda_native_phase import plan_phase
from scripts.cuda_readonly_profile_bindings import bind_profiles,EqualAxisProfileClass
from scripts.cuda_space_time_ir import FIELDS,yee_operator
from scripts.cuda_stencil_trace import Expr,emit_cuda
from scripts.cuda_trace_specialize import specialize_scalar_class,MagneticScalarClass
from scripts.cuda_trace_runtime_uniform import specialize_runtime_e,UniformElectricRegionClass
from scripts.cuda_native_phase_state_fact import generate_guard


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Reviewed anchor changed:'+old[:100])
    return text.replace(old,new,1)


def generate():
    operator,_=specialize_scalar_class(yee_operator(cpml_axes=(0,1,2)),MagneticScalarClass())
    operator,_=specialize_runtime_e(operator,UniformElectricRegionClass())
    # Existing planner proves output reconstruction and groups the canonical
    # same-axis terms. Its alias slots are used as classification keys only.
    # Emission below loads every original term independently.
    grouping=bind_profiles(operator,EqualAxisProfileClass())
    ids={name:i for i,name in enumerate(FIELDS)}
    code=['#pragma once','namespace beamz::cuda::trace_grouped_phase {',
          'struct Result{float x,y,z;};',
          'template<int Phase,int ActiveAxes,class Environment,class ReadField>',
          '__device__ __forceinline__ Result Advance(Environment&env,ReadField read,int version,int z,int y,int x){',
          'static_assert(!Environment::HasSources&&!Environment::HasObserve,"NoIO only");',
          'Result result{0.f,0.f,0.f};']
    plans=[]
    for phase in (0,1):
        pairs,outputs,proof=plan_phase(operator,phase,grouping)
        proof['profile_slots_are_grouping_keys_only']=True
        proof['runtime_profile_equality_required']=False
        plans.append(proof)
        code.append(f'if constexpr(Phase=={phase}){{')
        for c in range(3):code.append(f'const bool valid_{c}=env.Valid({3*phase+c},z,y,x);')
        for slot in range(phase*6,phase*6+6):code.append(f'float correction_{slot}=0.f;')
        def read_name(r):
            if r.field not in ids:raise ValueError('Mutable read escaped its recurrence')
            return f'read({ids[r.field]},{"version" if r.version else "(version-1)"},{r.offset[0]},{r.offset[1]},{r.offset[2]})'
        def coefficient(r):
            for prefix,kind in (('source_',1),('decay_',0)):
                if r.name.startswith(prefix) and r.name[len(prefix):] in ids:
                    return f'env.Coefficient({kind},{ids[r.name[len(prefix):]]},z,y,x)'
            raise ValueError('Profile escaped its independent term scope')
        def native(value):
            if value.op=='mul_rn':
                difference,scale=value.args
                name=scale.args[0] if scale.op=='parameter' else scale.args[0].name if scale.op=='coefficient' else None
                if name=='inv_spacing' and difference.op=='sub' and all(a.op=='read' for a in difference.args):
                    operands=[a.args[0] for a in difference.args]
                    center=next((r for r in operands if not any(r.offset)),None)
                    adjacent=next((r for r in operands if any(r.offset)),None)
                    if center and adjacent and center.field==adjacent.field and center.version==adjacent.version:
                        axis=next(a for a,d in enumerate(adjacent.offset) if d)
                        return Expr('parameter',(f'env.Difference({phase},{ids[center.field]},{axis},z,y,x,({read_name(center)}),({read_name(adjacent)}))',))
            return Expr(value.op,tuple(native(a) if isinstance(a,Expr) else a for a in value.args))
        completed,emitted=set(),set()
        for representative,pair in pairs:
            axis=pair[0].term.a.args[0].axes[0]
            code.append('{')
            for item in pair:
                derivative=emit_cuda(native(item.term.derivative),read_name,coefficient)
                code += [f'float derivative_{item.slot}=0.f;',
                         f'if(valid_{item.component}){{derivative_{item.slot}={derivative};correction_{item.slot}=__fmul_rn({item.term.sign!r}f,derivative_{item.slot});}}']
            condition='||'.join(f'valid_{item.component}' for item in pair)
            code += [f'if constexpr(ActiveAxes<0||(ActiveAxes&{1<<axis})){{',
                     f'if(({condition})&&env.LocalActive({representative},z,y,x)){{']
            for item in pair:
                code += [f'if(valid_{item.component}){{',
                         f'const float a=env.Coefficient(2,{item.slot},z,y,x);',
                         f'const float b=env.Coefficient(3,{item.slot},z,y,x);',
                         f'const float inv=env.Coefficient(4,{item.slot},z,y,x);',
                         f'const float psi=__fmaf_rn(b,env.LoadLocal({item.slot},z,y,x),__fmul_rn(a,derivative_{item.slot}));',
                         f'correction_{item.slot}=__fmul_rn({item.term.sign!r}f,__fmaf_rn(derivative_{item.slot},inv,psi));',
                         f'env.StoreLocal({item.slot},z,y,x,psi);','}']
            code+=['}}','}']
            completed.update(item.slot for item in pair)
            for c in range(3):
                if c not in emitted and {phase*6+2*c,phase*6+2*c+1}<=completed:
                    def parameter(name,c=c):
                        if name in ('corrected_0','corrected_1'):return f'correction_{phase*6+2*c+int(name[-1])}'
                        raise ValueError('Unscheduled parameter')
                    expression=emit_cuda(native(outputs[c]),read_name,coefficient,parameter)
                    code.append(f'if(valid_{c})result.{"xyz"[c]}=env.Finish({phase*3+c},version,z,y,x,{expression});')
                    emitted.add(c)
        assert emitted=={0,1,2}
        code.append('}')
    code+=['return result;','}','}']
    source='\n'.join(code)+'\n'
    return source,dict(phase_plans=plans,source_sha256=sha(source.encode()),
                       independent_profile_load_slots=list(range(12)),
                       runtime_requires='NoIO, canonical CPML slabs, pairwise independent mutable states and immutable profiles')


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    generated,proof=generate()
    text=once(original,'#include "asymmetric_native_cell_generated.cuh"','#include "asymmetric_native_cell_generated.cuh"\n#include "trace_grouped_phase.cuh"')
    text=once(text,'int physical_n[3];float h_source;','int physical_n[3];bool trace_state_independent;float h_source;')
    text=once(text,'out->macro_step=macro;out->h_source=','out->macro_step=macro;out->trace_state_independent=w.trace_state_independent;out->h_source=')
    clones=[]
    for signature,newname in [('void UniformBody(','GroupedNoIOBody'),('void FixedCubeCachedBody(','GroupedFixedNoIOBody')]:
        begin,end,old=block(text,signature)
        template_start=text.rfind('template<',0,begin)
        clone=text[template_start:end].replace(signature,'void '+newname+'(',1)
        if newname=='GroupedNoIOBody':
            prefix=clone[:clone.index('{')+1]
            clone=clone.replace(prefix,'template<int Depth,int ActiveAxes=-1>\n__device__ __forceinline__ void GroupedNoIOBody(const Parameters&p,const int32_t*tiles){\n constexpr int Layout=1,Policy=1;constexpr bool Fast=false,IsotropicH=true,UniformE=true,CachedProfiles=true;',1)
        for phase,family in [(0,'h'),(1,'e')]:
            guard='c+3' if phase else 'c'
            oldloop=f'''     #pragma unroll
     for(int c=0;c<3;++c){{float value=0.f;if(env.Valid({guard},z,y,x)){{
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<{phase},0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<{phase},1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<{phase},2,false>(env,read,n,z,y,x);
     }}{family}[c][j]=value;}}'''
            replacement=f'''     const auto result=beamz::cuda::trace_grouped_phase::Advance<{phase},ActiveAxes>(env,read,n,z,y,x);
     {family}[0][j]=result.x;{family}[1][j]=result.y;{family}[2][j]=result.z;'''
            clone=once(clone,oldloop,replacement)
        clones.append(clone)
    entries='''template<int Depth,int N=0>
__global__ __launch_bounds__(1024,1) void AsymmetricTraceGroupedNoIO(const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14){
'''
    for flag,axis in [(2,1),(4,2),(8,4)]:entries+=f' case {flag}:if constexpr(N)GroupedFixedNoIOBody<Depth,N,{axis}>(p,tiles);else GroupedNoIOBody<Depth,{axis}>(p,tiles);return;\n'
    entries+=' default:if constexpr(N)GroupedFixedNoIOBody<Depth,N>(p,tiles);else GroupedNoIOBody<Depth>(p,tiles);return;\n }\n}\n'
    marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__'
    text=once(text,marker,'\n'.join(clones)+'\n'+entries+'\n'+marker)
    for n in (512,564):
        anchor=f'  if(FixedNoIOCubeEligible<{n}>(p)){{'
        selected=f'''  if(FixedNoIOCubeEligible<{n}>(p)&&p.trace_state_independent){{
   const auto kernel=AsymmetricTraceGroupedNoIO<Depth,{n}>;
   status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FixedCubeSharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricTraceGroupedNoIO<Depth,{n}><<<uniform_count,1024,FixedCubeSharedBytes,stream>>>(p,ids);return cudaGetLastError();
  }}
'''
        text=once(text,anchor,selected+anchor)
    anchor='  const auto kernel=AsymmetricNoIOUniform1024<Depth>;'
    selected='''  if(p.trace_state_independent){
   const auto kernel=AsymmetricTraceGroupedNoIO<Depth>;
   status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ProfileCachedSharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricTraceGroupedNoIO<Depth><<<uniform_count,1024,ProfileCachedSharedBytes,stream>>>(p,ids);return cudaGetLastError();
  }
'''
    text=once(text,anchor,selected+anchor)
    header=once((source/'src/native_moving.h').read_text(),'  uint32_t source_presence_mask=0;','  uint32_t source_presence_mask=0;\n  bool trace_state_independent=false;')
    host=(source/'src/native_extension.cc').read_text()
    host=once(host,'#include "native_moving.h"','#include "native_moving.h"\n#include "runtime_trace_state_fact.h"')
    host=once(host,'    BeamzNativeDeepPlan observation;','''    BeamzBuffer cache_state[12]{},cache_profile[36]{};
    for(int family=0;family<2;++family){const auto&v=family?validate.program.e_ab:validate.program.h_ab;
      for(int term=0;term<6;++term){cache_state[6*family+term]=v.inputs[31+term];
        for(int kind=0;kind<3;++kind)cache_profile[18*family+3*term+kind]=v.inputs[13+3*term+kind];}}
    moving_workspace_.trace_state_independent=BeamzNativeIndependentPhaseState(cache_state,cache_profile);
    BeamzNativeDeepPlan observation;''')
    host=once(host,'    const bool source_selector=','    append_bytes(&moving_workspace_.trace_state_independent,sizeof(bool));\n    const bool source_selector=')
    host=once(host,'    result["moving_source_presence_mask"]=','    result["moving_trace_grouped_state_independent"]=moving_workspace_.trace_state_independent;\n    result["moving_source_presence_mask"]=')
    shutil.copytree(source,output)
    for name,value in [('asymmetric_native.cuh',text),('native_moving.h',header),('native_extension.cc',host),('trace_grouped_phase.cuh',generated),('runtime_trace_state_fact.h',generate_guard()['source'])]:(output/'src'/name).write_text(value)
    proof.update(schema='beamz.independent-profile-traced-phase-scheduling.v1',consumer_sha256=sha(text.encode()),
                 parent_sha256=PARENT_CONSUMER,producer_sha256=sha(Path(__file__).read_bytes()),expected_kernels=72,GPU_executed=False)
    (output.parent/'grouped-phase-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=stage(a.source,a.output);print(json.dumps({k:r[k] for k in ('schema','consumer_sha256','source_sha256','expected_kernels')}))
