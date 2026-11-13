"""Prune inactive traced CPML profiles and optionally stage aligned vectors.

Every surviving phase/term/kind retains its independent24FP32 words. The
canonical cell trace and four-barrier field schedule are unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha
from scripts.cuda_native_geometry_ir import derive_geometry


def helper(vector):
    terms=derive_geometry()['cpml_terms'];slots={}
    for axis in range(3):
        slots[axis]=sorted(6*x['phase']+int(x['term'][-1]) for x in terms if x['axis_zyx']==axis)
        assert len(slots[axis])==4
    code='template<int Axes>__device__ __forceinline__ int SelectedProfileSlot(int i){\n'
    for axis,values in slots.items():
        expression=':'.join(f'i=={j}?{v}' for j,v in enumerate(values[:-1]))+':'+str(values[-1])
        code+=f' if constexpr(Axes=={1<<axis})return {expression};\n'
    code+=' return i;\n}\n'
    chunk=4 if vector else 1
    code+=f'''template<int Axes>__device__ __forceinline__ void StageSelectedProfiles(const Parameters&p,float*profiles){{
 constexpr bool selected=Axes==1||Axes==2||Axes==4;
 constexpr int Chunk={chunk},Count=(selected?4:12)*3*24/Chunk;
 for(int i=threadIdx.x;i<Count;i+=blockDim.x){{
  const int local=i/(72/Chunk),slot=SelectedProfileSlot<Axes>(local),kind=(i/(24/Chunk))%3,group=i%(24/Chunk);
  const auto&psi=p.phase[slot/6].psi[slot%6];const float*input=kind==0?psi.a:kind==1?psi.b:psi.inv;
  const int output=(slot*3+kind)*24+group*Chunk;
'''
    if vector:
        code+='''  if(!(reinterpret_cast<uintptr_t>(input)&15)){
   reinterpret_cast<float4*>(profiles+output)[0]=__ldg(reinterpret_cast<const float4*>(input)+group);
  }else{
   #pragma unroll
   for(int j=0;j<4;++j)profiles[output+j]=__ldg(input+group*4+j);
  }
'''
    else:code+='  profiles[output]=__ldg(input+group);\n'
    code+=' }\n}\n'
    return code,slots


def stage(source,output,vector=True):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    generated,slots=helper(vector);text=original
    marker='// Fast is admitted only for source/CPML/observation/native-wall-free OWNED'
    if text.count(marker)!=1:raise ValueError('Profile helper insertion changed')
    text=text.replace(marker,generated+'\n'+marker)
    old='if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);'
    if text.count(old)!=2:raise ValueError('Exact two cached body staging sites required')
    text=text.replace(old,'if constexpr(CachedProfiles)StageSelectedProfiles<ActiveAxes>(p,profiles);')
    if vector:
        text=text.replace('constexpr size_t ProfileCachedSharedBytes=SharedBytes+864ULL*sizeof(float);',
                          'constexpr size_t ProfileCacheOffset=(SharedBytes+15)&~size_t(15);\nconstexpr size_t ProfileCachedSharedBytes=ProfileCacheOffset+864ULL*sizeof(float);')
        text=text.replace('constexpr size_t FixedCubeSharedBytes=SharedBytes+864ULL*sizeof(float);','constexpr size_t FixedCubeSharedBytes=ProfileCachedSharedBytes;')
        text=text.replace('FixedCubeSharedBytes==55204','FixedCubeSharedBytes==55216').replace('ProfileCachedSharedBytes==55204','ProfileCachedSharedBytes==55216')
        text=text.replace('float*profiles=CachedProfiles?frame+SharedBytes/sizeof(float):nullptr;',
                          'float*profiles=CachedProfiles?frame+ProfileCacheOffset/sizeof(float):nullptr;')
        # Both cached consumers explicitly align their shared base for float4.
        text=text.replace('extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);',
                          'extern __shared__ __align__(16) float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);')
    # Each stage writes exactly the coefficient addresses reachable by the
    # selected axis guard, with all three independent kinds for H and E.
    checks=0
    for axis,selected in slots.items():
        expected={(s,k,j) for s in selected for k in range(3) for j in range(24)}
        actual=set()
        chunk=4 if vector else 1
        for i in range(4*3*24//chunk):
            slot=selected[i//(72//chunk)];kind=(i//(24//chunk))%3;group=i%(24//chunk)
            actual.update((slot,kind,group*chunk+j) for j in range(chunk))
        assert actual==expected;checks+=len(actual)
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    record=dict(schema='beamz.trace-selected-independent-profiles.v1',vector=vector,axis_slots=slots,
                parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
                coefficient_word_checks=checks,selected_words=288,original_words=864,
                vector_alignment_guard_and_scalar_fallback=vector,cached_shared_bytes=55216 if vector else 55204,
                profile_equality_assumptions=False,expected_kernels=66,GPU_executed=False)
    (output.parent/'selected-profile-proof.json').write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--scalar',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,not a.scalar)))
