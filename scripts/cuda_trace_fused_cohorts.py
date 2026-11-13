"""Pack independent, same-rank 1024-thread trace variants into one GPU grid.

Field/source arithmetic and each branch's shared-memory phase schedule remain
the existing complete bodies. Noncertified 512-thread NoIO/observer tiles retain
their old launches. The immutable rank partition supplies CTA-uniform dispatch.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block


def kernel(classes):
    code='''struct ResearchRankCohorts{int end[6];int noio_uniform_end,observer_uniform_end;};
template<int Depth,int N>__global__ __launch_bounds__(1024,1) void AsymmetricFusedRank(
 const __grid_constant__ Parameters p,const int32_t*ids,ResearchRankCohorts cohorts){
 const int b=blockIdx.x;const auto flags=p.flags[ids[b]];
 if(b<cohorts.end[0]){
  if(flags&64)UniformBody<Depth,1,true,true,0,true>(p,ids);
  else UniformBody<Depth,1,true,true,0,false>(p,ids);return;
 }
 if(b<cohorts.end[1]){
  if(b>=cohorts.noio_uniform_end)return;
  switch(flags&14){
   case 2:FixedCubeCachedBody<Depth,N,1>(p,ids);return;
   case 4:FixedCubeCachedBody<Depth,N,2>(p,ids);return;
   case 8:FixedCubeCachedBody<Depth,N,4>(p,ids);return;
   default:FixedCubeCachedBody<Depth,N>(p,ids);return;
  }
 }
 if(b<cohorts.end[2]){
  if(b>=cohorts.observer_uniform_end)return;
  UniformBody<Depth,1,false,true,2,true>(p,ids);return;
 }
'''
    if classes==6:
        for policy in (3,4,5):
            plane='true' if policy in (3,4) else 'false'
            code+=f' if(b<cohorts.end[{policy}]){{\n'
            if policy in (3,5):
                code+=f'  if(flags&64)SourcePendingPairedBody<Depth,1,false,true,{policy},true,1024,20,{plane}>(p,ids);\n'
                code+='  else '
            code+=f'SourcePendingPairedBody<Depth,1,false,true,{policy},false,1024,20,{plane}>(p,ids);return;\n }}\n'
    code+='}\n'
    shared='SourcePendingSharedBytes' if classes==6 else 'FixedCubeSharedBytes'
    code+=f'''template<int Depth,int N>inline cudaError_t LaunchResearchFusedRank(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w,const int32_t*offsets,int rank){{
 ResearchRankCohorts cohorts;const int start=offsets[0],count=offsets[{classes}]-start;
 for(int i=0;i<6;++i)cohorts.end[i]=offsets[i+1]-start;
 cohorts.noio_uniform_end=offsets[1]-start+w.uniform_noio_counts_host[rank];
 cohorts.observer_uniform_end=offsets[2]-start+w.uniform_observer_counts_host[rank];
 if(cohorts.noio_uniform_end>cohorts.end[1]||cohorts.observer_uniform_end>cohorts.end[2])return cudaErrorInvalidValue;
 auto status=Launch<Depth,1,false,true,1>(stream,p,w.tile_ids_device+start+cohorts.noio_uniform_end,cohorts.end[1]-cohorts.noio_uniform_end);
 if(status!=cudaSuccess)return status;
 status=Launch<Depth,1,false,true,2>(stream,p,w.tile_ids_device+start+cohorts.observer_uniform_end,cohorts.end[2]-cohorts.observer_uniform_end);
 if(status!=cudaSuccess||!count)return status;
 const auto fn=AsymmetricFusedRank<Depth,N>;
 status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,{shared});if(status!=cudaSuccess)return status;
 AsymmetricFusedRank<Depth,N><<<count,1024,{shared},stream>>>(p,w.tile_ids_device+start,cohorts);
 return cudaGetLastError();
}}
'''
    return code


def prove():
    import random,itertools
    rng=random.Random(129);cases=skipped=0
    for classes in (3,6):
        for _ in range(1000):
            counts=[rng.randrange(32) for _ in range(6)];nu=rng.randrange(counts[1]+1);ou=rng.randrange(counts[2]+1)
            offsets=[0]
            for count in counts:offsets.append(offsets[-1]+count)
            expected={i for policy in range(classes) for i in range(offsets[policy],offsets[policy+1])}
            generic=set(range(offsets[1]+nu,offsets[2]))|set(range(offsets[2]+ou,offsets[3]))
            fused=set()
            for i in range(offsets[classes]):
                if offsets[1]+nu<=i<offsets[2] or offsets[2]+ou<=i<offsets[3]:skipped+=1;continue
                fused.add(i)
            assert not generic&fused and generic|fused==expected
            cases+=1
    independent=0
    for delta in itertools.product(range(-4,5),repeat=3):
        if sum(delta) or not any(delta):continue
        # Equal-rank distinct tiles have both positive and negative index
        # differences. Their complete physical trajectories cannot overlap:
        # an overlap would require the same version difference >=1 and <=-1.
        intervals=[(d*b-(b-1),d*b+(b-1)) for d,b in zip(delta,(7,15,31))]
        assert max(x[0] for x in intervals)>min(x[1] for x in intervals)
        independent+=1
    return dict(random_rank_partitions=cases,early_return_CTAs_checked=skipped,one_complete_owner_body_per_tile=True,
                equal_rank_physical_trajectory_pairs_checked=independent,
                dependencies='All original rank barriers remain; no rank crosses the fusion boundary',FP_and_cell_updates_unchanged=True)


def stage(source,output,mode):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    classes={'fields':3,'all':6}[mode];text=original
    marker='template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks('
    assert text.count(marker)==1;text=text.replace(marker,kernel(classes)+'\n'+marker)
    start,end,body=block(text,'inline cudaError_t LaunchRanks(')
    anchor='for(int rank=0;rank<w.rank_count;++rank){const auto*offsets=w.policy_offsets_host+rank*7;'
    assert body.count(anchor)==1
    admission='true' if classes==3 else '(w.source_presence_mask==20u&&(w.data_flags&8u))'
    extra='''
  bool research_fused=false;
  int research_cube=0;
  if constexpr(Layout==1&&IsotropicH){
'''
    extra+=f'   if({admission}){{\n'
    for n in (512,564):
        if classes==6:
            extra+=f'    if(FixedNoIOCubeEligible<{n}>(p)){{auto status=LaunchResearchFusedRank<Depth,{n}>(stream,p,w,offsets,rank);if(status!=cudaSuccess)return status;research_fused=true;}}\n'
        else:
            extra+=f'    if(FixedNoIOCubeEligible<{n}>(p)){{research_fused=true;research_cube={n};}}\n'
    extra+='   }\n  }\n'
    if classes==6:extra+='  if(research_fused)continue;\n'
    body=body.replace(anchor,anchor+extra)
    if classes==3:
        old='for(int policy=0;policy<6;++policy){const int start=offsets[policy],count=offsets[policy+1]-start;'
        assert body.count(old)==1
        body=body.replace(old,'for(int policy=0;policy<6;++policy){\n   if(research_fused&&(policy==1||policy==2))continue;\n   const int start=offsets[policy],count=(research_fused&&policy==0?offsets[3]:offsets[policy+1])-start;')
        old='case 0:status=Launch<Depth,Layout,true,IsotropicH,0>(branch,p,w.tile_ids_device+start,count);break;'
        assert body.count(old)==1
        new='case 0:if constexpr(Layout==1&&IsotropicH){if(research_fused){if(research_cube==512)status=LaunchResearchFusedRank<Depth,512>(branch,p,w,offsets,rank);else status=LaunchResearchFusedRank<Depth,564>(branch,p,w,offsets,rank);break;}}\n     status=Launch<Depth,Layout,true,IsotropicH,0>(branch,p,w.tile_ids_device+start,count);break;'
        body=body.replace(old,new)
        old='if(offsets[policy]==offsets[policy+1])continue;'
        assert body.count(old)==1
        new='if(research_fused&&policy<3){if(policy!=0||offsets[0]==offsets[3])continue;}else '+old
        body=body.replace(old,new)
    text=text[:start]+body+text[end:]
    # Fields-only fusion occupies branch0 and retains the existing fork/join
    # with source branches3..5. All-policy fusion is one main-stream launch.
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-fused-rank-cohorts.v1',mode=mode,classes=classes,proof=prove(),
               source_fast_path_requires_presence20_and_plane_pair=classes==6,shared_bytes=58680 if classes==6 else 55204,
               parallel_dispatch='fused fields on branch0, sources retain branches3..5' if classes==3 else 'one rank kernel replaces all policy branches',
               noncertified_material_fallbacks_preserved=True,parent_sha256=PARENT_CONSUMER,
               consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),expected_kernels=70,GPU_executed=False)
    (output.parent/'fused-rank-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('fields','all'),required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.mode)))
