"""Lower traced trail-write predicates to predicated PTX stores.

The predicate still controls every memory effect. Inactive lanes form a valid
clamped word index, but issue no store. Field arithmetic, ownership, rank order
and all publication/reader-retirement barriers remain unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block


HELPER=r'''
__device__ __forceinline__ void ResearchPredicatedTrailStore(float*base,size_t word,float value,bool active){
 const uintptr_t address=reinterpret_cast<uintptr_t>(base)+word*sizeof(float);
 asm volatile("{.reg .pred enabled;setp.ne.u32 enabled,%2,0;@enabled st.global.f32 [%0],%1;}"
              ::"l"(address),"f"(value),"r"(unsigned(active)):"memory");
}
'''


def transform(text):
    anchor='if(surface>=0){p.trails[';sites=text.count(anchor)
    if sites!=4:raise ValueError('Exact four complete publication sites required')
    for _ in range(sites):
        start,end,body=block(text,anchor);stores=body[body.index('{')+1:-1].split(';');stores=[s for s in stores if s]
        assert len(stores)==2
        replacement=''
        for statement in stores:
            lhs,rhs=statement.split(']=',1);assert lhs.startswith('p.trails[')
            index=lhs[len('p.trails['):]
            assert index.endswith('+surface')
            index=index[:-len('+surface')]+'+max(surface,0)'
            replacement+=f'ResearchPredicatedTrailStore(p.trails,{index},{rhs},surface>=0);'
        text=text[:start]+replacement+text[end:]
    return text,sites


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text,sites=transform(original)
    marker='// Fast is admitted only for source/CPML/observation/native-wall-free OWNED'
    assert text.count(marker)==1;text=text.replace(marker,HELPER+'\n'+marker)
    checks=0
    for version in range(1,17):
        for field in range(6):
            for surface in range(735):
                original_index=((version-1)*6+field)*735+surface
                predicated_index=((version-1)*6+field)*735+max(surface,0)
                assert original_index==predicated_index;checks+=1
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-predicated-trail-publication.v1',publication_sites=sites,
               positive_store_index_checks=checks,negative_predicate_has_no_store=True,clamped_inactive_index=True,
               parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               added_shared_bytes=0,added_barriers=0,expected_kernels=66,GPU_executed=False)
    (output.parent/'predicated-publication-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
