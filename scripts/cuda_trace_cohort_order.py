"""Schedule independent traced policy cohorts in a chosen launch order.

Each rank retains the exact fork/join dependencies. The immutable policy
partition and every device body remain unchanged. Reordering applies to both
sequential launch construction and captured parallel branches, so numerical
gates and actual activity evidence remain required.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block
from scripts.cuda_trace_fused_cohorts import prove


def stage(source,output,order):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    orders={'heavy-first':(5,4,3,2,1,0),'source-noio-first':(5,1,3,2,0,4)}
    sequence=orders[order];assert set(sequence)==set(range(6))
    start,end,body=block(original,'inline cudaError_t LaunchRanks(')
    old='for(int policy=0;policy<6;++policy){const int start=offsets[policy],count=offsets[policy+1]-start;'
    assert body.count(old)==1
    replacement='constexpr int ResearchPolicyOrder[6]={'+','.join(map(str,sequence))+'};\n  for(int dispatch=0;dispatch<6;++dispatch){const int policy=ResearchPolicyOrder[dispatch],start=offsets[policy],count=offsets[policy+1]-start;'
    body=body.replace(old,replacement);text=original[:start]+body+original[end:]
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-policy-launch-order.v1',order=sequence,device_bodies_unchanged=True,
               rank_fork_join_unchanged=True,physical_independence=prove(),
               parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               expected_kernels=66,GPU_executed=False)
    (output.parent/'cohort-order-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--order',choices=('heavy-first','source-noio-first'),required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.order)))
