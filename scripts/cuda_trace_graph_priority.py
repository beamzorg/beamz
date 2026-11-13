"""Give source-bearing trace branches priority in captured CUDA graphs.

Only host stream construction and graph instantiation change. CUDA copies
stream priorities to captured kernel nodes; UseNodePriority opts into them.
The chosen priorities are scheduling hints and never alter graph dependencies.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_bytes()
    if sha(original)!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    extension=(source/'src/native_extension.cc').read_text();graph=(source/'src/graph.cu').read_text()
    anchor='Stream() { Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }'
    assert extension.count(anchor)==1
    extension=extension.replace(anchor,anchor+'\n  explicit Stream(int priority) { Check(cudaStreamCreateWithPriority(&value,cudaStreamNonBlocking,priority),"create prioritized research policy stream"); }')
    old='while(moving_policy_streams_.size()<6)moving_policy_streams_.push_back(std::make_unique<Stream>());'
    assert extension.count(old)==1
    new='''int research_low_priority=0,research_high_priority=0;
    Check(cudaDeviceGetStreamPriorityRange(&research_low_priority,&research_high_priority),"query research policy priorities");
    while(moving_policy_streams_.size()<6){const int policy=int(moving_policy_streams_.size());
     moving_policy_streams_.push_back(std::make_unique<Stream>(policy>=3?research_high_priority:research_low_priority));
    }'''
    extension=extension.replace(old,new)
    old='cudaGraphInstantiate(&executable, graph, 0)'
    assert graph.count(old)==2
    graph=graph.replace(old,'cudaGraphInstantiate(&executable, graph, cudaGraphInstantiateFlagUseNodePriority)')
    shutil.copytree(source,output);(output/'src/native_extension.cc').write_text(extension);(output/'src/graph.cu').write_text(graph)
    proof=dict(schema='beamz.traced-graph-priority.v1',source_policies_high_priority=[3,4,5],other_policies_low_priority=[0,1,2],
               graph_node_priority_enabled=True,device_consumer_sha256=PARENT_CONSUMER,FP_and_graph_dependencies_unchanged=True,
               extension_sha256=sha(extension.encode()),graph_sha256=sha(graph.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               expected_kernels=66,GPU_executed=False)
    (output.parent/'graph-priority-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
