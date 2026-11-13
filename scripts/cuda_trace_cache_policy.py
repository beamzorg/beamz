"""Screen coherent trail-load caching and the shared/L1 partition independently.

The producer rank completes in the same stream before a consumer rank starts.
Cache-all is a coherent global load (not a read-only alias assumption). Carveout
is a preference; the actual required dynamic shared allocation is unchanged.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block


def stage(source,output,policy):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text=original;changed=0
    if policy=='trail-ca':
        start,end,body=block(text,'FieldPacket ImportPacket(')
        changed=body.count('__ldcg(')
        if changed!=6:raise ValueError('Exact six trail field loads required')
        body=body.replace('__ldcg(','__ldca(');text=text[:start]+body+text[end:]
    elif policy in ('shared50','shared100'):
        import re
        percentage=50 if policy=='shared50' else 100
        # Apply only after each existing successful dynamic-shared admission.
        pattern=r'(status=cudaFuncSetAttribute\((\w+),cudaFuncAttributeMaxDynamicSharedMemorySize,\w+\);if\(status!=cudaSuccess\)return status;)'
        def lower(match):
            return match[1]+f'\n status=cudaFuncSetAttribute({match[2]},cudaFuncAttributePreferredSharedMemoryCarveout,{percentage});if(status!=cudaSuccess)return status;'
        text,changed=re.subn(pattern,lower,text)
        if changed!=7:raise ValueError('Expected seven admitted kernel attribute sites, got '+str(changed))
    else:raise ValueError('Explicit cache policy required')
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.traced-cache-policy.v1',policy=policy,changed_sites=changed,
               parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               FP_and_dependency_schedule_unchanged=True,expected_kernels=66,GPU_executed=False)
    (output.parent/'cache-policy-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',choices=('trail-ca','shared50','shared100'),required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.policy)))
