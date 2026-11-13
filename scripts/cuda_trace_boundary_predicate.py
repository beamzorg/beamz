"""Use an existing traced boundary fact inside one body, without cloning it.

Bit32 is the immutable host certificate for possible native-wall/support
contact anywhere in the complete tile trajectory. CPML remains independent.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block
from scripts.cuda_trace_safe_noio import prove


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    planner=(source/'src/native_moving.cc').read_text()
    assert 'if(low[a]<=0||high[a]+1>=minimum[a])flags|=kNativeBoundary;' in planner
    text=original
    old='if constexpr(SupportSafe)return true;'
    assert text.count(old)==2
    text=text.replace(old,old+'\n  if(!(flags&32))return true;')
    old='if constexpr(!SupportSafe){if(phase){'
    assert text.count(old)==2
    text=text.replace(old,'if constexpr(!SupportSafe){if((flags&32)&&phase){')
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.traced-boundary-predicate.v1',parent_sha256=PARENT_CONSUMER,
               consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),proof=prove(),
               changes=['Valid fast predicate','Difference native-wall predicate'],CPML_and_import_checks_retained=True,
               additional_launches=0,body_clones=0,expected_kernels=66,GPU_executed=False)
    (output.parent/'boundary-predicate-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
