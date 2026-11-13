"""Copy-only resource isolation: Fast bulk; original persistent P5 otherwise."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once


def bulk_only(code):
    code=once(code,'C().body_kind=!(facts&63)?0:(!(facts&17)&&(facts&64))?1:2;',
        'C().body_kind=!(facts&63)?0:2;')
    code=once(code,'  else if(C().body_kind==1)PersistentUniformBody<Depth,false,1,true>(p);\n','')
    return code


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint copied source required')
    old=(source/'src/native_persistent.cuh').read_text();new=bulk_only(old)
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'workspace_abi_changed':False,'expected_entries':65,
        'policy':'Fast original UniformBody for flags low6==0; paired P5 for every other tile',
        'rationale':'Remove newly instantiated NoIO Uniform body with 4-byte spills at R64, while retaining bulk specialization',
        'gpu_executed':False,'requires':['zero resources','actual controller proof','all30 exact','sanitizers','paired timings']}
    (output.parent/'persistent-bulk-only-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
