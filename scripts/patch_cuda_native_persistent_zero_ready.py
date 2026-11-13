"""Fix zero-ready success in a floor controller used for startup tag waits.

WaitProducer returns an acquired ready value, for which zero is valid. Every
failure path separately sets abort_local. Never use zero as its success flag.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once


def zero_ready(code):
    return once(code,'const uint32_t seen=WaitProducer(producer,owner,step);if(!seen)return;floor=seen<floor?seen:floor;',
        'const uint32_t seen=WaitProducer(producer,owner,step);if(C().abort_local)return;floor=seen<floor?seen:floor;')


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text();new=zero_ready(old)
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'workspace_abi_changed':False,'expected_entries':65,
        'cause':'A valid acquired ready0 was incorrectly treated as failure, exiting startup after only first parent tag',
        'fix':'Only abort_local signals WaitProducer failure; ready0 contributes min0 and ALL parents must still validate tags',
        'CPU_proof_gap':'Old actual-controller harness waited current ready before checking payload but omitted immediate startup base/tag validation',
        'requires':['forced ready0/late-parent negative','all startup base tags checked before initialization','actual controller proof','resources','all30 exact','sanitizers'],'gpu_executed':False}
    (output.parent/'persistent-zero-ready-fix-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
