"""Expose proved low-X face topology to persistent-only import lowering.

The generic import computes a long-lived neighbor byte offset from thread ID.
Its low-X caller guarantees dz>=1,dy>=1,dx=0, making that offset exactly32B.
The dedicated import retains canonical support masks, .cg loads and bad flags.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once


def lowx_trace(code,original):
    a=original.index('template<int Depth,bool Fast>\n__device__ __forceinline__ FieldPacket ImportPacket(')
    b=original.index('__global__ void BeginEpoch',a)
    packet=original[a:b]
    packet=once(packet,'FieldPacket ImportPacket(', 'FieldPacket PersistentImportLowX(')
    packet=once(packet,'int oz,int oy,int ox,int dz,int dy,int dx){','int oz,int oy,int ox,int dz,int dy){')
    packet=once(packet,'ox-1+dx-version','ox-1-version')
    packet=once(packet,'const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);','const int neighbor=4; // proved dz>=1,dy>=1,dx==0')
    packet=once(packet,'Surface(dz?dz-1:BZ-1,dy?dy-1:BY-1,dx?dx-1:BX-1)', 'Surface(dz-1,dy-1,BX-1)')
    a=original.index('template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageRegisterLow(')
    b=original.index('template<int T,int Slots>',a)
    stage=original[a:b].replace('void StageRegisterLow(', 'void PersistentStageRegisterLow(')
    stage=once(stage,'ImportPacket<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy,0)',
                     'PersistentImportLowX<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy)')
    a=original.index('template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageSourcePendingH(')
    b=original.index('// Two-plane publication groups;',a)
    paired=original[a:b].replace('void StageSourcePendingH(', 'void PersistentStageSourcePendingH(')
    paired=once(paired,'ImportPacket<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy,0)',
                       'PersistentImportLowX<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy)')
    # Insert after the environment so the actual-control extraction stays a
    # pure protocol unit, without introducing cell types into its CPU harness.
    match=re.search(r'template<[^\n]+>\n__device__ __forceinline__ void PersistentPairedBody',code)
    if not match:raise ValueError('Expected persistent paired body')
    code=code[:match.start()]+packet+stage+paired+'\n'+code[match.start():]
    if 'StageRegisterLow<Depth,SupportSafe,T>(p,low_stage' in code:
        code=once(code,'StageRegisterLow<Depth,SupportSafe,T>(p,low_stage',
                       'PersistentStageRegisterLow<Depth,SupportSafe,T>(p,low_stage')
    code=once(code,'StageSourcePendingH<Depth,SupportSafe,T>(p,low_stage',
                   'PersistentStageSourcePendingH<Depth,SupportSafe,T>(p,low_stage')
    return code,packet


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint copied source required')
    old=(source/'src/native_persistent.cuh').read_text()
    original=(source/'src/asymmetric_native.cuh').read_text()
    new,packet=lowx_trace(old,original);shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'lowx_import_sha256':sha(packet.encode()),
        'workspace_abi_changed':False,'expected_entries':65,'original_consumer_unchanged':True,
        'fact':'active StageLow i in[0,104] => dz=1+i/15 in[1,7],dy=1+i%15 in[1,15],dx=0 => neighbor4 exactly',
        'preserved':'same support mask, same Surface priority/index, same six cg scalar loads, same local stage writers, same bad flags',
        'rationale':'SASS specialized-v1 spills one generic neighbor byte offset in prologue and reloads it only for active low-X staging threads',
        'requires':['address/support proof','actual control proof','resource gate','all30 exact','sanitizers','paired timings'],'gpu_executed':False}
    (output.parent/'persistent-lowx-trace-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
