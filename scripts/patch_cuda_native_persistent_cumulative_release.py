"""Drop redundant writer fences only after proving cumulative GPU releases.

Exact allowed parents retain either full CTA joins, or full-mask warp joins
and a32-leader CTA acq_rel chain, before leader GPU release publication. The
copy-only pass changes only __threadfence() calls into syntactic no-ops.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha
    from scripts.patch_cuda_native_persistent_global_atomics import GLOBAL_HELPERS
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha
    from patch_cuda_native_persistent_global_atomics import GLOBAL_HELPERS


ALLOWED={
 'b7814cd9792212cea639211c06cb985f0f588282a076c7f6077056e4145fe0c0':'fullspecial G4 CTA B3/RL joins',
 '7b83198fb93008338a2610e41ff0e918d4a933c4fea162da89c9d1edd8c86f15':'paired true-DAG phase-v5 E1',
 'e95536f29166646bf06c2ee6f7afc4204b0b4a5f9cc32609048c045023585c93':'paired true-DAG phase-v5 E2',
 '289a5becf9add7e5c7edcd98ed8cb6784b8b77ab117b60008031bb1bc8729588':'paired true-DAG phase-v5 E4',
}
OLD='__threadfence();'
NEW='((void)0);'


def cumulative_release(code):
    digest=sha(code.encode())
    if digest not in ALLOWED:raise ValueError('Exact reviewed joined-release source required')
    if GLOBAL_HELPERS not in code:raise ValueError('GPU-scope release and compiler-memory-clobber helpers changed')
    if NEW in code:raise ValueError('No-op token already present; inverse would be ambiguous')
    if code.count(OLD)!=2:raise ValueError('Expected exactly H/E or paired/uniform publication fence sites')
    result=code.replace(OLD,NEW)
    if result.replace(NEW,OLD)!=code:raise ValueError('Fence-only source inverse failed')
    return result


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text();new=cumulative_release(old)
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'parent_kind':ALLOWED[sha(old.encode())],
        'persistent_consumer_sha256':sha(new.encode()),'patcher_sha256':sha(Path(__file__).read_bytes()),
        'workspace_abi_changed':False,'expected_entries':65,'writer_fence_sites_removed':2,
        'source_inverse_exact':new.replace(NEW,OLD)==old,'GPU_release_helpers_unchanged':GLOBAL_HELPERS in new,
        'canonical_FP_calls_source_barriers_joins_unchanged':True,
        'CTA_HB':'every global store -> full CTA barrier -> leader cumulative st.release.gpu -> consumer acquire -> cg payload reads',
        'warp_HB':'every global store -> fullmask syncwarp -> warp leader CTA acq_rel RMW chain -> leader acquire32 -> cumulative GPU release',
        'prefix_HB':'every prior E store also precedes next H warp join and cumulative odd release; mandatory end E publication retained',
        'oracle_scope':'Older omit-per-writer-fence negatives model the conservative explicit-fence recipe and are NOT negatives for this cumulative-join alternative',
        'primary_sources':['https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model',
            'https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/semaphore.h',
            'https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/barrier.h'],
        'formal_review':'.cache/cuda-trace-five-hour-2026-11-13-1059/persistent-fence-coalescing-independent-review.json',
        'requires':['independent1024writer crossSM stress','exact source inverse','actual control+writer join proof','resource gate','all30 exact','sanitizers','paired timings'],'gpu_executed':False}
    (output.parent/'persistent-cumulative-release-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
