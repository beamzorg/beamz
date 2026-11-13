"""Stage exact low-X trail values with SM80+ async global-to-shared copies.

The immutable lower-rank source addresses are resolved by the same checked
ImportPacket prefix. Invalid lanes keep exact positive-zero/NaN behavior.
Every thread commits its group before H work and waits before the original
B1 CTA publication/retirement barrier; no arithmetic or phase barrier changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_plane_retirement_binding import braced_body
from scripts.patch_cuda_native_stage_prefetch import (
    B1,
    CALL,
    HELPER_HASHES,
    LOOP,
    OLD,
    prove_stage,
)
from scripts.patch_cuda_native_stage_prefetch import (
    transform as early_transform,
)

ASYNC_CALL = (
    "  StageRegisterLowAsync<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);\n"
)
WAIT = "  __pipeline_wait_prior(0); // Complete this thread's copies before existing B1 publication.\n"
PACKET_RETURN = """ return {mask&1?point[0*SurfacePoints]:0.f,mask&2?point[1*SurfacePoints]:0.f,
         mask&4?point[2*SurfacePoints]:0.f,mask&8?point[3*SurfacePoints]:0.f,
         mask&16?point[4*SurfacePoints]:0.f,mask&32?point[5*SurfacePoints]:0.f};"""
BAD_RETURN = "  return {mask&1?bad:0.f,mask&2?bad:0.f,mask&4?bad:0.f,mask&8?bad:0.f,mask&16?bad:0.f,mask&32?bad:0.f};"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def resolve_body(consumer):
    body = braced_body(consumer, "FieldPacket ImportPacket(")
    if digest(body.encode()) != HELPER_HASHES["FieldPacket ImportPacket("]:
        raise ValueError("Unreviewed original import pointer/validity contract")
    edits = [
        ("if(!mask)return {};", "if(!mask)return {nullptr,0,false};"),
        (
            "atomicOr(p.bad,2);const float bad=__int_as_float(0x7fffffff);",
            "atomicOr(p.bad,2);",
        ),
        (BAD_RETURN, "  return {nullptr,mask,true};"),
        (PACKET_RETURN, " return {point,mask,false};"),
    ]
    result = body
    for old, new in edits:
        if result.count(old) != 1:
            raise ValueError(f"Missing unique import-resolution anchor: {old!r}")
        result = result.replace(old, new, 1)
    inverse = result
    for old, new in reversed(edits):
        inverse = inverse.replace(new, old, 1)
    if inverse != body:
        raise ValueError("Original import prefix cannot be reconstructed exactly")
    return result


def helper(consumer):
    resolve = resolve_body(consumer)
    return (
        """struct AsyncTrailView {const float* point;unsigned mask;bool bad;};
template<int Depth,bool Fast>
__device__ __forceinline__ AsyncTrailView ResolveAsyncTrail(const Parameters&p,const size_t*bases,int version,
 int oz,int oy,int ox,int dz,int dy,int dx){
"""
        + resolve
        + """
}
template<int Depth,bool Safe,int T>
__device__ __forceinline__ void StageRegisterLowAsync(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ResolveAsyncTrail<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy,0);
  if(!v.point){
   const float bad=__int_as_float(0x7fffffff);
   #pragma unroll
   for(int field=0;field<6;++field)stage[field*LowXPoints+i]=(v.bad&&(v.mask&(1u<<field)))?bad:0.f;
  }else{
   #pragma unroll
   for(int field=0;field<6;++field){
    if(v.mask&(1u<<field))__pipeline_memcpy_async(stage+field*LowXPoints+i,v.point+field*SurfacePoints,sizeof(float));
    else stage[field*LowXPoints+i]=0.f;
   }
  }
 }
 // Every participating thread commits, including threads with an empty group.
 __pipeline_commit();
}

"""
    )


def transform(consumer, cell):
    text = consumer.decode()
    if LOOP + CALL in text:
        original = text.replace(LOOP + CALL, LOOP, 1).replace(B1, OLD, 1)
        if early_transform(original.encode(), cell) != consumer:
            raise ValueError("Input prefetch source is not the reviewed transformation")
        early = text
    else:
        early = early_transform(consumer, cell).decode()
    code = helper(early)
    anchor = (
        "template<int T,int Slots>\n__device__ __forceinline__ void RetireRegisterLow("
    )
    if early.count(anchor) != 1 or "cuda_pipeline.h" in early:
        raise ValueError("Ambiguous async helper/header insertion")
    result = early.replace(
        "#include <algorithm>", "#include <cuda_pipeline.h>\n#include <algorithm>", 1
    )
    result = result.replace(anchor, code + anchor, 1)
    result = result.replace(LOOP + CALL, LOOP + ASYNC_CALL, 1)
    result = result.replace(B1, WAIT + B1, 1)
    inverse = result.replace(WAIT + B1, B1, 1)
    inverse = inverse.replace(LOOP + ASYNC_CALL, LOOP + CALL, 1)
    inverse = inverse.replace(code + anchor, anchor, 1)
    inverse = inverse.replace("#include <cuda_pipeline.h>\n", "", 1)
    if inverse != early:
        raise ValueError("Async lowering changes more than helper/copy/commit/wait")
    if result.count("__syncthreads()") != early.count("__syncthreads()"):
        raise ValueError("Async stage may not change phase-barrier count")
    return result.encode(), {
        "early_source_sha256": digest(early.encode()),
        "resolve_prefix_sha256": digest(resolve_body(early).encode()),
        "helper_sha256": digest(code.encode()),
        "inverse_to_early_exact": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        output.exists()
        or output == canonical
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        p.error("Output must be a new disjoint noncanonical source tree")
    consumer = source / "src/asymmetric_native.cuh"
    cell = source / "src/asymmetric_native_cell_generated.cuh"
    changed, binding = transform(consumer.read_bytes(), cell.read_bytes())
    proof = {
        "parent_consumer_sha256": digest(consumer.read_bytes()),
        "consumer_sha256": digest(changed),
        "cell_sha256": digest(cell.read_bytes()),
        "source_binding": binding,
        "stage_lifetime": prove_stage(),
        "async_contract": {
            "source": "immutable lower-rank completed trail faces",
            "copy_size_and_alignment": 4,
            "copies_per_active_thread": 6,
            "commit": "all threads after issuing their copies, before H work",
            "wait": "all threads before original B1; B1 publishes across warps",
            "invalid_fields": "synchronous positive zero; no invalid global address used",
            "error_fields": "same bad-bit2 and masked0x7fffffff payload as ImportPacket",
            "architecture": "SM80+; candidate build must targetSM86 and show actual LDGSTS",
        },
    }
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_bytes(changed)
    (output.parent / "async-stage-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in proof.items() if k != "stage_lifetime"}))


if __name__ == "__main__":
    main()
