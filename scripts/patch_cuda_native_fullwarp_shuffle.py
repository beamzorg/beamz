"""Lower only two proven complete-warp exchanges to explicit PTX under RDC.

The CUDA SDK intrinsic adds a potentially divergent slow-path device call under
RDC, whose ABI spills live field state. Both sites are independently proved to
have all32 lanes active, initialized and executing the same width32 exchange.
No arithmetic, lane mapping, masks, memory ordering or CTA barrier changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "12486a02201fcd072e1f1aabd5e66b20e1e8049176f2043afa878f9cd5ace12d"
HELPER = r"""// Requires the independently proved full-warp participation contract.
__device__ __forceinline__ float FullWarpIndex32(float value,int lane){
 const uint32_t word=__float_as_uint(value);uint32_t result;
 asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
              : "=r"(result) : "r"(word), "r"(lane));
 return __uint_as_float(result);
}
"""
OLD_EY = "__shfl_sync(0xffffffffu,get_ey(zs,ys,sx,si),(sx+1)&31)"
NEW_EY = "FullWarpIndex32(get_ey(zs,ys,sx,si),(sx+1)&31)"
OLD_HY = "__shfl_sync(0xffffffffu,h[1][si],(sx+31)&31)"
NEW_HY = "FullWarpIndex32(h[1][si],(sx+31)&31)"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transform(source):
    if digest(source.encode()) != PARENT:
        raise ValueError("Exact frozen partial-Ey bulk-v3 consumer required")
    if source.count("__shfl_sync(") != 2:
        raise ValueError("Unreviewed warp collective sites")
    for old in (OLD_EY, OLD_HY):
        if source.count(old) != 1:
            raise ValueError("Reviewed warp-lane expression changed")
    anchor = "template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StagePartialH"
    if source.count(anchor) != 1:
        raise ValueError("Partial frame source changed")
    changed = source.replace(anchor, HELPER + anchor, 1)
    changed = changed.replace(OLD_EY, NEW_EY, 1).replace(OLD_HY, NEW_HY, 1)
    inverse = (
        changed.replace(HELPER, "", 1)
        .replace(NEW_EY, OLD_EY, 1)
        .replace(NEW_HY, OLD_HY, 1)
    )
    if inverse != source:
        raise ValueError("Unchanged schedule/FP source cannot be reconstructed")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src, out = args.source.resolve(), args.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or out == canonical
        or src == out
        or src in out.parents
        or out in src.parents
    ):
        parser.error("New disjoint private source required")
    consumer = src / "src/asymmetric_native.cuh"
    changed = transform(consumer.read_text())
    shutil.copytree(src, out)
    (out / "src/asymmetric_native.cuh").write_text(changed)
    proof = {
        "parent_consumer_sha256": PARENT,
        "consumer_sha256": digest(changed.encode()),
        "helper_sha256": digest(HELPER.encode()),
        "exact_source_inverse": True,
        "RDC_retained": True,
        "contract": "All32 lanes, initialized inputs, lane0..31, full0xffffffff membermask, width32/clamp0x1f",
        "sites": [
            "privateEy to Hz(+X), before owned/Valid",
            "currentHy to Ez(-X), before owned/Valid",
        ],
        "PTX_source": "https://docs.nvidia.com/cuda/archive/13.2.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync",
        "scope": "Compiler lowering only; exact-context resource and bitpattern/numerical/synchronization gates still required",
    }
    (out.parent / "fullwarp-source-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
