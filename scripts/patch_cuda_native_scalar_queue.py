"""Emit an opt-in native consumer with a compact fixed-register Z-slot loop.

The input consumer is never overwritten. This changes storage routing only;
physical shared addresses and the three-barrier temporal schedule are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one queue patch anchor: {old!r}")
    return source.replace(old, new, 1)


def compact_phase(source, start, end, phase):
    lo = source.index(start)
    hi = source.index(end, lo)
    block = source[lo:hi]
    block = once(
        block,
        "#pragma unroll\n   for(int j=0;j<CZ;++j)",
        "#pragma unroll 1\n   for(int j=0;j<CZ;++j)",
    )
    if phase == 0:
        block = block.replace("ey[(j+1)%CZ]", "ey[1]")
    else:
        block = block.replace("hy[(j+CZ-1)%CZ]", "hy[CZ-1]")
    block = block.replace("hy[j]", "hy[0]").replace("ey[j]", "ey[0]")
    tail = "   }\n  }\n  "
    if not block.endswith(tail):
        raise ValueError("Unexpected Z-slot loop end")
    block = block[: -len(tail)] + (
        "    // Every lane rotates, including excluded head/ghost/invalid slots.\n"
        "    RotatePrivateQueue(hy);RotatePrivateQueue(ey);\n" + tail
    )
    return source[:lo] + block + source[hi:]


def patch(source):
    source = once(
        source,
        "template<int Depth>\n__device__ __forceinline__ float Import",
        """// Queue[k] is physical slot (j+k)%CZ at iteration j. CZ rotations
// restore the physical slot order before imports/publication.
__device__ __forceinline__ void RotatePrivateQueue(float(&values)[CZ]){
 const float first=values[0];
 #pragma unroll
 for(int k=0;k<CZ-1;++k)values[k]=values[k+1];
 values[CZ-1]=first;
}
template<int Depth>
__device__ __forceinline__ float Import""",
    )
    source = compact_phase(
        source,
        "  {const int step=*phase_step,",
        "__syncthreads(); // B1:",
        0,
    )
    return compact_phase(
        source,
        "  {const int step=*phase_step-1,",
        "__syncthreads(); // B3:",
        1,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=ROOT / "scripts/cuda_asymmetric_native.cuh"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve():
        p.error("Input is frozen; output must be a different path")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(patch(a.input.read_text()))
    print(
        json.dumps(
            {
                "input": str(a.input),
                "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
                "output": str(a.output),
                "output_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
                "scope": "Storage routing candidate; GPU arithmetic/race gates required",
            }
        )
    )


if __name__ == "__main__":
    main()
