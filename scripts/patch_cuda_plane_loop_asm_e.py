"""Copy-only E-loop selector using literal-register PTX bit selections.

The C++ one-hot selector was reconstructed as a local-memory gather by nvcc.
This experiment exposes every H slot as an explicit asm register operand. It
preserves the admitted 512-thread eighteen-barrier schedule and cell arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_plane_loop_e import patch as loop_e


def patch(source):
    if "BZ=15,BY=15,BX=31,CZ=16,CY=16,CX=32,Threads=512" not in source:
        raise ValueError("The literal sixteen-plane/512-thread geometry is required")
    source = loop_e(source)
    selector = """   #pragma unroll
   for(int k=0;k<Slots;++k)if(j==k){hp[0]=h[0][k];hp[1]=h[1][k];hp[2]=h[2][k];}"""
    if source.count(selector) != 1:
        raise ValueError("Expected exactly one C++ one-hot H selector")
    # Read every input before writing outputs. This avoids relying on an
    # early-clobber constraint unsupported by some CUDA inline-asm versions.
    blocks = []
    for k in range(16):
        blocks.append(
            f"""   asm volatile("{{ .reg .pred selected; .reg .f32 ax,ay,az; "
    "mov.f32 ax,%3; mov.f32 ay,%4; mov.f32 az,%5; "
    "setp.eq.s32 selected,%6,{k}; "
    "selp.f32 %0,ax,%0,selected; selp.f32 %1,ay,%1,selected; selp.f32 %2,az,%2,selected; }}"
    : "+f"(hp[0]),"+f"(hp[1]),"+f"(hp[2])
    : "f"(h[0][{k}]),"f"(h[1][{k}]),"f"(h[2][{k}]),"r"(j));"""
        )
    source = source.replace(selector, "\n".join(blocks), 1)
    return source.replace("AsymmetricRetireLoopE", "AsymmetricRetireLoopAsmE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("A private copied source is required")
    path = args.source / "src/asymmetric_native.cuh"
    parent = path.read_text()
    result = patch(parent)
    path.write_text(result)
    print(
        json.dumps(
            {
                "parent_sha256": hashlib.sha256(parent.encode()).hexdigest(),
                "consumer_sha256": hashlib.sha256(result.encode()).hexdigest(),
                "scope": "Literal register selection experiment; resource/physics gates required",
            }
        )
    )


if __name__ == "__main__":
    main()
