"""Roll only the E-plane consumer while retaining scalarized private H slots.

The 512-thread ownership, canonical cell, field lifetimes and eighteen barriers
are inherited from the admitted last-use18 parent. This tests instruction-size
versus register-selection cost; compilation and full physics gates are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def patch(source):
    if "void AsymmetricRetirePlane(" not in source:
        raise ValueError("An admitted eighteen-barrier plane parent is required")
    start = source.index(
        "  #pragma unroll\n  for(int j=0;j<Slots;++j){float current[3];"
    )
    end = source.index("  StagePlaneLow<Depth,SupportSafe,1>", start)
    body = source[start:end]
    if "if(j==CZ-1)last[c]=current[c]" not in body:
        raise ValueError("Both final E planes must remain deferred until R3")
    body = body.replace(
        "  #pragma unroll\n  for(int j=0;j<Slots;++j){float current[3];",
        """  #pragma unroll 1
  for(int j=0;j<Slots;++j){float current[3],hp[3]={0.f,0.f,0.f};
   // Every H array index becomes a compile-time constant. The runtime j is
   // CTA-uniform and selects register values, never a local-memory address.
   #pragma unroll
   for(int k=0;k<Slots;++k)if(j==k){hp[0]=h[0][k];hp[1]=h[1][k];hp[2]=h[2][k];}""",
        1,
    )
    body = body.replace("h[c][j]", "hp[c]").replace("h[field][j]", "hp[field]")
    if "h[c][j]" in body or "h[field][j]" in body:
        raise ValueError("Dynamic private H indexing is not admitted")
    return (source[:start] + body + source[end:]).replace(
        "AsymmetricRetirePlane", "AsymmetricRetireLoopE"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("A private copied source is required")
    path = args.source / "src/asymmetric_native.cuh"
    source = path.read_text()
    path.write_text(patch(source))
    print(
        json.dumps(
            {
                "parent_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "consumer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scope": "512-thread E-loop code-size experiment; not validated",
            }
        )
    )


if __name__ == "__main__":
    main()
