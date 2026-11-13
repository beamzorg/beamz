"""Copy the resource-clean v7 TU context with small all-shared Ey storage.

No numerical primitive, ABI field layout, source/DFT clock or CPML addressing
change. The new body keeps H private, Ey shared, Ex/Ez pending private and direct
Hx/Hy Z snapshots in a separate shared area. Generic kernels use512 threads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "12486a02201fcd072e1f1aabd5e66b20e1e8049176f2043afa878f9cd5ace12d"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
BODY = "83c46665e731760ddd6c6826d70d855dc9d330b8ee8056506d1c8c9a350d8312"


def sha(value):
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Reviewed source anchor changed: {old!r}")
    return source.replace(old, new, 1)


def transform(source, body):
    if sha(source) != PARENT or sha(body) != BODY:
        raise ValueError("Exact reviewed v7 consumer and shared-Ey body required")
    start = source.index(
        "template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StagePartialH"
    )
    end = source.index(
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__",
        start,
    )
    old_region = source[start:end]
    changed = source[:start] + body + "\n" + source[end:]
    edits = [
        ("BZ=7,BY=31,BX=31,CZ=8,CY=32", "BZ=7,BY=15,BX=31,CZ=8,CY=16"),
        (
            "(2ULL*PAD+3*PAD/4+2048)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+3ULL*LowXPoints*sizeof(float);",
            "(3ULL*PAD+2048)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+3ULL*LowXPoints*sizeof(float);",
        ),
        (
            "__launch_bounds__((((Policy==3||Policy==4)||(!Fast&&!IsotropicH))?256:512),1)",
            "__launch_bounds__(512,1)",
        ),
        (
            "<<<count,(((Policy==3||Policy==4)||(!Fast&&!IsotropicH))?256:512),SharedBytes,stream>>>",
            "<<<count,512,SharedBytes,stream>>>",
        ),
    ]
    for old, new in edits:
        changed = once(changed, old, new)
    inverse = changed
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    inverse = once(inverse, body + "\n", old_region)
    if inverse != source:
        raise ValueError("Unchanged original consumer cannot be recovered")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
        parser.error("New disjoint private source tree required")
    if sha((src / "src/asymmetric_native_cell_generated.cuh").read_bytes()) != CELL:
        raise ValueError("Canonical point operator changed")
    body = (
        Path(__file__)
        .with_name("cuda_native_shared_ey_private_pending.cuh")
        .read_text()
    )
    changed = transform((src / "src/asymmetric_native.cuh").read_text(), body)
    shutil.copytree(src, out)
    (out / "src/asymmetric_native.cuh").write_text(changed)
    h = out / "src/native_moving.h"
    text = h.read_text()
    for old, new in (
        ("kOwnedZ=7,kOwnedY=31", "kOwnedZ=7,kOwnedY=15"),
        ("counts{},owned{7,31,31}", "counts{},owned{7,15,31}"),
        ("int owned[3]{7,31,31};", "int owned[3]{7,15,31};"),
    ):
        text = once(text, old, new)
    h.write_text(text)
    e = out / "src/native_extension.cc"
    text = once(
        e.read_text(),
        'module.attr("partial_ey_cache")=nb::make_tuple(8,32,32);',
        'module.attr("partial_ey_cache")=nb::make_tuple(8,16,32);\n  module.attr("shared_ey_private_pending")=true;',
    )
    text = once(
        text,
        'module.attr("partial_ey_shared_bytes")=100984;',
        'module.attr("partial_ey_shared_bytes")=58680;',
    )
    e.write_text(text)
    proof = {
        "parent_consumer_sha256": PARENT,
        "consumer_sha256": sha(changed),
        "canonical_cell_sha256": CELL,
        "body_sha256": BODY,
        "source_inverse": True,
        "cache_zyx": [8, 16, 32],
        "owned_zyx": [7, 15, 31],
        "shared_bytes": 58680,
        "threads": {"uniform": 1024, "generic": 512},
        "source_field_floats": {"uniform1024": 16, "generic512": 28},
        "barriers_per_step": {"uniform1024": 7, "generic512": 11},
        "snapshot_contract": "Hx/Hy producer-indexed2x1024 shared scalars, unique opposite-Z-chunk same-XY consumer at j0, P0 publication",
        "pending_contract": "Ex/Ez2floats perthread/Ygroup; store after own E, restore previous after Pj, final after RL",
        "build_context": "Inherited moving-only nonRDC v7; ordinary core/runtime RDC and actual Ninja definitions must be rechecked",
        "scope": "Independent CPU tag proof only; resources, numerical and sanitizer gates pending",
    }
    (out.parent / "shared-ey-source-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
