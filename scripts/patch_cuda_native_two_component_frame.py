"""Private larger15x15x31 tile with H/Ey scalars and two shared components.

This is a storage/schedule experiment; the traced point equations remain exact.
The2Zchunks exchange only boundary snapshots; XY uses converged warp shuffles
or a shared current plane. Full original bounds, import and owner checks remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "3ed4a4f837df1ca60f409a243c0dd8a63778c0b2b983bac0ee895b56da04d6aa"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one anchor:{old!r}")
    return text.replace(old, new, 1)


def transform(text, body):
    if sha(text.encode()) != PARENT:
        raise ValueError("Expected exact4941 consumer")
    start = text.index(
        "template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageRegisterLow"
    )
    end = text.index(
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__",
        start,
    )
    text = text[:start] + body + "\n" + text[end:]
    text = once(text, "BZ=7,BY=15,BX=31,CZ=8", "BZ=15,BY=15,BX=31,CZ=16")
    text = once(
        text,
        "3ULL*PAD*sizeof(float)+8*sizeof(size_t)",
        "(2ULL*PAD+3072)*sizeof(float)+8*sizeof(size_t)",
    )
    # Existing Slot/WrapZ work for CZ16; all explicit ring masks in the newbody
    # are16. The old unused At helper is excluded to avoid implying3sharedfields.
    old = "return shared[(2*(field/3)+(field%3)/2)*PAD+Slot(z,y,x)];"
    text = once(text, old, "return shared[((field%3)==0?0:1)*PAD+Slot(z,y,x)];")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src = a.source.resolve()
    out = a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or src == out
        or out == canonical
        or src in out.parents
        or out in src.parents
    ):
        p.error("New disjoint private copied source required")
    c = src / "src/asymmetric_native.cuh"
    cell = src / "src/asymmetric_native_cell_generated.cuh"
    body = Path(__file__).with_name("cuda_native_two_component_frame.cuh").read_text()
    if sha(cell.read_bytes()) != CELL:
        raise ValueError("Canonical traced point operator changed")
    changed = transform(c.read_text(), body)
    shutil.copytree(src, out)
    (out / "src/asymmetric_native.cuh").write_text(changed)
    h = out / "src/native_moving.h"
    t = h.read_text()
    for old, new in [
        ("kOwnedZ=7,kOwnedY=15", "kOwnedZ=15,kOwnedY=15"),
        ("counts{},owned{7,15,31}", "counts{},owned{15,15,31}"),
        ("int owned[3]{7,15,31};", "int owned[3]{15,15,31};"),
    ]:
        t = once(t, old, new)
    h.write_text(t)
    e = out / "src/native_extension.cc"
    t = e.read_text()
    t = once(
        t,
        "NB_MODULE(_cuda_native, module) {",
        'NB_MODULE(_cuda_native, module) {\n  module.attr("two_component_cached_zyx")=nb::make_tuple(16,16,32);\n  module.attr("two_component_shared_bytes")=83300;',
    )
    e.write_text(t)
    proof = {
        "parent_consumer_sha256": sha(c.read_bytes()),
        "consumer_sha256": sha(changed.encode()),
        "body_sha256": sha(body.encode()),
        "canonical_cell_sha256": CELL,
        "owned_zyx": [15, 15, 31],
        "cached_zyx": [16, 16, 32],
        "shared_bytes": 83300,
        "field_scalars_1024": 34,
        "cross_snapshot_scalars": 3072,
        "barriers_1024": 19,
        "barriers_512": 35,
        "scope": "Experimental mapping/resource candidate; full numerical, token, race gates required before any promotion",
    }
    (out.parent / "two-component-source-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
