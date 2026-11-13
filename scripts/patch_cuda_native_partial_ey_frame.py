"""Private larger-tile mapping with partial shared Ey and last-use scratch.

Six of each eight Ey planes live in shared memory; two stay private. H stays
private, Hx/Hz and Ex/Ez reuse two frames, boundary snapshots retire per-cell
into pending E. The canonical point operator is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "3ed4a4f837df1ca60f409a243c0dd8a63778c0b2b983bac0ee895b56da04d6aa"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"


def sha(x):
    return hashlib.sha256(x).hexdigest()


def once(t, a, b):
    if t.count(a) != 1:
        raise ValueError(f"Expected one partial-Ey anchor:{a!r}")
    return t.replace(a, b, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", choices=("16x16x32", "8x32x32"), required=True)
    a = p.parse_args()
    src = a.source.resolve()
    out = a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or out == canonical
        or src == out
        or src in out.parents
        or out in src.parents
    ):
        p.error("New disjoint private copied source required")
    c = (src / "src/asymmetric_native.cuh").read_text()
    cell = (src / "src/asymmetric_native_cell_generated.cuh").read_bytes()
    if sha(c.encode()) != PARENT or sha(cell) != CELL:
        raise ValueError("Exact frozen4941 consumer/canonical point required")
    cz, cy = (16, 16) if a.cache == "16x16x32" else (8, 32)
    bz, by = cz - 1, cy - 1
    body = Path(__file__).with_name("cuda_native_partial_ey_frame.cuh").read_text()
    start = c.index(
        "template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageRegisterLow"
    )
    end = c.index(
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__",
        start,
    )
    c = c[:start] + body + "\n" + c[end:]
    c = once(c, "BZ=7,BY=15,BX=31,CZ=8,CY=16", f"BZ={bz},BY={by},BX=31,CZ={cz},CY={cy}")
    c = once(
        c,
        "3ULL*PAD*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+6ULL*LowXPoints*sizeof(float);",
        "(2ULL*PAD+3*PAD/4+2048)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+3ULL*LowXPoints*sizeof(float);",
    )
    c = once(c, "z=WrapZ(z);y&=15;x&=31;", "z=WrapZ(z);y&=(CY-1);x&=31;")
    shutil.copytree(src, out)
    (out / "src/asymmetric_native.cuh").write_text(c)
    h = out / "src/native_moving.h"
    t = h.read_text()
    for old, new in [
        ("kOwnedZ=7,kOwnedY=15", f"kOwnedZ={bz},kOwnedY={by}"),
        ("counts{},owned{7,15,31}", f"counts{{}},owned{{{bz},{by},31}}"),
        ("int owned[3]{7,15,31};", f"int owned[3]{{{bz},{by},31}};"),
    ]:
        t = once(t, old, new)
    h.write_text(t)
    shared = (2 * 8192 + 6144 + 2048) * 4 + 76 + 3 * bz * by * 4
    e = out / "src/native_extension.cc"
    t = e.read_text()
    t = once(
        t,
        "NB_MODULE(_cuda_native, module) {",
        f'NB_MODULE(_cuda_native, module) {{\n  module.attr("partial_ey_cache")=nb::make_tuple({cz},{cy},32);\n  module.attr("partial_ey_shared_bytes")={shared};',
    )
    e.write_text(t)
    j = {
        "parent_consumer_sha256": PARENT,
        "consumer_sha256": sha(c.encode()),
        "canonical_point_sha256": CELL,
        "body_sha256": sha(body.encode()),
        "cache_zyx": [cz, cy, 32],
        "owned_zyx": [bz, by, 31],
        "shared_bytes": shared,
        "persistent_field_floats_per1024thread": 26,
        "current_ExEz_floats": 2,
        "barriers_per1024step": 11,
        "pending_storage": "same2048 shared cells as retired H boundary snapshots; unique same-XY consumer",
        "scope": "Experimental storage schedule, requires independent token/collective/resource/numerical/race gates",
    }
    (out.parent / "partial-ey-source-proof.json").write_text(
        json.dumps(j, indent=2) + "\n"
    )
    print(json.dumps(j))


if __name__ == "__main__":
    main()
