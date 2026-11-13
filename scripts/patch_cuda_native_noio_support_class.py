"""Specialize the existing NoIO native-boundary-free trajectory certificate.

SupportSafe/FullWindow are separate from Fast: CPML remains active. The host
native-boundary bit already certifies every owned/startup/imported support and
the complete1..Depth window. Both old and specialized bodies keep4barriers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "f0657105ea5a57219d48a7a067d1ef53f6246676579e722ffd0670a14d3d8f49"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
ENV_OLD = "template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false> struct Environment {"
ENV_NEW = "template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false,bool ProvenSupport=false> struct Environment {"
BODY_OLD = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>\n__device__ __forceinline__ void UniformBody"
BODY_NEW = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,bool ProvenSupport=false>\n__device__ __forceinline__ void UniformBody"
SUPPORT = "SupportSafe=Fast||(Policy>=2&&Policy<=4);"
NEW_SUPPORT = "SupportSafe=Fast||(Policy>=2&&Policy<=4)||ProvenSupport;"
WRAPPER_OLD = """ // Host compact range contains only certified NoIO IDs.
 UniformBody<Depth,1,false,true,1,true>(p,tiles);"""
WRAPPER_NEW = """ const uint8_t flags=p.flags[tiles[blockIdx.x]];
 // Host compact range contains only certified NoIO IDs.
 // Native-boundary-free is an orthogonal support fact. Policy1 still hasCPML.
 if(!(flags&beamz::cuda::moving::kNativeBoundary))UniformBody<Depth,1,false,true,1,true,true>(p,tiles);
 else UniformBody<Depth,1,false,true,1,true,false>(p,tiles);"""


def sha(x):
    return hashlib.sha256(x).hexdigest()


def once(t, a, b):
    if t.count(a) != 1:
        raise ValueError(f"Expected one support-class anchor:{a!r}")
    return t.replace(a, b, 1)


def transform(text, cell):
    if sha(text.encode()) != PARENT or sha(cell) != CELL:
        raise ValueError("Exact frozen Obs9f6a source/canonical point required")
    original = text
    text = once(text, ENV_OLD, ENV_NEW)
    text = once(text, BODY_OLD, BODY_NEW)
    if text.count(SUPPORT) != 2:
        raise ValueError("Environment/body support facts changed")
    text = text.replace(SUPPORT, NEW_SUPPORT)
    start = text.index(BODY_NEW)
    end = text.index(
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__",
        start,
    )
    body = text[start:end]
    body = once(
        body,
        " constexpr bool SupportSafe=",
        " constexpr bool FullWindow=Fast||ProvenSupport;\n constexpr bool SupportSafe=",
    )
    body = once(
        body,
        "Environment<Layout,Fast,IsotropicH,Policy,UniformE>env",
        "Environment<Layout,Fast,IsotropicH,Policy,UniformE,ProvenSupport>env",
    )
    assert (
        body.count("if constexpr(!Fast)") == 2 and body.count("if constexpr(Fast)") == 1
    )
    body = body.replace("if constexpr(!Fast)", "if constexpr(!FullWindow)").replace(
        "if constexpr(Fast)", "if constexpr(FullWindow)"
    )
    body = once(
        body,
        "for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n)",
        "for(int n=(FullWindow?1:window[0]);n<=(FullWindow?Depth:window[1]);++n)",
    )
    text = text[:start] + body + text[end:]
    text = once(text, WRAPPER_OLD, WRAPPER_NEW)
    # Exact source inverse binds every old phase/barrier/import/FP expression.
    inverse = (
        text.replace(WRAPPER_NEW, WRAPPER_OLD, 1)
        .replace(NEW_SUPPORT, SUPPORT)
        .replace(ENV_NEW, ENV_OLD, 1)
        .replace(BODY_NEW, BODY_OLD, 1)
    )
    inverse = inverse.replace(
        " constexpr bool FullWindow=Fast||ProvenSupport;\n", "", 1
    ).replace(
        "Environment<Layout,Fast,IsotropicH,Policy,UniformE,ProvenSupport>env",
        "Environment<Layout,Fast,IsotropicH,Policy,UniformE>env",
        1,
    )
    inverse = (
        inverse.replace("if constexpr(!FullWindow)", "if constexpr(!Fast)")
        .replace("if constexpr(FullWindow)", "if constexpr(Fast)")
        .replace("FullWindow?", "Fast?")
    )
    assert inverse == original, "Default source cannot be reconstructed bytewise"
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
        or out == canonical
        or src == out
        or src in out.parents
        or out in src.parents
    ):
        p.error("New disjoint private copied source required")
    cell = (src / "src/asymmetric_native_cell_generated.cuh").read_bytes()
    changed = transform((src / "src/asymmetric_native.cuh").read_text(), cell)
    planner = (src / "src/native_moving.cc").read_text()
    anchors = (
        "if(low[a]<=0||high[a]+1>=minimum[a])flags|=kNativeBoundary;",
        "low[a]=origin[a]-depth;high[a]=origin[a]+p.owned[a]-2;",
    )
    if any(x not in planner for x in anchors):
        raise ValueError("Actual planner certificate inequalities changed")
    shutil.copytree(src, out)
    (out / "src/asymmetric_native.cuh").write_text(changed)
    proof = {
        "parent_consumer_sha256": PARENT,
        "consumer_sha256": sha(changed.encode()),
        "canonical_cell_sha256": CELL,
        "planner_sha256": sha(planner.encode()),
        "exact_source_inverse": True,
        "HasCpml_unchanged": True,
        "barriers_per_step": 4,
        "host_sort_and_launches_unchanged": True,
        "fact": "!(flags&32) => origin-depth>0 and origin+B-1<minimum on allaxes",
        "scope": "Resource/correctness candidate only; NoIO retains physical CPML histories and signed derivatives",
    }
    (out.parent / "support-class-source-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
