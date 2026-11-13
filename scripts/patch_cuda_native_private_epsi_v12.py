"""Copy-only plain-register Epsi placement in the actual canonical V12 context.

This resource experiment preserves scalar cell arithmetic and all admission,
reload/flush, axes and field schedules from the canonical shared-Epsi candidate.
The extra8 state floats are plain constant-index arrays; no volatile private
references or dynamic array index may force local memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENTS = {
    "asymmetric_native.cuh": "33a5928617f589bf5ff4f68ba8f4ec179108cfb9fada85f16eb9e519850cb9d1",
    "native_extension.cc": "2d16eec337f62454b2d6ec076e903592dc656469541e3ef4ed6927da12172162",
    "native_moving.h": "082576ce0fd5774a0ae236002bfe35d799fc65429c0ded60db8d676721791e7d",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
    "runtime_shared_epsi_state_fact.h": "cbfb91ca46f3ef8c72885f34c7e23b7a24cad67064fbcbaf2e946cfb01840d66",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one private-register anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, digest in PARENTS.items():
        if sha(parent[name]) != digest:
            raise ValueError(f"Exact canonical shared-Epsi V2 required: {name}")
    c = parent["asymmetric_native.cuh"].decode()
    e = parent["native_extension.cc"].decode()
    h = parent["native_moving.h"].decode()
    edits = [
        (
            "template<class Base> struct SharedEPsiPoint",
            "template<bool Enabled,int Slots> struct PrivateEPsiBank {};\ntemplate<int Slots> struct PrivateEPsiBank<true,Slots>{float a[Slots],b[Slots];};\ntemplate<class Base> struct SharedEPsiPoint",
        ),
        ("volatile float &a,&b;", "float &a,&b;"),
        (
            "Base&base,volatile float&pa,volatile float&pb,bool fresh",
            "Base&base,float&pa,float&pb,bool fresh",
        ),
        (
            "__device__ __forceinline__ volatile float& State(int slot)",
            "__device__ __forceinline__ float& State(int slot)",
        ),
        (
            "SharedEPsiBytes=SharedBytes+8ULL*1024*sizeof(float)",
            "SharedEPsiBytes=SharedBytes",
        ),
        (
            'SharedEPsiBytes==84516,"Frozen shared-bank allocation"',
            'SharedEPsiBytes==51748,"Frozen private-register allocation"',
        ),
        (
            " volatile float*epsi_bank=nullptr;\n if constexpr(CacheE)epsi_bank=frame+SharedBytes/sizeof(float);",
            " PrivateEPsiBank<CacheE,Slots> epsi;",
        ),
        (
            "epsi_bank[j*1024+threadIdx.x],epsi_bank[(Slots+j)*1024+threadIdx.x]",
            "epsi.a[j],epsi.b[j]",
        ),
    ]
    out = c
    for old, new in edits:
        out = once(out, old, new)
    inverse = out
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    if inverse != c:
        raise ValueError("Exact device-source inverse failed")
    # Names are host/kernel identity only; all existing entries remain unchanged.
    out = out.replace("AsymmetricNoIOSharedEPsi1024", "AsymmetricNoIOPrivateEPsi1024")
    out = out.replace(
        "BeamzResearchSharedEPsiEnabled", "BeamzResearchPrivateEPsiEnabled"
    )
    host_edits = [
        (
            'result["moving_shared_epsi_bytes"]=84516;',
            'result["moving_shared_epsi_bytes"]=51748;result["moving_psi_cache_storage"]="private-registers";result["moving_private_epsi_floats_per_thread"]=8;',
        ),
        (
            'module.attr("moving_shared_epsi_shared_bytes")=84516;',
            'module.attr("moving_shared_epsi_shared_bytes")=51748;module.attr("moving_private_epsi")=true;module.attr("moving_psi_cache_storage")="private-registers";',
        ),
    ]
    host = e
    for old, new in host_edits:
        host = once(host, old, new)
    check = host
    for old, new in reversed(host_edits):
        check = once(check, new, old)
    if check != e:
        raise ValueError("Exact host inverse failed")
    host = host.replace(
        "BeamzResearchSharedEPsiEnabled", "BeamzResearchPrivateEPsiEnabled"
    )
    h = h.replace("BeamzResearchSharedEPsiEnabled", "BeamzResearchPrivateEPsiEnabled")
    files = dict(parent)
    files["asymmetric_native.cuh"] = out.encode()
    files["native_extension.cc"] = host.encode()
    files["native_moving.h"] = h.encode()
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parent, out = a.parent.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or out in (parent, canonical)
        or parent in out.parents
        or out in parent.parents
        or canonical in out.parents
    ):
        p.error("New disjoint private output required")
    source = {name: (parent / "src" / name).read_bytes() for name in PARENTS}
    files = transformed(source)
    shutil.copytree(parent, out)
    for name, data in files.items():
        (out / "src" / name).write_bytes(data)
    proof = {
        "parent_hashes": PARENTS,
        "result_hashes": {n: sha(b) for n, b in files.items()},
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "same_canonical_arithmetic": True,
        "same_admission_first_flush_axes_and_field_schedule": True,
        "source5_body_unchanged": True,
        "private_state": "two plain arrays[Slots4], constant j and field color",
        "shared_bytes": 51748,
        "expected_kernels": 60,
        "compile_context": "actual complete nonRDC moving TU, RDCordinarycore/runtime",
        "scope": "Resource gate first: reject any STACK/LOCAL/PTXASspill; no GPU pass implied",
    }
    (out.parent / "private-epsi-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
