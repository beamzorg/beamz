"""Copy-only four-register Epsi color-A cache in canonical four-barrier NoIO.

Terms0/1/3 may cache under the existing single-axis fact; terms2/4/5 and all H
recurrences retain immediate ordinary global loads/stores. No B state exists.
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
from scripts.patch_cuda_native_paired_private_epsi import PARENTS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one colorA anchor:{old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, d in PARENTS.items():
        if sha(parent[name]) != d:
            raise ValueError(
                f"Exact original canonical private8 parent required:{name}"
            )
    old = parent["asymmetric_native.cuh"].decode()
    c = old
    edits = [
        (
            "PrivateEPsiBank<true,Slots>{float a[Slots],b[Slots];}",
            "PrivateEPsiBank<true,Slots>{float a[Slots];}",
        ),
        (
            " float &a,&b;bool first,flush,enabled;",
            " float &a;bool first,flush,enabled;",
        ),
        ("Base&base,float&pa,float&pb,bool fresh", "Base&base,float&pa,bool fresh"),
        (":Base(base),a(pa),b(pb),first(fresh)", ":Base(base),a(pa),first(fresh)"),
        (
            "  const int term=slot%6;return term==2||term==4||term==5?b:a;",
            "  return a;",
        ),
        (
            "  if(slot<6||!enabled||first)return Base::LoadLocal(slot,z,y,x);",
            "  const int term=slot%6;\n  if(slot<6||!enabled||first||term==2||term==4||term==5)return Base::LoadLocal(slot,z,y,x);",
        ),
        (
            "  if(slot<6||!enabled){Base::StoreLocal(slot,z,y,x,value);return;}",
            "  const int term=slot%6;\n  if(slot<6||!enabled||term==2||term==4||term==5){Base::StoreLocal(slot,z,y,x,value);return;}",
        ),
        ("env,epsi.a[j],epsi.b[j],", "env,epsi.a[j],"),
    ]
    for a, b in edits:
        c = once(c, a, b)
    inv = c
    for a, b in reversed(edits):
        inv = once(inv, b, a)
    assert inv == old
    c = c.replace(
        "AsymmetricNoIOPrivateEPsi1024", "AsymmetricNoIOColorAPrivateEPsi1024"
    )
    e = parent["native_extension.cc"].decode()
    host_edits = [
        (
            'result["moving_private_epsi_floats_per_thread"]=8;',
            'result["moving_private_epsi_floats_per_thread"]=4;result["moving_private_epsi_cached_e_terms"]=nb::make_tuple(0,1,3);',
        ),
        (
            "for(int c=0;c<3;++c){if(axes[2*c]!=axis&&axes[2*c+1]!=axis)continue;const auto&f=base_inputs_[3+c];",
            "const int cached_term=axis==0?1:axis==1?0:3;\n          for(int c=0;c<3;++c){if(c!=cached_term/2)continue;const auto&f=base_inputs_[3+c];",
        ),
        ("const int term=axes[2*c]==axis?2*c:2*c+1;", "const int term=cached_term;"),
        (
            'module.attr("moving_private_epsi")=true;',
            'module.attr("moving_private_epsi")=true;module.attr("moving_private_epsi_colors")=1;module.attr("moving_private_epsi_cached_e_terms")=nb::make_tuple(0,1,3);',
        ),
    ]
    for a, b in host_edits:
        e = once(e, a, b)
    h = parent["native_extension.cc"].decode()
    inv = e
    for a, b in reversed(host_edits):
        inv = once(inv, b, a)
    assert inv == h
    out = dict(parent)
    out["asymmetric_native.cuh"] = c.encode()
    out["native_extension.cc"] = e.encode()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src, out = a.parent.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or out in (src, canonical)
        or src in out.parents
        or out in src.parents
        or canonical in out.parents
    ):
        p.error("New disjoint private output required")
    files = transformed({n: (src / "src" / n).read_bytes() for n in PARENTS})
    shutil.copytree(src, out)
    for n, b in files.items():
        (out / "src" / n).write_bytes(b)
    proof = {
        "parents": PARENTS,
        "result_hashes": {n: sha(b) for n, b in files.items()},
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "private_state": "one plain float[Slots4], no B allocation",
        "cached_e_terms": [0, 1, 3],
        "always_global_e_terms": [2, 4, 5],
        "host_witness_actual_A_term_zyx": [1, 0, 3],
        "same_reload_flush_onehot_disjoint_and_selector": True,
        "original_canonical_field4barriers_and_source5B_unchanged": True,
        "shared_bytes": 51748,
        "expected_kernels": 60,
        "gpu_executed": False,
        "gate": "Reject any stack/local/spill before GPU",
    }
    (out.parent / "private-colorA-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
