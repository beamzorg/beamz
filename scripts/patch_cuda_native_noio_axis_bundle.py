"""Copy-only CTA-uniform axis dispatch, preserving one full NoIO launch/grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_HASHES = {
    "asymmetric_native.cuh": "f0657105ea5a57219d48a7a067d1ef53f6246676579e722ffd0670a14d3d8f49",
    "native_extension.cc": "8f122cdbb850374f44c8e59c3282274b744ebff0ef5a907540a245a53a730bcf",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact bundled-axis anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if digest(parent[name]) != expected:
            raise ValueError(f"Exact9f6a parent required: {name}")
    c = parent["asymmetric_native.cuh"].decode()
    edits = (
        (
            "bool UniformE=false> struct Environment",
            "bool UniformE=false,int ActiveAxes=-1> struct Environment",
        ),
        (
            "  if(!(flags&(2u<<axis)))return false;",
            "  if constexpr(ActiveAxes>=0){if(!(ActiveAxes&(1u<<axis)))return false;}\n  else if(!(flags&(2u<<axis)))return false;",
        ),
        (
            "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>\n__device__",
            "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__",
        ),
        (
            " Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};",
            " Environment<Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes>env{p,flags};",
        ),
        (
            " // Host compact range contains only certified NoIO IDs.\n UniformBody<Depth,1,false,true,1,true>(p,tiles);",
            """ // Immutable per-tile selector is uniform across ALL threads before any
 // owner write or barrier. One original full-grid launch retains SM packing.
 switch(p.flags[tiles[blockIdx.x]]&14){
  case 2:UniformBody<Depth,1,false,true,1,true,1>(p,tiles);return;
  case 4:UniformBody<Depth,1,false,true,1,true,2>(p,tiles);return;
  case 8:UniformBody<Depth,1,false,true,1,true,4>(p,tiles);return;
  default:UniformBody<Depth,1,false,true,1,true>(p,tiles);return;
 }""",
        ),
    )
    for old, new in edits:
        c = once(c, old, new)
    inverse = c
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    if inverse.encode() != parent["asymmetric_native.cuh"]:
        raise ValueError("Exact inverse source binding failed")
    e = once(
        parent["native_extension.cc"].decode(),
        '  module.attr("moving_observer_uniform_1024")=true;',
        '  module.attr("moving_noio_axis_bundle")=true;\n  module.attr("moving_observer_uniform_1024")=true;',
    )
    return {
        "asymmetric_native.cuh": c.encode(),
        "native_extension.cc": e.encode(),
        "asymmetric_native_cell_generated.cuh": parent[
            "asymmetric_native_cell_generated.cuh"
        ],
    }


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        output.exists()
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be a new disjoint research source")
    if output == canonical or canonical in output.parents:
        raise ValueError("Canonical source must remain unchanged")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {
        "parent_module_sha256": "9f6ade0cfb908fb1a71c883bfd2b106321a0401a2b8ec1bc87472047a7cb22ec",
        "parent_hashes": PARENT_HASHES,
        "result_hashes": {name: digest(value) for name, value in files.items()},
        "patcher_sha256": digest(Path(__file__).read_bytes()),
        "exact_inverse_source": True,
        "scope": "One certifiedNoIO fullgrid; CTAuniform exactmask2/4/8 elsegeneric; no host sorting/cohort/launch/order/ABI/FP changes",
    }
    output.with_name(output.name + "-bundle-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patch_copy(args.source, args.output)


if __name__ == "__main__":
    main()
