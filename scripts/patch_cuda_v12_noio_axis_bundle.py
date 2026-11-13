"""Copy-only NoIO axis bundle on the exact paired-source v12 winner.

SourceMask and ActiveAxes are distinct Environment template arguments. Only the
existing certified NoIO1024 entry branches; source bodies and all host launches,
queues, clocks, keys and certificates remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_MODULE = "a9394c3031d6635a875d673d50e7e09123807b7d9d2fe353c49fa05063b5941b"
PARENT_HASHES = {
    "asymmetric_native.cuh": "3b4039564fad23a88c8e801768a93b1ced302643d640886095e0f0ad1fc29aaf",
    "native_extension.cc": "ec7f48eb42ce5a6c5136e88ea2542c809d786762e26fe9bae8f8a2c211787f45",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
    "native_moving.h": "917fe9043362071cd1cac5ba503571a72de3d34b51b5e3c68eeea1fa35dbd7a9",
}
AXIS_GUARD = "  if constexpr(ActiveAxes>=0){if(!(ActiveAxes&(1u<<axis)))return false;}\n  else if(!(flags&(2u<<axis)))return false;"
EDITS = (
    ("bool UniformE=false,int SourceMask=-1> struct Environment",
     "bool UniformE=false,int SourceMask=-1,int ActiveAxes=-1> struct Environment"),
    ("  if(!(flags&(2u<<axis)))return false;", AXIS_GUARD),
    ("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>\n__device__",
     "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__"),
    (" Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};",
     " Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,ActiveAxes>env{p,flags};"),
    (" // Host compact range contains only certified NoIO IDs.\n UniformBody<Depth,1,false,true,1,true>(p,tiles);",
     """ // Existing certified NoIO prefix; immutable selector is CTA-uniform.
 // One original grid/launch, with no host sorting or serial axis subgrids.
 switch(p.flags[tiles[blockIdx.x]]&14){
  case 2:UniformBody<Depth,1,false,true,1,true,1>(p,tiles);return;
  case 4:UniformBody<Depth,1,false,true,1,true,2>(p,tiles);return;
  case 8:UniformBody<Depth,1,false,true,1,true,4>(p,tiles);return;
  default:UniformBody<Depth,1,false,true,1,true>(p,tiles);return;
 }"""),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact v12 axis anchor: {old!r}")
    return text.replace(old, new, 1)


def compose_text(text):
    result = text
    for old, new in EDITS:
        result = once(result, old, new)
    inverse = result
    for old, new in reversed(EDITS):
        inverse = once(inverse, new, old)
    if inverse != text:
        raise ValueError("Exact source inverse failed")
    return result


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact a939 v12 parent required: {name}")
    original = parent["asymmetric_native.cuh"].decode()
    result = compose_text(original)
    # Entire appended paired-source body and all launch/Prepare code are exact.
    boundary = "// Private source-only storage schedule; source-bound"
    if result[result.index(boundary):] != original[original.index(boundary):]:
        raise ValueError("Paired source/host launch region changed")
    if "Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask>env" not in result:
        raise ValueError("SourceMask callback binding lost")
    files = dict(parent)
    files["asymmetric_native.cuh"] = result.encode()
    old = '  module.attr("moving_source_pending_mask20_1024")=true;'
    new = old + '\n  module.attr("moving_noio_axis_bundle")=true;'
    files["native_extension.cc"] = once(parent["native_extension.cc"].decode(), old, new).encode()
    return files


def dispatch_axis(flags):
    """Typed logical axis bits, or -1 for the original runtime guard."""
    return {2: 1, 4: 2, 8: 4}.get(flags & 14, -1)


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents
            or output in source.parents or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical source output required")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {
        "parent_module_sha256": PARENT_MODULE,
        "parent_hashes": PARENT_HASHES,
        "result_hashes": {name: sha(data) for name, data in files.items()},
        "patcher_sha256": sha(Path(__file__).read_bytes()),
        "exact_inverse_source": True,
        "changed_consumer_regions": len(EDITS),
        "source_pending_and_all_host_launches_byte_identical": True,
        "canonical_cell_byte_identical": True,
        "native_workspace_and_host_key_byte_identical": True,
        "scope": "CTA-uniform single-axis guard specialization on same certified NoIO grid; no FP graph/order or mutable psi publication changes",
        "template_binding": "Environment<...,SourceMask=-1,ActiveAxes>; source Environment<...,SourceMask> retains default ActiveAxes=-1",
        "requires": ["CPU actual LocalActive equivalence", "all moving resource gate", "v12-matched numerical/sanitizer/performance gates"],
        "gpu_executed": False,
    }
    (output.parent / "axis-composition-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output)))


if __name__ == "__main__":
    main()
