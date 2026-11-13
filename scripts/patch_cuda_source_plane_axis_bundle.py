"""Compose the verified single-grid NoIO axis pass with source-plane topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_v12_noio_axis_bundle import AXIS_GUARD, once
    from scripts.validate_cuda_v12_axis_bundle import method
except ModuleNotFoundError:
    from patch_cuda_v12_noio_axis_bundle import AXIS_GUARD, once
    from validate_cuda_v12_axis_bundle import method

PARENTS = {
    "asymmetric_native.cuh": "1adf7a8145669b8000d09eae46422b5803ac04d313c511a95c7586c19115a5fc",
    "native_extension.cc": "8a9e945bcbd6a4d09a34742a22202afaff20d9dd814db40125ad9ed4dad25be4",
    "native_moving.h": "d3d1abaca2ca066b75295df35ec26da2c1df53c75bdb9f140b067fea09288b3c",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}
PLANNER_SHA256 = "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1"
EDITS = (
    ("int SourceMask=-1,bool PlanePair=false> struct Environment",
     "int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1> struct Environment"),
    ("  if(!(flags&(2u<<axis)))return false;", AXIS_GUARD),
    ("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>\n__device__",
     "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__"),
    (" Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};",
     " Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes>env{p,flags};"),
    (" // Host compact range contains only certified NoIO IDs.\n UniformBody<Depth,1,false,true,1,true>(p,tiles);",
     """ // Original full certified NoIO grid; selector uniform before any barrier.
 switch(p.flags[tiles[blockIdx.x]]&14){
  case 2:UniformBody<Depth,1,false,true,1,true,1>(p,tiles);return;
  case 4:UniformBody<Depth,1,false,true,1,true,2>(p,tiles);return;
  case 8:UniformBody<Depth,1,false,true,1,true,4>(p,tiles);return;
  default:UniformBody<Depth,1,false,true,1,true>(p,tiles);return;
 }"""),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(parent):
    for n, expected in PARENTS.items():
        if sha(parent[n]) != expected:
            raise ValueError(f"Exact source-plane-pair-v1 parent required: {n}")
    original = parent["asymmetric_native.cuh"].decode()
    c = original
    for old, new in EDITS:
        c = once(c, old, new)
    inverse = c
    for old, new in reversed(EDITS):
        inverse = once(inverse, new, old)
    if inverse != original:
        raise ValueError("Composed axis source inverse failed")
    start = "// Private source-only storage schedule; source-bound"
    assert c[c.index(start):] == original[original.index(start):]
    assert "UniformE,SourceMask,PlanePair>env" in c
    files = dict(parent)
    files["asymmetric_native.cuh"] = c.encode()
    anchor = '  module.attr("moving_source_plane_pair")=true;'
    files["native_extension.cc"] = once(parent["native_extension.cc"].decode(), anchor,
                                         '  module.attr("moving_noio_axis_bundle")=true;\n' + anchor).encode()
    return files


def patch_copy(source, output, reference_axis):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    if sha((source / "src/native_moving.cc").read_bytes()) != PLANNER_SHA256:
        raise ValueError("Actual planner trajectory flag formula changed")
    parent = {n: (source / "src" / n).read_bytes() for n in PARENTS}
    files = transformed(parent)
    # Bind the exact emitted guard to the previously CPU-compiled1.517M oracle.
    reference = reference_axis.read_bytes()
    if sha(reference) != "b5136f6f5c252f7bd3390875af55b5e9ec9753bead6668b3d6ac320146e1e917":
        raise ValueError("Verified axis reference changed")
    if method(files["asymmetric_native.cuh"].decode()) != method(reference.decode()):
        raise ValueError("Actual LocalActive method differs from verified reference")
    shutil.copytree(source, output)
    for n, data in files.items():
        (output / "src" / n).write_bytes(data)
    proof = {"parent_hashes": PARENTS, "parent_control": "source-plane-pair-v1 plusmandatoryoriginala939",
             "planner_sha256": PLANNER_SHA256,
             "result_hashes": {n: sha(d) for n, d in files.items()}, "producer_sha256": sha(Path(__file__).read_bytes()),
             "exact_consumer_inverse": True, "source_storage_launch_prepare_tail_byte_identical": True,
             "template_order": "Environment<...,SourceMask,PlanePair,ActiveAxes>; UniformBody binds<-1,false,ActiveAxes>; source paired binds<SourceMask,PlanePair> and defaultsAxes−1",
             "actual_guard_matches_verified_reference_sha256": sha(reference),
             "local_active_method_sha256": sha(method(reference.decode()).encode()),
             "canonical_cell_header_unchanged": True, "programfact_key_hidden_selectors_unchanged": True,
             "same_single_noio_grid": True, "expected_moving_kernels": 62,
             "gpu_executed": False, "gate": "Compileonlyafterparentclean; zero-local plus strictpositiveSourcePlane/NoIOaxis hardware gates andv12matchedperformance"}
    (output.parent / "source-plane-axis-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-axis", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, a.reference_axis)))


if __name__ == "__main__":
    main()
