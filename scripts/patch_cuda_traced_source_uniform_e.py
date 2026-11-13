"""Extend the exact captured material fact to source-bearing tile trajectories.

The existing prefix certificate proves three coefficient bit patterns over each
tile's complete owned trajectory. This pass enables it for source classes too,
and uses a CTA-uniform branch before any barrier. It retains every FP node and
all source order, rather than pre-summing source amplitudes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENTS = {
    "asymmetric_native.cuh": "f082980ad0687ef165c7807b415e87cbc1003edfdbd2acf7a4ee018dae4d4547",
    "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27",
    "native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Expected one source-material anchor: " + old)
    return text.replace(old, new, 1)


def transform(files, policies=(3, 4, 5)):
    for name, expected in PARENTS.items():
        if sha(files[name]) != expected:
            raise ValueError("Exact selected52865 required: " + name)
    c = files["asymmetric_native.cuh"].decode()
    original = c
    edits = []
    # Each kernel still has one original grid; selector is uniform per CTA.
    for plane in (False, True):
        suffix = ",true" if plane else ""
        old = f" SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20{suffix}>(p,ids);"
        condition = "||".join(f"Policy=={p}" for p in policies)
        new = (f" if constexpr({condition}){{\n"
               f"  if(p.flags[ids[blockIdx.x]]&64)SourcePendingPairedBody<Depth,1,false,true,Policy,true,1024,20{suffix}>(p,ids);\n"
               f"  else SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20{suffix}>(p,ids);\n"
               f" }}else SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20{suffix}>(p,ids);")
        c = once(c, old, new)
        edits.append((old, new))
    restored = c
    for old, new in reversed(edits):
        restored = once(restored, new, old)
    assert restored == original
    e = files["native_extension.cc"].decode()
    old = "(beamz::cuda::moving::Policy(flags)!=1&&beamz::cuda::moving::Policy(flags)!=2)"
    new = "(beamz::cuda::moving::Policy(flags)<1||beamz::cuda::moving::Policy(flags)>5)"
    e = once(e, old, new)
    # Existing public counts keep their exact meaning: sources are not observers.
    e = once(e, "++certified_noio;else ++certified_observer;",
             "++certified_noio;else if(beamz::cuda::moving::Policy(flags)==2)++certified_observer;")
    return {**files, "asymmetric_native.cuh": c.encode(), "native_extension.cc": e.encode()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policies", default="3,4,5")
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        p.error("New disjoint noncanonical output required")
    policies = tuple(int(x) for x in a.policies.split(","))
    if not policies or any(x not in (3, 4, 5) for x in policies):
        p.error("Source policies3/4/5 required")
    files = {n: (source / "src" / n).read_bytes() for n in PARENTS}
    changed = transform(files, policies)
    shutil.copytree(source, output)
    for n, data in changed.items():
        (output / "src" / n).write_bytes(data)
    proof = {"parent_hashes": PARENTS, "policies": policies,
             "result_hashes": {n: sha(v) for n, v in changed.items()},
             "point_graph_byte_identical": True,
             "fact": "Existing independent per-component exact coefficient-bit trajectory certificate",
             "launches_and_source_FP_order_unchanged": True,
             "requires_GPU_validation": True}
    (output.parent / "source-uniform-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
