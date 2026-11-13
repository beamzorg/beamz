"""Restore captured one-hot CPML partial evaluation inside persistent NoIO.

The existing immutable tile flags already certify axes over the entire moving
trajectory. This copied-source pass keeps the multi-axis fallback and every
ordered cell/barrier statement, specializing only the environment's axis mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "b7814cd9792212cea639211c06cb985f0f588282a076c7f6077056e4145fe0c0"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform(old):
    if sha(old.encode()) != PARENT:
        raise ValueError("Exact direct-global specialized parent required")
    text, edits = old, []

    def change(a, b):
        nonlocal text
        if text.count(a) != 1:
            raise ValueError("One exact persistent axis anchor required: " + a)
        text = text.replace(a, b, 1)
        edits.append((a, b))

    change("template<bool Fast,int Policy,bool UniformE>\nstruct PersistentUniformEnvironment:Environment<1,Fast,true,Policy,UniformE>{\n using Base=Environment<1,Fast,true,Policy,UniformE>;",
           "template<bool Fast,int Policy,bool UniformE,int ActiveAxes=-1>\nstruct PersistentUniformEnvironment:Environment<1,Fast,true,Policy,UniformE,-1,false,ActiveAxes>{\n using Base=Environment<1,Fast,true,Policy,UniformE,-1,false,ActiveAxes>;")
    change("template<int Depth,bool Fast,int Policy,bool UniformE>\n__device__ __forceinline__ void PersistentUniformBody(const Parameters&p){\n constexpr int Layout=1,ActiveAxes=-1;",
           "template<int Depth,bool Fast,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void PersistentUniformBody(const Parameters&p){\n constexpr int Layout=1;")
    change("PersistentUniformEnvironment<Fast,Policy,UniformE>env{p,flags};",
           "PersistentUniformEnvironment<Fast,Policy,UniformE,ActiveAxes>env{p,flags};")
    change("  else if(C().body_kind==1)PersistentUniformBody<Depth,false,1,true>(p);",
           """  else if(C().body_kind==1){
   // Immutable per-tile full-trajectory mask; every thread takes one arm.
   switch(p.flags[C().tile]&14){
    case 2:PersistentUniformBody<Depth,false,1,true,1>(p);break;
    case 4:PersistentUniformBody<Depth,false,1,true,2>(p);break;
    case 8:PersistentUniformBody<Depth,false,1,true,4>(p);break;
    default:PersistentUniformBody<Depth,false,1,true>(p);break;
   }
  }""")
    inverse = text
    for a, b in reversed(edits):
        if inverse.count(b) != 1:
            raise ValueError("Axis specialization inverse is ambiguous")
        inverse = inverse.replace(b, a, 1)
    if inverse != old or text.count("__syncthreads()") != old.count("__syncthreads()"):
        raise ValueError("Unrelated body/barrier mutation")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        p.error("New disjoint noncanonical output required")
    old = (source / "src/native_persistent.cuh").read_text()
    new = transform(old)
    shutil.copytree(source, output)
    (output / "src/native_persistent.cuh").write_text(new)
    proof = dict(parent_consumer_sha256=PARENT, consumer_sha256=sha(new.encode()),
                 producer_sha256=sha(Path(__file__).read_bytes()), exact_source_inverse=True,
                 axis_cases={2: 1, 4: 2, 8: 4}, other_masks_keep_original_generic=True,
                 ordered_cell_and_barrier_statements_unchanged=True, expected_entries=65,
                 gpu_executed=False, scope="captured full-trajectory CPML mask partial evaluation only")
    (output.parent / "persistent-noio-axis-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
