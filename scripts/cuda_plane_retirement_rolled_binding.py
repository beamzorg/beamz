"""Prove the uniform E-plane selector and bind its exact parent derivation.

No schedule or field lifetime changes are admitted: removing the selector and
rolling directive must reconstruct the already admitted 18-barrier parent byte
for byte. Fixed-index scalarization is a source request, not a compiler promise;
resource and SASS checks remain required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_plane_retirement_binding import braced_body, digest, validate

OLD_PREFIX = "  #pragma unroll\n  for(int j=0;j<Slots;++j){float current[3];"
NEW_PREFIX = """  #pragma unroll 1
  for(int j=0;j<Slots;++j){float current[3],hp[3]={0.f,0.f,0.f};
   // Every H array index becomes a compile-time constant. The runtime j is
   // CTA-uniform and selects register values, never a local-memory address.
   #pragma unroll
   for(int k=0;k<Slots;++k)if(j==k){hp[0]=h[0][k];hp[1]=h[1][k];hp[2]=h[2][k];}"""
END = "  StagePlaneLow<Depth,SupportSafe,1>"
ROLLED_SHA = "ce59c99112926421aa5abc2c14e36bc9334b35e582b744215ba3d358a7d74a44"


def selector_certificate(*, wrong_selector=False, barrier_inside_owned=False):
    planes, workers, steps = 16, 512, 17
    selected, participation = 0, 0
    for j in range(planes):
        for worker in range(workers):
            # Distinct tokens prove all three component and slot identities.
            registers = tuple(tuple((c, k, worker) for k in range(planes)) for c in range(3))
            hp, matches = [None] * 3, 0
            for k in range(planes):
                if (j + int(wrong_selector)) % planes == k:
                    hp = [registers[c][k] for c in range(3)]
                    matches += 1
            if matches != 1 or hp != [registers[c][j] for c in range(3)]:
                raise ValueError("Runtime E selector changed H component/slot identity")
            selected += 3
    expected = tuple(range(planes - 1))
    for n in range(1, steps + 1):
        hz, hy, hx = (-n - 1) % 16, (-n - 1) % 16, (-n - 1) % 32
        for worker in range(workers):
            y, x = divmod(worker, 32)
            visited = tuple(j for j in range(planes - 1)
                            if not barrier_inside_owned or (j != hz and y != hy and x != hx))
            if visited != expected:
                raise ValueError("Plane publication barrier has divergent ghost participation")
            participation += len(visited)
    return {"planes": planes, "threads": workers, "wrapped_versions": steps,
            "selected_component_tokens": selected,
            "uniform_barrier_participations": participation,
            "selector_matches_per_thread_plane": 1, "result": "pass"}


def reconstruct_parent(source):
    if source.count(NEW_PREFIX) != 1 or source.count(END) != 1:
        raise ValueError("Uniform runtime E loop or exact constant-k selector changed")
    start = source.index(NEW_PREFIX)
    end = source.index(END, start)
    block = source[start:end]
    if re.search(r"h\[[^\]]+\]\[j\]", block):
        raise ValueError("Dynamic private H indexing remains in the runtime E loop")
    if block.count("hp[c]") != 3 or block.count("hp[field]") != 1:
        raise ValueError("Publication, center, trail or final H selector use changed")
    body = braced_body(block, "#pragma unroll 1\n  for(int j=0;j<Slots;++j)")
    if body.count("__syncthreads();") != 1:
        raise ValueError("Unexpected runtime E barrier count")
    # The one barrier is outside every per-thread scope. Its sole condition
    # j!=CZ-1 depends only on the CTA-uniform induction variable.
    clean = re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: " " * len(m.group()), body, flags=re.DOTALL)
    barrier = clean.index("__syncthreads();")
    if clean[:barrier].count("{") != clean[:barrier].count("}"):
        raise ValueError("Runtime E barrier moved inside a per-thread scope")
    if "if(j!=CZ-1)__syncthreads();" not in body:
        raise ValueError("Runtime E barrier lost its uniform final-plane condition")
    restored = block.replace(NEW_PREFIX, OLD_PREFIX, 1)
    restored = restored.replace("hp[c]", "h[c][j]").replace("hp[field]", "h[field][j]")
    parent = source[:start] + restored + source[end:]
    if "AsymmetricRetireLoopE" not in parent:
        raise ValueError("Missing uniquely identified rolled-E kernel")
    return parent.replace("AsymmetricRetireLoopE", "AsymmetricRetirePlane")


def admit(source, cell, proof):
    parent = reconstruct_parent(source)
    parent_admission = validate(parent, cell, proof)
    if digest(source) != ROLLED_SHA:
        raise ValueError("Rolled E consumer differs from independently reviewed source")
    selector = selector_certificate()
    return {"schema": "beamz.cuda-plane-retirement-rolled-admission.v1",
            "admitted_source": True, "consumer_sha256": digest(source),
            "reconstructed_parent_sha256": digest(parent),
            "source_binding_sha256": digest(Path(__file__).read_bytes()),
            "parent_source_admission": parent_admission,
            "barriers_per_step": 18, "selector": selector,
            "derivation": "Exact inverse recovers admitted parent; only E loop rolling, fixed-k hp selection and kernel identity differ",
            "scope": "Source-level dependency/participation admission; compiler scalarization, resource, arithmetic and sanitizer gates remain required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.consumer, args.cell, args.proof)}:
        parser.error("Admission output must be distinct from source and proof inputs")
    result = admit(args.consumer.read_bytes().decode(), args.cell.read_bytes().decode(), json.loads(args.proof.read_text()))
    result["proof_file_sha256"] = digest(args.proof.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
