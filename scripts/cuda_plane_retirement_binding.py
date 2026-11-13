"""Fail-closed binding of the reviewed 18-barrier source to its CPU proof.

This is a source admission certificate, not a C++ verifier or GPU parity gate.
Any consumer/cell edit, including formatting, requires explicit re-review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_stencil_trace import serialize_updates, trace_cpml_step
from scripts.generate_cuda_native_cell import generate_native_cell

CELL_SHA = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
CONSUMER_SHA = "bde8854b6924b052bd68839534572ec798f48fc97ab5a87f63e567aaf74e15a8"
ORDER_ANCHORS = (
    "StagePlaneLow<Depth,SupportSafe,0>(p,low_stage,bases,n,oz,oy,ox);",
    "__syncthreads(); // R1:",
    "RetirePlaneH(low_stage,h,n,sy,sx);",
    "last[c]=frame[c*PAD+Slot(CZ-1,sy,sx)];frame[c*PAD+Slot(CZ-1,sy,sx)]=h[c][CZ-1];",
    "for(int j=0;j<Slots;++j){float current[3];",
    "if(j!=CZ-1)__syncthreads(); // P0",
    "AdvanceNativeCell<1,0,false>(env,read,n,z,y,x);",
    "if(j==CZ-1)last[c]=current[c];else{if(j>=1)frame[c*PAD+Slot(j-1,sy,sx)]=pending[c];pending[c]=current[c];}",
    "StagePlaneLow<Depth,SupportSafe,1>(p,low_stage,bases,n,oz,oy,ox);",
    "__syncthreads(); // R3 retires ALL E(last-1)/Elast readers",
    "frame[c*PAD+Slot(CZ-2,sy,sx)]=pending[c];frame[c*PAD+Slot(CZ-1,sy,sx)]=last[c];",
    "frame[Slot(j,sy,sx)]=v.ex;frame[PAD+Slot(j,sy,sx)]=v.ey;frame[2*PAD+Slot(j,sy,sx)]=v.ez;",
    "frame[c*PAD+Slot(j,sy,sx)]=low_stage[c*LowXPoints+key];",
    "if(!threadIdx.x)*phase_step=n+1;",
    "__syncthreads(); // R4:",
)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def braced_body(source, anchor):
    """Extract one C++ brace region, ignoring comments and string literals."""
    if source.count(anchor) != 1:
        raise ValueError(f"Expected exactly one anchor: {anchor}")
    clean = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',
                   lambda match: " " * len(match.group()), source, flags=re.DOTALL)
    start = clean.index("{", source.index(anchor))
    depth = 1
    for end in range(start + 1, len(clean)):
        depth += (clean[end] == "{") - (clean[end] == "}")
        if not depth:
            return source[start + 1:end]
    raise ValueError("Unbalanced C++ body")


def inspect_schedule(step):
    """Check actual emitted operational order and static barrier multiplicity."""
    positions = []
    for anchor in ORDER_ANCHORS:
        if step.count(anchor) != 1:
            raise ValueError(f"Missing/duplicated retirement order anchor: {anchor}")
        positions.append(step.index(anchor))
    if positions != sorted(positions):
        raise ValueError("Retirement anchor order changed")
    lines = re.findall(r"__syncthreads\(\);[^\n]*", step)
    labels = ("R1", "P0", "R3", "R4")
    if len(lines) != 4 or any(label not in line for label, line in zip(labels, lines, strict=True)):
        raise ValueError("Unexpected emitted barrier sites")
    sites = [{"label": label, "emitted_source": line,
              "executions_per_step": 15 if label == "P0" else 1}
             for label, line in zip(labels, lines, strict=True)]
    return {"barrier_sites": sites, "static_barrier_sites": 4,
            "barriers_per_step": sum(site["executions_per_step"] for site in sites),
            "ordered_anchor_sha256": digest(json.dumps(ORDER_ANCHORS)),
            "timestep_body_sha256": digest(step)}


def validate(consumer, cell, proof):
    if digest(cell) != CELL_SHA or generate_native_cell()["source_sha256"] != CELL_SHA:
        raise ValueError("Canonical native cell changed; regenerate and re-prove")
    proof_file = Path(__file__).with_name("prove_cuda_plane_retirement.py")
    if proof["proof_source_sha256"] != digest(proof_file.read_bytes()):
        raise ValueError("Stale plane-retirement proof source")
    operator_sha = digest(json.dumps(serialize_updates(trace_cpml_step(active_axes=(0, 1, 2))), sort_keys=True))
    if proof["canonical_native_operator_sha256"] != operator_sha:
        raise ValueError("Proof/operator mismatch")
    required = {"early_pending_before_final_Q", "ghost_restore_before_final_plane_store",
                "early_stage_reuse", "overwrite_reserved_last", "missing_last_old", "wrong_z_wrap"}
    if not required.issubset(proof["negative_controls_rejected"]):
        raise ValueError("Required lifetime negative controls are missing")
    cases = proof["cases"]
    if (any(case["result"] != "pass" for case in cases)
            or not any(case["cache_zyx"] == [16, 16, 32] and case["steps"] >= 17 for case in cases)):
        raise ValueError("Missing successful actual-geometry wrap proof")
    if "BZ=15,BY=15,BX=31,CZ=16,CY=16,CX=32,Threads=512" not in consumer:
        raise ValueError("Only reviewed owned15/cache16/512-thread geometry is admitted")
    if "void AsymmetricRetirePlane(" not in consumer:
        raise ValueError("Missing unique plane-retirement kernel symbol")
    body = braced_body(consumer, "void UniformBody(")
    step = braced_body(body, "for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n)")
    schedule = inspect_schedule(step)
    if digest(consumer) != CONSUMER_SHA:
        raise ValueError("Consumer changed beyond the exact independently reviewed source")
    return {"schema": "beamz.cuda-plane-retirement-admission.v1",
            "admitted_source": True, "consumer_sha256": digest(consumer),
            "native_cell_sha256": CELL_SHA, "proof_source_sha256": proof["proof_source_sha256"],
            "canonical_native_operator_sha256": operator_sha,
            "proof_payload_sha256": digest(json.dumps(proof, sort_keys=True)),
            "binding_source_sha256": digest(Path(__file__).read_bytes()),
            "cached_zyx": [16, 16, 32], "threads": 512,
            "schedule": "repaired-last-use18", **schedule,
            "scope": "Exact reviewed source admission only; resource, all30 arithmetic, race/sync gates remain required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in (args.consumer, args.cell, args.proof)}:
        parser.error("Admission output must be distinct from source and proof inputs")
    # Preserve exact source bytes, including newline spelling, for SHA binding.
    result = validate(args.consumer.read_bytes().decode(), args.cell.read_bytes().decode(), json.loads(args.proof.read_text()))
    result["proof_file_sha256"] = digest(args.proof.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
