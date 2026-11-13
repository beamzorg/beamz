"""Independent read-site, paired-cache poison and runtime bridge review.

The root pass proves temporal field versions. This companion additionally binds
every actual ImportPacket site, paired E snapshots/centers and host macro/tail
bridge to the frozen selected52865 source. Poisoning discarded current E ghosts
must preserve all actual owned reads/outputs; poisoning old E or current H must
fail. This is CPU/source evidence, not CUDA arithmetic or sanitizer evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_traced_exit_elision import CELL, PARENT, prove, transform
from scripts import prove_cuda_cube_paired_geometry as paired

CONTEXT = {
    "asymmetric_native.cuh": PARENT,
    "asymmetric_native_cell_generated.cuh": CELL,
    "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27",
    "native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
    "program.cu": "dcf437de50c8c8957c9e5074f1d06b6a8d071dc2f06a7870d6937a26a7ace06a",
}
PAIRED_ORACLE = "3d0e66dbcf700a60b22d39ad0001a12e76a0ea88224361e888cea0cc55148b55"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_review(source):
    files = {name: (source / "src" / name).read_bytes() for name in CONTEXT}
    for name, expected in CONTEXT.items():
        if sha(files[name]) != expected:
            raise ValueError(f"Frozen actual read-site/runtime bridge changed: {name}")
    consumer = files["asymmetric_native.cuh"].decode()
    host = files["native_extension.cc"].decode()
    program = files["program.cu"].decode()
    if consumer.count("ImportPacket<") != 7 or consumer.count("p.trails") != 7:
        raise ValueError("Uninventoried moving trail consumer or producer")
    sites = [
        ("StageRegisterLow", "n", "H+E; E atDepth only ghost staging, never owned current center"),
        ("UniformBody startup", "prev=first-1", "H+E; first<=Depth henceprev<=Depth-1"),
        ("UniformBody Z/Y ghost", "n", "H+E; E atDepth only ghost register replacement"),
        ("StageSourcePendingH", "n", "H only"),
        ("SourcePendingPairedBody startup", "prev=first-1", "H+E; first<=Depth henceprev<=Depth-1"),
        ("SourcePendingPairedBody Z/Y H ghost", "n", "H only"),
        ("SourcePendingPairedBody full E ghost", "n", "E only afterR0; saved/pending values used only for same ghost, no owned output"),
    ]
    required = [
        "if(window[0]>window[1])return;", "window[1]=min(Depth,",
        "const int first=window[0],prev=first-1;", "const int first=Fast?1:window[0],prev=first-1;",
        "if(zs!=hz&&ys!=hy&&sx!=hx)", "if(zs!=ehz&&sy!=ehy&&sx!=ehx)",
        "snapshots[key]=h[0][si];snapshots[1024+key]=h[1][si];",
        "// R0: retire all old E readers; publish low-X H imports.",
        "if(field>=3)return field==3?ex0[gy]:field==4?get_ey(zs,ys,sx,si):ez0[gy];",
        "if(field>=3)return field==3?ex1[gy]:field==4?get_ey(zs,ys,sx,si):ez1[gy];",
    ]
    if any(anchor not in consumer for anchor in required):
        raise ValueError("Actual cache read-site binding changed")
    macro_bridge = """            int error=BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);
            if(error)return static_cast<cudaError_t>(error);
            error=EnqueueMovingDft(launch.program,offset);
            if(error)return static_cast<cudaError_t>(error);
            std::swap(current_,other_);"""
    tail_bridge = "for(int offset=(count/depth)*depth;offset<count;++offset){const int error=BeamzEnqueueProgramStepAt(stream_.value,tail.program,offset);"
    if macro_bridge not in host or tail_bridge not in host or "inputs_[c] = current_[c];" not in host:
        raise ValueError("Next macro/tail no longer consumes completed field bank")
    if "field.data=obs.values+size_t(version)*obs.key_count;" not in host:
        raise ValueError("Chronological observation reader changed")
    ordinary = program[program.index("int BeamzEnqueueProgramStepAt("):program.index("int BeamzLaunchMstep(")]
    if "trails" in ordinary or "BeamzEnqueuePhase(raw_stream, h)" not in ordinary or "BeamzEnqueuePhase(raw_stream, e)" not in ordinary:
        raise ValueError("Ordinary tail acquired an unmodeled trail dependency")
    changed = transform(consumer)
    for anchor in required:
        if changed.count(anchor) != consumer.count(anchor):
            raise ValueError("Elision changed a bound cache read-site")
    return {"context_sha256": {name: sha(value) for name, value in files.items()},
            "seven_actual_ImportPacket_sites": sites, "sole_global_trail_reader": "ImportPacket",
            "three_E_trail_producer_sites_bound": True,
            "H_snapshots_and_owned_E_centers_unchanged": True,
            "next_macro_tail_DFT_and_graph_replay_field_bank_bridge_bound": True,
            "candidate_consumer_sha256": sha(changed.encode())}


def poison_oracle(mutation=None):
    if sha(Path(paired.__file__).read_bytes()) != PAIRED_ORACLE:
        raise ValueError("Actual paired-address oracle changed")
    code = inspect.getsource(paired.simulate)
    old = "                frame[p][f] = token(f + 3, version, p)"
    if code.count(old) != 1:
        raise ValueError("Current E ghost site changed")
    code = code.replace(old, '                frame[p][f] = ("elided-current-E-ghost",version,f,p)', 1)
    final = """    for p in frame:
        for c in range(3):
            require(frame[p][c], c + 3, version, physical(p, version), "RE full E including ghosts")"""
    owned_final = """    for p in frame:
        if ghost(p):continue
        for c in range(3):
            require(frame[p][c], c + 3, version, physical(p, version), "RE owned E outputs")"""
    if code.count(final) != 1:
        raise ValueError("Final cache/output witness changed")
    code = code.replace(final, owned_final, 1)
    if mutation == "erase-old-E-ghosts":
        code = code.replace("            frame[p][f] = token(f + 3, version - 1, p)",
                            '            frame[p][f] = ("wrong-old-E-ghost",f,p) if ghost(p) else token(f + 3, version - 1, p)', 1)
    elif mutation == "erase-current-H-ghosts":
        code = code.replace("                h[f, p] = token(f, version, p)",
                            '                h[f, p] = ("wrong-current-H-ghost",f,p)', 1)
    elif mutation == "admit-ghost-E-center":
        code = code.replace("                if not ghost(p):", "                if True:", 1)
    elif mutation == "reuse-old-cache-for-next-macro":
        anchor = "    return {\"cache\": g.shape"
        extra = """    q=(heads[0],(heads[1]+2)%g.y,(heads[2]+1)%g.x)
    require(frame[q][2],5,version,physical(q,version),"next macro H transverse E read without final-bank bootstrap")
"""
        code = code.replace(anchor, extra + anchor, 1)
    elif mutation is not None:
        raise ValueError("Unknown review mutation")
    namespace = dict(vars(paired))
    exec(compile(code, "<bound-paired-exit-poison-oracle>", "exec"), namespace)
    return namespace["simulate"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = source_review(args.source)
    simulate = poison_oracle()
    cases = [simulate(paired.Geometry(8, 16, 32), version=depth, seed=seed)
             for depth in (4, 16) for seed in (7, 13, 29, 41)]
    negatives = []
    for name in ("erase-old-E-ghosts", "erase-current-H-ghosts", "admit-ghost-E-center", "reuse-old-cache-for-next-macro"):
        try:
            poison_oracle(name)(paired.Geometry(8, 16, 32), version=16, seed=13)
        except paired.Hazard as exc:
            negatives.append({"mutation": name, "rejected": True, "witness": str(exc)})
        else:
            raise AssertionError(f"Unmodeled live consumer escaped: {name}")
    late_starts = [{"depth": depth, "admitted_first_steps": list(range(1, depth+1)),
                    "maximum_startup_import_version": max(first-1 for first in range(1, depth+1))}
                   for depth in (4, 16)]
    review.update({"schema": "beamz.exit-elision-independent-review.v1", "passed": True,
                   "reviewer": "trace_lifetime", "producer_sha256": sha(Path(__file__).read_bytes()),
                   "root_pass_sha256": sha(Path(sys.modules[transform.__module__].__file__).read_bytes()),
                   "paired_oracle_sha256": PAIRED_ORACLE,
                   "root_temporal_version_proofs": [prove(depth) for depth in (4, 16)],
                   "current_ghost_E_poison_positive_cases": cases,
                   "actual_paired_reads_checked_with_poison": sum(case["checked_reads"] for case in cases),
                   "negative_live_consumer_controls": negatives, "late_start_bounds": late_starts,
                   "finding": "No unmodeled consumer found in frozen selected52865; currentDepthEghosts/trails are dead while finalEfields/Observe remain live and retained.",
                   "integration_proof_gap_in_root_pass": "Root pass alone hash-binds consumer/cell but only asserts host bridge; this independent artifact additionally binds actual host/runtime files.",
                   "limitations": "Source and CPU token/address/visibility evidence only; CUDA numerical, actual dispatch, sanitizer and complete workload gates remain required."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps({"passed": True, "positive_poison_cases": len(cases), "negative_rejections": len(negatives),
                      "paired_actual_reads_checked": review["actual_paired_reads_checked_with_poison"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
