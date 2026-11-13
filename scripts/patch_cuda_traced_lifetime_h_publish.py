"""Derive minimal H-frame publication predicates from exact E-phase reads.

Private H values remain dense, retaining ghosts that will enter the next moving
owned window. Only the temporary H shared frame between B2 and B3 is sparse.
The old E frame and every E publication remain dense. This distinction matters:
dropping E ghost stores would corrupt the following moving H phase.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_stencil_trace import reads, trace_yee_step
from scripts.patch_cuda_traced_lifetime_warp_x import PARENTS, BODY_BEGIN, BODY_END, once

CONSUMERS = {
    "none": PARENTS["asymmetric_native.cuh"],
    "h": "80b473fa3739ac7eaf16decc67e84276d5bca2629bb545fb71d450e3efca7376",
    "e": "9ce52e781a1844df5e6e2e895da570cc4594fb6ea353b57cd26d8d3a73ad9abc",
    "both": "fc708c9797aa903b89db3532fc76179e6058b55a7978d5b2143238598155688b",
}
PUBLICATION = """  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]=h[c][j];
  }
  __syncthreads(); // B2 publishes all current H values."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def shared_reads(warp_mode="both"):
    if warp_mode not in CONSUMERS:
        raise ValueError("Unknown warp lowering mode")
    result = {name: set() for name in ("Hx", "Hy", "Hz")}
    for update in trace_yee_step():
        if update.output.field[0] != "E":
            continue
        for read in reads(update.value):
            if read.field[0] != "H" or not any(read.offset):
                continue  # Own center values are private.
            if read.offset[2] and warp_mode in ("e", "both"):
                continue  # These exact reads use unconditional warp exchange.
            if any(value not in (-1, 0) for value in read.offset) or sum(map(abs, read.offset)) != 1:
                raise ValueError("The actual E stencil is not the admitted unit negative-neighbor class")
            result[read.field].add(read.offset)
    if any(not value for value in result.values()):
        raise ValueError("Missing H component publication footprint")
    return {name: sorted(offsets) for name, offsets in result.items()}


def predicate(offsets):
    """Writer p is live iff at least one owned target p-offset consumes it."""
    variables, high = ("dz", "dy", "dx"), (7, 15, 31)
    terms = []
    for offset in offsets:
        terms.append("(" + "&&".join(f"{variable}!={limit if delta == -1 else 0}"
                                     for variable, limit, delta in zip(variables, high, offset, strict=True)) + ")")
    return "||".join(terms)


def generated_publication(warp_mode="both"):
    footprints = shared_reads(warp_mode)
    lines = ["  #pragma unroll", "  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,dz=(zs-hz)&7;"]
    for component, name in enumerate(("Hx", "Hy", "Hz")):
        lines.append(f"   if({predicate(footprints[name])})frame[{component}*PAD+Slot(zs,sy,sx)]=h[{component}][j];")
    lines += ["  }", "  __syncthreads(); // B2 publishes every H word consumed by the traced E phase."]
    return "\n".join(lines)


def footprint_predicate(point, offsets, high=(7, 15, 31)):
    # CPU semantics of the generated per-axis != comparisons.
    return any(all(value != (maximum if shift == -1 else 0)
                   for value, maximum, shift in zip(point, high, offset, strict=True))
               for offset in offsets)


def prove_publication(warp_mode="both", *, omit_z0_hy=False, sparse_e_ghosts=False):
    footprints = shared_reads(warp_mode)
    owned = tuple(itertools.product(range(1, 8), range(1, 16), range(1, 32)))
    cache = tuple(itertools.product(range(8), range(16), range(32)))
    expected = {name: {tuple(p + d for p, d in zip(target, offset, strict=True))
                       for target in owned for offset in offsets}
                for name, offsets in footprints.items()}
    actual = {name: {point for point in cache if footprint_predicate(point, offsets)
                     and not (omit_z0_hy and name == "Hy" and point[0] == 0)}
              for name, offsets in footprints.items()}
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"Generated H predicate misses a consumed {name} shared word")
    if sparse_e_ghosts:
        # Current Ez Z-ghost values become transverse +Y neighbors of an owned
        # next-step Hx point after the window shifts by (-1,-1,-1). Centers are
        # private, but this transverse neighbor is genuinely read from shared.
        current_ghost = (0, 2, 1)
        next_owned_target_in_current_coordinates = (0, 1, 1)
        read = tuple(p + d for p, d in zip(next_owned_target_in_current_coordinates, (0, 1, 0), strict=True))
        if read == current_ghost:
            raise ValueError("Sparse E publication removes a next-window ghost dependency")
    payload_owners = 0
    for threads in (512, 1024):
        zstep, slots = threads // 512, 4096 // threads
        for step in range(1, 17):
            hz, hy, hx = (-step - 1) & 7, (-step - 1) & 15, (-step - 1) & 31
            published = {name: set() for name in actual}
            for tid in range(threads):
                sx, sy, zbase = tid & 31, (tid // 32) & 15, tid // 512
                for j in range(slots):
                    zs = zbase + j * zstep
                    point = ((zs - hz) & 7, (sy - hy) & 15, (sx - hx) & 31)
                    for name, offsets in footprints.items():
                        if footprint_predicate(point, offsets):
                            # Each private owner must publish each needed word once.
                            if point in published[name]:
                                raise ValueError("H shared publication has multiple writers")
                            published[name].add(point)
            if published != expected:
                raise ValueError("Actual ring owner mapping changes the traced publication footprint")
            payload_owners += sum(map(len, published.values()))
    return {"warp_mode": warp_mode, "traced_shared_neighbor_offsets": footprints,
            "published_h_words_per_step": {name: len(points) for name, points in actual.items()},
            "total_published_h_words_per_step": sum(map(len, actual.values())),
            "dense_h_words_per_step": 12288, "checked_single_writer_payloads": payload_owners,
            "thread_counts": [512, 1024], "steps": 16,
            "dense_private_ghost_state_and_dense_E_shared_publication_retained": True}


def transform(parent, warp_mode="both"):
    if warp_mode not in CONSUMERS:
        raise ValueError("Unknown warp lowering mode")
    for name, expected in PARENTS.items():
        if sha(parent[name]) != (CONSUMERS[warp_mode] if name == "asymmetric_native.cuh" else expected):
            raise ValueError(f"Exact selected52865/warp parent required: {name}")
    old = parent["asymmetric_native.cuh"].decode()
    start, end = old.index(BODY_BEGIN), old.index(BODY_END)
    old_body = old[start:end]
    publication = generated_publication(warp_mode)
    new_body = once(old_body, PUBLICATION, publication)
    if once(new_body, publication, PUBLICATION) != old_body:
        raise ValueError("H publication exact source inverse failed")
    changed = old[:start] + new_body + old[end:]
    if changed.count("__syncthreads()") != old.count("__syncthreads()"):
        raise ValueError("Synchronization changed")
    result = dict(parent)
    result["asymmetric_native.cuh"] = changed.encode()
    return result


def patch_copy(source, output, warp_mode="both"):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint private source required")
    parent = {n: (source / "src" / n).read_bytes() for n in PARENTS}
    result = transform(parent, warp_mode)
    proof = prove_publication(warp_mode)
    proof.update({"parent_hashes": {n: sha(data) for n, data in parent.items()},
                  "result_hashes": {n: sha(data) for n, data in result.items()},
                  "exact_source_inverse": True, "gpu_executed": False,
                  "unchanged": ["ordered FP32 cell graph", "CPML state/profile accesses", "private fields",
                                "all E-frame stores", "all barriers/imports", "source/DFT order", "host admission"],
                  "requires": ["zero stack/local/spill exact-context build", "all30 exact GPU parity",
                               "positive memcheck/synccheck/racecheck", "matched full-workload measurement"]})
    shutil.copytree(source, output)
    for name, data in result.items():
        (output / "src" / name).write_bytes(data)
    (output.parent / "traced-h-publication-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warp-mode", choices=tuple(CONSUMERS), default="both")
    args = parser.parse_args()
    print(json.dumps(patch_copy(args.source, args.output, args.warp_mode)))


if __name__ == "__main__":
    main()
