"""Point-certified coherent CG/CA psi loads for persistent CUDA experiments.

Fresh points use CG after scheduler acquisition. Continued owned points use
coherent CA only after their same-thread prior scalar StoreLocal. No read-only
or noncoherent loads are introduced. GPU stress and exact parity remain gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


PARENTS = {
    "p5": "2e3408b1d514a8b0d90620cbb495199a438e4e5f729b1f7c6830dc1f3c2174b2",
    "specialized": "b7814cd9792212cea639211c06cb985f0f588282a076c7f6077056e4145fe0c0",
}
BOUND = {"asymmetric_native.cuh": "87920106bbfbc40fc2ae2240afe1159041081215ec856bbf3e60f5900bbbe71f",
         "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
         "native_persistent.h": "00b85f62ad6128a8d391128a6cc1f6d2459daf9e8fa45a91fff7c3877d44160f",
         "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27"}
HELPER = r'''
// Bit128 is point-local load provenance; all original low7 tile facts survive.
__device__ __forceinline__ uint8_t PersistentMixedPsiFlags(uint8_t facts,int n,int first,int dz,int dy,int dx){
 return uint8_t((facts&127u)|((n==first||dz==1||dy==1||dx==1)?128u:0u));
}
'''
LOAD = "  return __ldcg(this->p.phase[slot/6].psi[slot%6].input+this->PsiOffset(slot,z,y,x));"
MIXED_LOAD = """  const float*word=this->p.phase[slot/6].psi[slot%6].input+this->PsiOffset(slot,z,y,x);
  // Continued CA follows this same thread's prior scalar store; fresh CG
  // follows the existing acquire and CTA barrier. Both are coherent loads.
  return (this->flags&128u)?__ldcg(word):__ldca(word);"""
SITE_ASSIGNMENTS = {
    "paired_H": "env.flags=PersistentMixedPsiFlags(flags,n,window[0],(zs-hz)&(CZ-1),(ys-hy)&(CY-1),(sx-hx)&31);",
    "paired_E0_E1": "env.flags=PersistentMixedPsiFlags(flags,n,window[0],dz,dy,dx);",
    "uniform_H": "if constexpr(!Fast)env.flags=PersistentMixedPsiFlags(flags,n,(Fast?1:window[0]),(zs-hz)&7,dy,dx);",
    "uniform_E": "if constexpr(!Fast)env.flags=PersistentMixedPsiFlags(flags,n,(Fast?1:window[0]),(zs-ehz)&7,(sy-ehy)&15,(sx-ehx)&31);",
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def transformed(text, variant):
    if variant not in PARENTS or sha(text.encode()) != PARENTS[variant]:
        raise ValueError("Exact reviewed floor/global persistent consumer required")
    if re.search(r"(?:this->)?flags\s*(?:==|!=)", text):
        raise ValueError("Whole-byte flag comparison requires independent fresh-bit masking")
    old, edits = text, []

    def change(before, after, count=1):
        nonlocal text
        if text.count(before) != count:
            raise ValueError("Exact mixed-psi anchor count differs:" + before)
        text = text.replace(before, after)
        edits.append((before, after, count))

    change("struct PersistentEnvironment:", HELPER + "\nstruct PersistentEnvironment:")
    change(LOAD, MIXED_LOAD, 2 if variant == "specialized" else 1)
    ph = "     if(zs!=hz&&ys!=hy&&sx!=hx){\n      const auto read="
    change(ph, ph.replace("\n      const auto read=", "\n      " + SITE_ASSIGNMENTS["paired_H"] + "\n      const auto read="))
    pe = "    if(zs!=hz&&ys!=hy&&sx!=hx){\n     const auto read="
    change(pe, pe.replace("\n     const auto read=", "\n     " + SITE_ASSIGNMENTS["paired_E0_E1"] + "\n     const auto read="), 2)
    if variant == "specialized":
        uh = "    if(zs!=hz&&sy!=hy&&sx!=hx){\n     const auto read="
        ue = "    if(zs!=ehz&&sy!=ehy&&sx!=ehx){\n     const auto read="
        change(uh, uh.replace("\n     const auto read=", "\n     " + SITE_ASSIGNMENTS["uniform_H"] + "\n     const auto read="))
        change(ue, ue.replace("\n     const auto read=", "\n     " + SITE_ASSIGNMENTS["uniform_E"] + "\n     const auto read="))
    inverse = text
    for before, after, count in reversed(edits):
        if inverse.count(after) != count:
            raise ValueError("Mixed-psi exact inverse became ambiguous")
        inverse = inverse.replace(after, before)
    if inverse != old or text.count("__syncthreads()") != old.count("__syncthreads()") or text.count("__threadfence()") != old.count("__threadfence()"):
        raise ValueError("Mixed-psi exact inverse/barrier/fence audit failed")
    if text.count("env.flags=PersistentMixedPsiFlags(") != (5 if variant == "specialized" else 3):
        raise ValueError("Not all and only owned point update sites are covered")
    return text


def patch_copy(source, output, variant, review):
    source, output, review = Path(source).resolve(), Path(output).resolve(), Path(review).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source required")
    for name, expected in BOUND.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError("Actual CPML graph/address/slot/host binding changed:" + name)
    review_data = json.loads(review.read_text())
    if review_data["schema"] != "beamz.persistent-mixed-psi-independent-review.v1" or sum(c["continued_normal_reads_with_prior_store"] for c in review_data["cases"]) != 952560:
        raise ValueError("Expected independent prior-store review required")
    old = (source / "src/native_persistent.cuh").read_text()
    new = transformed(old, variant)
    environment = (source / "src/asymmetric_native.cuh").read_text()
    if re.search(r"(?:this->)?flags\s*(?:==|!=)", environment):
        raise ValueError("Whole-byte original Environment comparison needs masking")
    shutil.copytree(source, output)
    (output / "src/native_persistent.cuh").write_text(new)
    proof = {"schema": "beamz.persistent-point-certified-mixed-psi.v1", "variant": variant,
             "parent_consumer_sha256": PARENTS[variant], "result_consumer_sha256": sha(new.encode()),
             "unchanged_bound_files": BOUND, "independent_review_sha256": sha(review.read_bytes()),
             "point_assignment_sites": 5 if variant == "specialized" else 3,
             "per_point_fresh_predicate": "n==first OR owned dz==1 OR dy==1 OR dx==1",
             "P5_first": "volatile window[0] read at each point; no new first live range",
             "fresh_load": "__ldcg coherent", "continued_load": "__ldca coherent; never read-only/noncoherent",
             "low7_flags_preserved": True, "whole_byte_flag_comparisons_present": False,
             "uniform_Fast_bulk_point_assignment_compiled_out": variant == "specialized",
             "exact_source_inverse": True, "barriers_fences_FP_nodes_and_workspace_unchanged": True,
             "GPU_stress_executed": False, "GPU_parity_executed": False,
             "remaining_gates": "actual ptxas resource gate, CG/CA coherent PTX and store identity audit, CUDA handoff stress, exact all30 first/replay plus sanitizers",
             "producer_sha256": sha(Path(__file__).read_bytes()), "GPU_executed": False}
    (output.parent / "mixed-psi-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--variant", required=True, choices=tuple(PARENTS))
    p.add_argument("--review", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, a.variant, a.review), indent=2))
