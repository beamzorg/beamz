"""512-thread private-psi frontier with eight Z slots and32 history floats.

The original uncached NoIO entry remains1024 threads. The new cached entry is
512 threads in every branch, including its multi-axis uncached fallback. Only
single-axis branches allocate history; one-hot facts, alias admission, exact
profiles, scalar arithmetic and first/exit handoffs match the1024 experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_phase_state_fact import generate_guard
from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES, once
from scripts.patch_cuda_traced_lifetime_private_psi import HOST_PARENT, transform as base_transform


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform(consumer, host, header, *, scope=2, debug_hits=False, enabled=True):
    text, host, header, base_proof = base_transform(consumer, host, header, scope=scope, debug_hits=debug_hits, enabled=enabled)
    begin = text.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachePsi=false>\n__device__ __forceinline__ void PermanentFamilyBody")
    end = text.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", begin)
    body = text[begin:end]
    original_body = body
    edits = [
        ("constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;",
         "constexpr int T=CachePsi?512:((Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512),Slots=PAD/T,ZStep=T/512;"),
        ('static_assert(T==1024,"Permanent-family frontier requires admitted1024-thread owner mapping");',
         'static_assert(T==1024||(CachePsi&&T==512),"Permanent-family frontier requires1024uncached or512cached mapping");'),
        ('static_assert(!CachePsi||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE&&Slots==4&&(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4)),"Exact single-axis NoIO private-psi class");',
         'static_assert(!CachePsi||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE&&Slots==8&&(ActiveAxes==-1||ActiveAxes==1||ActiveAxes==2||ActiveAxes==4)),"Exact512-thread NoIO private-psi class");\n'
         ' constexpr bool CacheState=CachePsi&&(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4);'),
        ("PrivatePsiHistory<CachePsi,Slots>history;", "PrivatePsiHistory<CacheState,Slots>history;"),
    ]
    for old, new in edits:
        body = once(body, old, new)
    if body.count("if constexpr(CachePsi)") != 2:
        raise ValueError("Expected exactly H/E point history selectors")
    body = body.replace("if constexpr(CachePsi)", "if constexpr(CacheState)")
    inverse = body.replace("if constexpr(CacheState)", "if constexpr(CachePsi)")
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    if inverse != original_body:
        raise ValueError("512owner transformation exact inverse failed")
    text = text[:begin] + body + text[end:]
    text = once(text, "__global__ __launch_bounds__(1024,1) void AsymmetricNoIOPrivatePsi1024(",
                "__global__ __launch_bounds__(512,1) void AsymmetricNoIOPrivatePsi1024(")
    text = once(text,
        "default:PermanentFamilyBody<Depth,1,false,true,1,true>(p,tiles);return;",
        "default:PermanentFamilyBody<Depth,1,false,true,1,true,-1,true>(p,tiles);return;")
    text = once(text,
        "AsymmetricNoIOPrivatePsi1024<Depth><<<uniform_count,1024,PermanentFamilySharedBytes,stream>>>",
        "AsymmetricNoIOPrivatePsi1024<Depth><<<uniform_count,512,PermanentFamilySharedBytes,stream>>>")
    if text.count("AsymmetricNoIOPrivatePsi1024") != 3:
        raise ValueError("Unbound cached kernel declaration/launch")
    text = text.replace("AsymmetricNoIOPrivatePsi1024", "AsymmetricNoIOPrivatePsi512")
    host = once(host, 'result["moving_private_psi_floats_per_thread"]=16;',
                'result["moving_private_psi_floats_per_thread"]=32;result["moving_private_psi_threads"]=512;result["moving_private_psi_point_slots"]=8;')
    host = once(host, 'module.attr("moving_private_psi_floats_per_thread")=16;',
                'module.attr("moving_private_psi_floats_per_thread")=32;module.attr("moving_private_psi_threads")=512;module.attr("moving_private_psi_point_slots")=8;')
    return text, host, header, {"base_private_psi_proof": base_proof, "cached_threads": 512,
        "cached_point_slots_per_thread": 8, "private_psi_floats_per_thread": 32,
        "total_CTA_history_floats": 16384, "uncached_original_entry_threads": 1024,
        "cached_entry_multiaxis_fallback": "512threads; ActiveAxes=-1 CacheState=false, ordinary globalpsi",
        "all_other_original_uncached_NoIO_and_source_entries_unchanged": True,
        "shared_bytes": 100904, "loop_barriers": 2,
        "block_register_capacity": "512threads allow128registers/thread within64Kregisterfile; actualcompilerfitstillrequired",
        "exact_body_source_inverse": True}


def patch_copy(source, output, *, scope=2, debug_hits=False, enabled=True):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint private source required")
    parents = dict(PARENT_HASHES, **{"native_extension.cc": HOST_PARENT})
    for name, expected in parents.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 source required: {name}")
    src = source / "src"
    text, host, header, proof = transform((src / "asymmetric_native.cuh").read_text(),
        (src / "native_extension.cc").read_text(), (src / "native_moving.h").read_text(),
        scope=scope, debug_hits=debug_hits, enabled=enabled)
    files = {"asymmetric_native.cuh": text, "native_extension.cc": host,
             "native_moving.h": header, "runtime_private_psi_state_fact.h": generate_guard()["source"]}
    proof.update({"parent_hashes": parents, "result_hashes": {name: sha(value.encode()) for name, value in files.items()},
                  "producer_sha256": sha(Path(__file__).read_bytes()), "gpu_executed": False,
                  "requires": ["512ownerandsharedfielddataflowproof", "packedCPMLhandoffproofunchanged",
                               "zero stack/local/spills64kernelcompile", "all30exact + actualretainedhitdiagnostics",
                               "memcheck/synccheck/racecheck", "matched full-workload benchmark"]})
    shutil.copytree(source, output)
    for name, value in files.items():
        (output / "src" / name).write_text(value)
    (output.parent / "private-psi512-history-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=int, default=2)
    parser.add_argument("--debug-hits", action="store_true")
    parser.add_argument("--selector-off", action="store_true")
    args = parser.parse_args()
    print(json.dumps(patch_copy(args.source, args.output, scope=args.scope,
                               debug_hits=args.debug_hits, enabled=not args.selector_off)))


if __name__ == "__main__":
    main()
