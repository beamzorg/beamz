"""Replace certified UniformBody X-neighbor shared reads with warp exchanges.

The admitted point graph needs two +X E neighbors in H and two -X H neighbors
in E. X is exactly CUDA lane ID in the selected schedule. Every collective is
outside owned/support predicates and executes with all 32 lanes, including the
ghost lane. The phase-entry private values equal the published shared frame.
All stores, four barriers, imports, CPML, sources and FP arithmetic stay intact.

This is a copied-source experiment, not a production backend or speed claim.
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

from scripts.cuda_stencil_trace import reads, trace_yee_step

PARENTS = {
    "asymmetric_native.cuh": "f082980ad0687ef165c7807b415e87cbc1003edfdbd2acf7a4ee018dae4d4547",
    "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27",
    "native_moving.h": "d3d1abaca2ca066b75295df35ec26da2c1df53c75bdb9f140b067fea09288b3c",
    "native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}
HELPER = r"""// Full-warp point-owner exchange; executes before all owned/Valid predicates.
__device__ __forceinline__ float TracedWarpX(float value,int lane){
 uint32_t result;
 asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;"
              : "=r"(result) : "r"(__float_as_uint(value)), "r"(lane));
 return __uint_as_float(result);
}
"""
BODY_BEGIN = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody"
BODY_END = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__"
H_POINT = """   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);
    if(zs!=hz&&sy!=hy&&sx!=hx){"""
H_NEW = """   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);
    const float warp_ey_x=TracedWarpX(e[1][j],(sx+1)&31);
    const float warp_ez_x=TracedWarpX(e[2][j],(sx+1)&31);
    if(zs!=hz&&sy!=hy&&sx!=hx){"""
H_READ = "if(!dz&&!ddy&&!ddx)return e[c][j];return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];"
H_REPLACEMENT = "if(!dz&&!ddy&&!ddx)return e[c][j];if(ddx)return c==1?warp_ey_x:warp_ez_x;return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];"
E_POINT = """   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);
    if(zs!=ehz&&sy!=ehy&&sx!=ehx){"""
E_NEW = """   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);
    const float warp_hy_x=TracedWarpX(h[1][j],(sx+31)&31);
    const float warp_hz_x=TracedWarpX(h[2][j],(sx+31)&31);
    if(zs!=ehz&&sy!=ehy&&sx!=ehx){"""
E_READ = "if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];"
E_REPLACEMENT = "if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];if(ddx)return field==1?warp_hy_x:warp_hz_x;return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Required exact schedule anchor changed")
    return text.replace(old, new, 1)


def traced_x_reads():
    """Bind neighbor routing to actual ordered Yee trace, not hand-picked loads."""
    result = {"H": set(), "E": set()}
    for update in trace_yee_step():
        for read in reads(update.value):
            if read.offset[2]:
                result[update.output.field[0]].add((read.field, read.offset))
    expected = {
        "H": {("Ey", (0, 0, 1)), ("Ez", (0, 0, 1))},
        "E": {("Hy", (0, 0, -1)), ("Hz", (0, 0, -1))},
    }
    if result != expected:
        raise ValueError("Actual ordered cell graph no longer has the admitted X reads")
    return {k: sorted(v) for k, v in result.items()}


def prove_ownership(*, threads=(512, 1024), lane_delta=None, predicate_before_collective=False):
    """Exhaustively compare original frame addresses to register-owner addresses.

    The exact source inverse below retains all phase-entry frame stores, so
    each frame word has the bits of the owning lane/slot's private value. IDs
    describe values without performing/reassociating any floating arithmetic.
    The negative options deliberately corrupt ownership or participation.
    """
    graph_reads = traced_x_reads()
    checked = 0
    for t in threads:
        if t not in (512, 1024):
            raise ValueError("Only the actual UniformBody thread counts are admitted")
        slots, zstep = 4096 // t, t // 512
        for step in range(1, 17):
            hz, hy, hx = (-step - 1) & 7, (-step - 1) & 15, (-step - 1) & 31
            for tid in range(t):
                sx, sy, zbase = tid & 31, (tid // 32) & 15, tid // 512
                warp_base = tid & ~31
                for j in range(slots):
                    zs = zbase + j * zstep
                    for phase, delta in (("H", 1), ("E", -1)):
                        # The whole warp must execute even when only some own points.
                        if predicate_before_collective and zs != hz and sy != hy:
                            members = {lane for lane in range(32) if lane != hx}
                            if members != set(range(32)):
                                raise ValueError("Collective has an inactive ghost lane")
                        if zs == hz or sy == hy or sx == hx:
                            continue
                        target_x = (sx + delta) & 31
                        source_tid = warp_base + ((sx + (delta if lane_delta is None else lane_delta)) & 31)
                        source_z = source_tid // 512 + j * zstep
                        source_y, source_x = (source_tid // 32) & 15, source_tid & 31
                        if (source_z, source_y, source_x) != (zs, sy, target_x):
                            raise ValueError("Warp exchange chooses a different field owner")
                        for field, offset in graph_reads[phase]:
                            assert offset[2] == delta
                            # Field identity remains part of the payload provenance.
                            original = (field, zs, sy, target_x)
                            candidate = (field, source_z, source_y, source_x)
                            if original != candidate:
                                raise ValueError("Warp exchange changes a field payload")
                            checked += 1
    return {"actual_traced_x_reads": graph_reads, "checked_payload_owners": checked,
            "threads": list(threads), "steps": 16, "all32_lanes_before_predicates": True,
            "original_frame_and_register_phase_entry_bits_equal": True}


SCOPES = {
    "all": "true",
    "noio": "Layout==1&&Policy==1&&IsotropicH&&UniformE",
    "bulk": "Fast&&Layout==1&&IsotropicH",
    "noio-bulk": "(Fast&&Layout==1&&IsotropicH)||(Layout==1&&Policy==1&&IsotropicH&&UniformE)",
}


def scope_enabled(scope, *, layout, fast, isotropic_h, policy, uniform_e):
    if scope not in SCOPES:
        raise ValueError("Unknown admitted warp scope")
    return (scope == "all"
            or scope in ("bulk", "noio-bulk") and fast and layout == 1 and isotropic_h
            or scope in ("noio", "noio-bulk") and layout == 1 and policy == 1 and isotropic_h and uniform_e)


def transform(parent, mode="both", scope="all"):
    if mode not in ("h", "e", "both"):
        raise ValueError("Mode must be h, e or both")
    if scope not in SCOPES:
        raise ValueError("Unknown admitted warp scope")
    for name, expected in PARENTS.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact selected52865 parent source required: {name}")
    traced_x_reads()
    old = parent["asymmetric_native.cuh"].decode()
    begin, end = old.index(BODY_BEGIN), old.index(BODY_END)
    body = old[begin:end]
    pairs = []
    h_new, e_new, h_read, e_read = H_NEW, E_NEW, H_REPLACEMENT, E_REPLACEMENT
    if scope != "all":
        pairs += [(" constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);",
                   " constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n"
                   f" constexpr bool TracedWarpXEnabled={SCOPES[scope]};")]
        h_new = H_NEW.replace(
            "    const float warp_ey_x=TracedWarpX(e[1][j],(sx+1)&31);\n    const float warp_ez_x=TracedWarpX(e[2][j],(sx+1)&31);",
            "    float warp_ey_x=0.f,warp_ez_x=0.f;\n    if constexpr(TracedWarpXEnabled){warp_ey_x=TracedWarpX(e[1][j],(sx+1)&31);warp_ez_x=TracedWarpX(e[2][j],(sx+1)&31);}")
        e_new = E_NEW.replace(
            "    const float warp_hy_x=TracedWarpX(h[1][j],(sx+31)&31);\n    const float warp_hz_x=TracedWarpX(h[2][j],(sx+31)&31);",
            "    float warp_hy_x=0.f,warp_hz_x=0.f;\n    if constexpr(TracedWarpXEnabled){warp_hy_x=TracedWarpX(h[1][j],(sx+31)&31);warp_hz_x=TracedWarpX(h[2][j],(sx+31)&31);}")
        h_read = H_REPLACEMENT.replace("if(ddx)return c==1?warp_ey_x:warp_ez_x;",
                                       "if constexpr(TracedWarpXEnabled){if(ddx)return c==1?warp_ey_x:warp_ez_x;}")
        e_read = E_REPLACEMENT.replace("if(ddx)return field==1?warp_hy_x:warp_hz_x;",
                                       "if constexpr(TracedWarpXEnabled){if(ddx)return field==1?warp_hy_x:warp_hz_x;}")
    if mode in ("h", "both"):
        pairs += [(H_POINT, h_new), (H_READ, h_read)]
    if mode in ("e", "both"):
        pairs += [(E_POINT, e_new), (E_READ, e_read)]
    changed_body = body
    for a, b in pairs:
        changed_body = once(changed_body, a, b)
    inverse = changed_body
    for a, b in reversed(pairs):
        inverse = once(inverse, b, a)
    if inverse != body:
        raise ValueError("Exact source inverse failed")
    if body.count("__syncthreads()") != changed_body.count("__syncthreads()"):
        raise ValueError("CTA barriers changed")
    changed = old[:begin] + HELPER + changed_body + old[end:]
    restored = changed.replace(HELPER, "", 1)
    for a, b in reversed(pairs):
        restored = once(restored, b, a)
    if restored != old:
        raise ValueError("Unrelated source changed")
    result = dict(parent)
    result["asymmetric_native.cuh"] = changed.encode()
    return result


def patch_copy(source, output, mode="both", scope="all"):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint private source required")
    parent = {n: (source / "src" / n).read_bytes() for n in PARENTS}
    result = transform(parent, mode, scope)
    proof = prove_ownership()
    proof.update({"parent_hashes": PARENTS, "result_hashes": {n: sha(v) for n, v in result.items()},
                  "mode": mode, "scope": scope, "scope_expression": SCOPES[scope], "exact_source_inverse": True,
                  "unchanged": ["ordered FP32 point graph", "CPML state/profile addressing",
                                "all field/trail stores", "all barriers", "host admission and graph identity",
                                "source callbacks", "observation and DFT order", "launches and geometry"],
                  "requires": ["zero-stack/local/spill exact-context compile", "all30 exact GPU comparisons",
                               "memcheck/synccheck/racecheck", "matched complete-workload benchmark"],
                  "gpu_executed": False})
    shutil.copytree(source, output)
    for name, data in result.items():
        (output / "src" / name).write_bytes(data)
    (output.parent / "traced-warp-x-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("h", "e", "both"), default="both")
    parser.add_argument("--scope", choices=tuple(SCOPES), default="all")
    args = parser.parse_args()
    print(json.dumps(patch_copy(args.source, args.output, args.mode, args.scope)))


if __name__ == "__main__":
    main()
