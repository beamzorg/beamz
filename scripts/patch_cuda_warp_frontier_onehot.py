"""Restrict the new warp frontier to existing certified one-hot CPML classes.

The CTA-uniform default branch retains the original512-thread aliased body
with canonical spatial E coefficient lookup. The launch allocates the maximum
of both bodies' shared footprints. This isolates whether the new entry's
genericmulti-axis branch caused the ptxas128-register cliff.
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

from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES, once
from scripts.patch_cuda_warp_frontier_512 import transformed as frontier
from scripts.patch_cuda_warp_frontier_phase_points import transformed as points


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(original, *, warp_y=4, point_clocks=False, fence=False):
    if point_clocks:
        code, header, proof = points(original, warp_y=warp_y, fence=fence)
    else:
        if fence:
            raise ValueError("Statefence requires literalpoint clocks")
        code, header, proof = frontier(original, warp_y=warp_y)
    code = once(code,
        "  default:UniformBody<Depth,1,false,true,1,true>(p,tiles);return;",
        "  // Multi-axis/no-axis/native-wall defaults retain the original512-thread\n  // canonical spatial coefficient decoder; the certificate is not weakened.\n  default:AliasedFamilyBody<Depth,1,false,true,1,false>(p,tiles);return;")
    code = once(code, "struct Source {",
        "constexpr size_t WarpFrontierLaunchSharedBytes=PermanentFamilySharedBytes>SharedBytes?PermanentFamilySharedBytes:SharedBytes;\nstruct Source {")
    code = once(code,
        "status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,PermanentFamilySharedBytes)",
        "status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,WarpFrontierLaunchSharedBytes)")
    code = once(code,
        "AsymmetricNoIOWarpFrontier512<Depth><<<uniform_count,512,PermanentFamilySharedBytes,stream>>>",
        "AsymmetricNoIOWarpFrontier512<Depth><<<uniform_count,512,WarpFrontierLaunchSharedBytes,stream>>>")
    proof.update({"onehot_only": True, "default_class": "originalAliasedFamilyBody<Depth,1,false,true,1,false>",
                  "default_threads": 512, "default_source_decode": "canonical spatial E, no commonE substitution",
                  "launch_shared_bytes": max(proof["shared_bytes"], 51748),
                  "new_frontier_axes": [1, 2, 4], "selector": "p.flags[tiles[blockIdx.x]]&14 in{2,4,8}",
                  "CTAuniform_before_all_body_barriers": True,
                  "admission_and_host_compaction_unchanged": True})
    return code, header, proof


def patch_copy(source, output, *, warp_y=4, point_clocks=False, fence=False):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    for name, expected in PARENT_HASHES.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 parent required: {name}")
    code, header, proof = transformed((source / "src/asymmetric_native.cuh").read_text(),
                                     warp_y=warp_y, point_clocks=point_clocks, fence=fence)
    proof.update({"parent_hashes": PARENT_HASHES, "patcher_sha256": sha(Path(__file__).read_bytes()),
                  "result_source_sha256": sha(code.encode()), "gpu_executed": False})
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(code)
    (output / "src/warp_frontier_layout_generated.cuh").write_text(header)
    (output.parent / "warp-frontier-onehot-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--warp-y", type=int, choices=(2, 4, 8), default=4)
    p.add_argument("--point-clocks", action="store_true")
    p.add_argument("--fence", action="store_true")
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, warp_y=a.warp_y, point_clocks=a.point_clocks, fence=a.fence)))


if __name__ == "__main__":
    main()
