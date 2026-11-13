"""Choose resource budgets for the partial-Ey larger-cache experiment.

Uniform byte/isotropic bulk and certified NoIO can use1024threads. Packed bulk
uses512; source and nonisotropic boundary fallbacks use256. This producer adds
the uniform bulk entry; a separate proven host partition must dispatch it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "0b645c46df6ab4d71956458f636cef44b5a8bf355a2a96acb10667181873d081"
OLD_BODY = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>\n__device__ __forceinline__ void UniformBody"
NEW_BODY = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0>\n__device__ __forceinline__ void UniformBody"
OLD_T = (
    " constexpr int T=(Fast||(Layout==1&&Policy==1&&IsotropicH&&UniformE))?1024:512;"
)
NEW_T = " constexpr int FallbackT=((Policy==3||Policy==4)||(!Fast&&!IsotropicH))?256:512;\n constexpr int T=BlockT?BlockT:((Layout==1&&Policy==1&&IsotropicH&&UniformE)?1024:FallbackT);"
BOUND = "__launch_bounds__((Fast?1024:512),1)"
NEW_BOUND = (
    "__launch_bounds__((((Policy==3||Policy==4)||(!Fast&&!IsotropicH))?256:512),1)"
)
ENTRY = """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricPartialUniformBulk1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 UniformBody<Depth,1,true,true,0,true,1024>(p,tiles);
}

"""


def sha(x):
    return hashlib.sha256(x).hexdigest()


def once(t, a, b):
    if t.count(a) != 1:
        raise ValueError(f"Expected one thread-budget anchor:{a!r}")
    return t.replace(a, b, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src = a.source.resolve()
    out = a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or out == canonical
        or src == out
        or src in out.parents
        or out in src.parents
    ):
        p.error("New disjoint private source required")
    c = (src / "src/asymmetric_native.cuh").read_text()
    if sha(c.encode()) != PARENT:
        raise ValueError("Exact frozen8x32 partial-Ey source required")
    changed = once(c, OLD_BODY, NEW_BODY)
    changed = once(changed, OLD_T, NEW_T)
    changed = once(changed, BOUND, NEW_BOUND)
    old = " Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<count,(Fast?1024:512),SharedBytes,stream>>>(p,ids);"
    new = " Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<count,(((Policy==3||Policy==4)||(!Fast&&!IsotropicH))?256:512),SharedBytes,stream>>>(p,ids);"
    changed = once(changed, old, new)
    changed = once(
        changed, "inline bool MakeSource(", ENTRY + "inline bool MakeSource("
    )
    inverse = (
        changed.replace(ENTRY, "", 1)
        .replace(new, old, 1)
        .replace(NEW_BOUND, BOUND, 1)
        .replace(NEW_T, OLD_T, 1)
        .replace(NEW_BODY, OLD_BODY, 1)
    )
    assert inverse == c
    shutil.copytree(src, out)
    (out / "src/asymmetric_native.cuh").write_text(changed)
    proof = {
        "parent_consumer_sha256": PARENT,
        "consumer_sha256": sha(changed.encode()),
        "exact_body_inverse": True,
        "same_point_arithmetic_and_shared_schedule": True,
        "threads": {
            "uniform_bulk": 1024,
            "certified_noio": 1024,
            "packed_bulk": 512,
            "isotropic_general": 512,
            "sources_or_nonisotropic_boundaries": 256,
        },
        "uniform_bulk_entry": "AsymmetricPartialUniformBulk1024<Depth>",
        "uniform_body_signature": "UniformBody<Depth,1,true,true,0,true,1024>",
        "scope": "Resource class experiment; host bulk partition/dispatch not added by this stage",
    }
    (out.parent / "thread-classes-source-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
