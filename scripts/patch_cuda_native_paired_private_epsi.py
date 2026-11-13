"""Copy-only paired NoIO field storage with8 plain private Epsi scalars.

The old-interleaved private-cache resource failure is not instantiated. Only
new NoIO entry uses the proven contiguous4+4 paired field schedule, with the
same canonical recurrence and first-owned/high/final cache lifetime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENTS = {
    "asymmetric_native.cuh": "ea1ff269eeaf57faba9e36bc2bb0a39f16f63c27b7e28833a6b9213fb60daaed",
    "native_extension.cc": "de99df0886104119f25613c728b696c5c57d62e17ba9bcb6816ffcfb47c55071",
    "native_moving.h": "00939dbe87d158f357cbac605e17c3f3ede809b66bf670b35bd17af133d7e826",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
    "runtime_shared_epsi_state_fact.h": "cbfb91ca46f3ef8c72885f34c7e23b7a24cad67064fbcbaf2e946cfb01840d66",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one paired-cache anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for n, d in PARENTS.items():
        if sha(parent[n]) != d:
            raise ValueError(f"Exact canonical private8 resource parent required:{n}")
    old = parent["asymmetric_native.cuh"].decode()
    c = old
    entry = """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOPrivateEPsi1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 UniformBody<Depth,1,false,true,1,true,true>(p,tiles);
}

"""
    c = once(c, entry, "")
    begin = c.index(
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1>\n__device__ __forceinline__ void SourcePendingPairedBody"
    )
    end = c.index("constexpr size_t SourcePendingSharedBytes=", begin)
    body = c[begin:end]
    original_body = body
    body = once(
        body,
        "int BlockT=0,int SourceMask=-1>",
        "int BlockT=0,int SourceMask=-1,bool CacheE=false>",
    )
    body = once(
        body,
        " float h[3][Slots],pending_ex0[YGroups]",
        """ static_assert(!CacheE||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE&&Slots==4),"Exact paired private Epsi class");
 PrivateEPsiBank<CacheE,Slots> epsi;
 float h[3][Slots],pending_ex0[YGroups]""",
    )
    anchor = "    if(zs!=hz&&ys!=hy&&sx!=hx){\n     const auto read="
    assert body.count(anchor) == 2
    wrapper = """    if(zs!=hz&&ys!=hy&&sx!=hx){
     auto point_env=[&] __device__(){
      if constexpr(CacheE){const unsigned axes=flags&14u;
       return SharedEPsiPoint<decltype(env)>(env,epsi.a[si],epsi.b[si],n==window[0]||dz==1||dy==1||dx==1,
        n==Depth||dz==BZ||dy==BY||dx==BX,axes==2u||axes==4u||axes==8u);
      }else return env;
     }();
     const auto read="""
    body = body.replace(anchor, wrapper)
    for k in range(3):
        a = f"AdvanceNativeCell<1,{k},false>(env,read,n,z,y,x)"
        assert body.count(a) == 2
        body = body.replace(
            a, f"AdvanceNativeCell<1,{k},false>(point_env,read,n,z,y,x)"
        )
    # Exact source inverse binds all arithmetic and barrier/storage sites.
    inverse = body.replace(wrapper, anchor)
    for k in range(3):
        inverse = inverse.replace(
            f"AdvanceNativeCell<1,{k},false>(point_env,read,n,z,y,x)",
            f"AdvanceNativeCell<1,{k},false>(env,read,n,z,y,x)",
        )
    inverse = once(
        inverse,
        "int BlockT=0,int SourceMask=-1,bool CacheE=false>",
        "int BlockT=0,int SourceMask=-1>",
    )
    inverse = once(
        inverse,
        ' static_assert(!CacheE||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE&&Slots==4),"Exact paired private Epsi class");\n PrivateEPsiBank<CacheE,Slots> epsi;\n',
        "",
    )
    assert inverse == original_body
    c = c[:begin] + body + c[end:]
    newentry = """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOPairedPrivateEPsi1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 SourcePendingPairedBody<Depth,1,false,true,1,true,1024,-1,true>(p,tiles);
}

"""
    c = once(c, "inline bool MakeSource(", newentry + "inline bool MakeSource(")
    c = c.replace(
        "AsymmetricNoIOPrivateEPsi1024<Depth>",
        "AsymmetricNoIOPairedPrivateEPsi1024<Depth>",
    )
    a = c.index("  if(shared_epsi){")
    b = c.index("  const auto kernel=AsymmetricNoIOUniform1024", a)
    r = c[a:b]
    assert r.count("SharedEPsiBytes") == 2
    c = c[:a] + r.replace("SharedEPsiBytes", "SourcePendingSharedBytes") + c[b:]
    e = parent["native_extension.cc"].decode()
    e = once(
        e,
        'result["moving_shared_epsi_bytes"]=51748;',
        'result["moving_shared_epsi_bytes"]=58680;result["moving_private_epsi_paired_storage"]=true;result["moving_private_epsi_barriers"]=5;',
    )
    e = once(
        e,
        'module.attr("moving_shared_epsi_shared_bytes")=51748;',
        'module.attr("moving_shared_epsi_shared_bytes")=58680;module.attr("moving_private_epsi_paired_storage")=true;module.attr("moving_private_epsi_barriers")=5;',
    )
    files = dict(parent)
    files["asymmetric_native.cuh"] = c.encode()
    files["native_extension.cc"] = e.encode()
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src, out = a.parent.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or out in (src, canonical)
        or src in out.parents
        or out in src.parents
        or canonical in out.parents
    ):
        p.error("New disjoint private output required")
    files = transformed({n: (src / "src" / n).read_bytes() for n in PARENTS})
    shutil.copytree(src, out)
    for n, b in files.items():
        (out / "src" / n).write_bytes(b)
    proof = {
        "parents": PARENTS,
        "result_hashes": {n: sha(b) for n, b in files.items()},
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "mapping": "contiguous zs=(tid/512)*4+si,si0..3,independentprivate2colors",
        "owned_psi_lifetime": "windowstart or anyrel1 refresh; highunion or depth final export; otheraxesglobal",
        "paired_body_exact_inverse": True,
        "same_canonical_cell_math_and_all_5_barriers": True,
        "old_source_cache_default_false": True,
        "old_spilled_UniformBody_cache_true_uninstantiated": True,
        "shared_bytes": 58680,
        "expected_kernels": 60,
        "gpu_executed": False,
        "gate": "Reject any stack/local/spill before GPU",
    }
    (out.parent / "paired-private-epsi-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
