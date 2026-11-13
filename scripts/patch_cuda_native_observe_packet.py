"""Copy-only deferred six-field raw-observation packet for policy2 on9f6a."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_HASHES = {
    "asymmetric_native.cuh": "f0657105ea5a57219d48a7a067d1ef53f6246676579e722ffd0670a14d3d8f49",
    "native_extension.cc": "8f122cdbb850374f44c8e59c3282274b744ebff0ef5a907540a245a53a730bcf",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}

PACKET = r"""
// Policy2 only: H private values survive E unchanged. Raw consumers run after
// ALL macro rank/policy joins, so store all six post-source samples together.
template<int Field>
__device__ __forceinline__ void StorePacketComponent(
 const beamz::cuda::deep::Observation&o,const int32_t*leaf,size_t row,unsigned support,float value){
 if(!(support&(1u<<Field)))return;
 const int raw=__ldg(leaf+Field);
 if(raw>=0)o.values[row+raw]=value;
}
__device__ __forceinline__ void ObservePacket(const Parameters&p,int version,int z,int y,int x,
 float hx,float hy,float hz,float ex,float ey,float ez){
 const auto&o=p.observation;if(!o.values)return;
 if(z<0||y<0||x<0||z>=o.mask_nz||y>=o.mask_ny||x>=o.mask_nx)return;
 const unsigned support=NativeSupportMask(p,z,y,x);if(!support)return;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
 const int tile=(z*p.observation_ny+(y>>3))*p.observation_nx+(x>>4);
 const int slot=__ldg(p.observation_tiles+tile);if(slot<0)return;
 const size_t base=size_t(slot)*128*6+(size_t(y&7)*16+(x&15))*6;
 const int32_t*leaf=p.observation_leaf+base;
 const size_t row=size_t(version-1)*o.key_count;
 StorePacketComponent<0>(o,leaf,row,support,hx);
 StorePacketComponent<1>(o,leaf,row,support,hy);
 StorePacketComponent<2>(o,leaf,row,support,hz);
 StorePacketComponent<3>(o,leaf,row,support,ex);
 StorePacketComponent<4>(o,leaf,row,support,ey);
 StorePacketComponent<5>(o,leaf,row,support,ez);
#else
 // Preserve the original nonsparse path; selected builds use sparse leaves.
 if(support&1)Observe(p,0,version,z,y,x,hx);
 if(support&2)Observe(p,1,version,z,y,x,hy);
 if(support&4)Observe(p,2,version,z,y,x,hz);
 if(support&8)Observe(p,3,version,z,y,x,ex);
 if(support&16)Observe(p,4,version,z,y,x,ey);
 if(support&32)Observe(p,5,version,z,y,x,ez);
#endif
}
"""
PUBLISH_ANCHOR = "const int dz=(zs-hz)&7,z=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);\n   #pragma unroll"
PUBLISH_REPLACEMENT = "const int dz=(zs-hz)&7,z=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);\n   if constexpr(Policy==2)ObservePacket(p,n,z,y,x,h[0][j],h[1][j],h[2][j],e[0][j],e[1][j],e[2][j]);\n   #pragma unroll"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact observation-packet anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if digest(parent[name]) != expected:
            raise ValueError(f"Exact9f6a observer parent required: {name}")
    files = {name: value.decode() for name, value in parent.items()}
    c = files["asymmetric_native.cuh"]
    old = "if constexpr(HasObserve)if(flags&beamz::cuda::moving::kMayObserve)Observe(p,field,version,z,y,x,value);"
    new = "if constexpr(HasObserve&&Policy!=2)if(flags&beamz::cuda::moving::kMayObserve)Observe(p,field,version,z,y,x,value);"
    c = once(c, old, new)
    c = once(
        c,
        "struct FieldPacket {float hx,hy,hz,ex,ey,ez;};",
        PACKET + "\nstruct FieldPacket {float hx,hy,hz,ex,ey,ez;};",
    )
    c = once(c, PUBLISH_ANCHOR, PUBLISH_REPLACEMENT)
    inverse = once(c, PUBLISH_REPLACEMENT, PUBLISH_ANCHOR)
    inverse = once(
        inverse,
        PACKET + "\nstruct FieldPacket {float hx,hy,hz,ex,ey,ez;};",
        "struct FieldPacket {float hx,hy,hz,ex,ey,ez;};",
    )
    inverse = once(inverse, new, old)
    if inverse.encode() != parent["asymmetric_native.cuh"]:
        raise ValueError("Exact inverse-source binding failed")
    files["asymmetric_native.cuh"] = c
    e = files["native_extension.cc"]
    e = once(
        e,
        '  module.attr("moving_observer_uniform_1024")=true;',
        '  module.attr("moving_observe_packet_policy2")=true;\n  module.attr("moving_observer_uniform_1024")=true;',
    )
    files["native_extension.cc"] = e
    return {name: value.encode() for name, value in files.items()}


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        output.exists()
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be a new disjoint research copy")
    if output == canonical or canonical in output.parents:
        raise ValueError("Canonical source must remain unchanged")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {
        "parent_module_sha256": "9f6ade0cfb908fb1a71c883bfd2b106321a0401a2b8ec1bc87472047a7cb22ec",
        "parent_hashes": PARENT_HASHES,
        "result_hashes": {name: digest(value) for name, value in files.items()},
        "patcher_sha256": digest(Path(__file__).read_bytes()),
        "exact_inverse_source": True,
        "scope": "ONLY policy2 defers raw stores until postB4 all-owned publisher; explicitn; canonicalcell/math/source/CPML/rankjoins unchanged; proof/GPUgates required",
    }
    output.with_name(output.name + "-observe-packet-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patch_copy(args.source, args.output)


if __name__ == "__main__":
    main()
