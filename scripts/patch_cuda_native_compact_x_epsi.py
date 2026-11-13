"""Copy-only compact X-axis auxiliary bank on canonical shared-Epsi v2.

Addressing occurs only inside State, after the canonical cell has established
Valid/LocalActive. No reference to an inactive packed coordinate is formed.
The same NoIO grid retains global recurrences for every non-X-axis tile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_MODULE = "4af3cc2e6e48f55a9bcab6dda8c30c7f11a1c5e5a5dcc19cdfdef5af486fdc16"
PARENTS = {
    "asymmetric_native.cuh": "33a5928617f589bf5ff4f68ba8f4ec179108cfb9fada85f16eb9e519850cb9d1",
    "native_extension.cc": "2d16eec337f62454b2d6ec076e903592dc656469541e3ef4ed6927da12172162",
    "native_moving.h": "082576ce0fd5774a0ae236002bfe35d799fc65429c0ded60db8d676721791e7d",
    "runtime_shared_epsi_state_fact.h": "cbfb91ca46f3ef8c72885f34c7e23b7a24cad67064fbcbaf2e946cfb01840d66",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}
POINT = r'''template<class Base> struct SharedEPsiPoint : Base {
 volatile float*bank;int zs,ys;bool first,flush,enabled;
 __device__ __forceinline__ SharedEPsiPoint(Base&base,volatile float*storage,int zslot,int yslot,bool fresh,bool publish,bool admit)
 :Base(base),bank(storage),zs(zslot),ys(yslot),first(fresh),flush(publish),enabled(admit){}
 __device__ __forceinline__ volatile float& State(int slot,int z,int y,int x)const{
  // Reached only for E X terms3/4 after canonical Valid+LocalActive.
  // No packed/index/reference computation occurs during adapter construction.
  const int color=slot%6==4?1:0;
  const int packed=this->Packed(slot,z,y,x);
  return bank[((color*8+zs)*16+ys)*12+packed%12];
 }
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if(slot<6||!enabled||first)return Base::LoadLocal(slot,z,y,x);
  return State(slot,z,y,x);
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  if(slot<6||!enabled){Base::StoreLocal(slot,z,y,x,value);return;}
  State(slot,z,y,x)=value;if(flush)Base::StoreLocal(slot,z,y,x,value);
 }
};
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one compact-X anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, expected in PARENTS.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact canonical shared-Epsi v2 parent required: {name}")
    c, e = parent["asymmetric_native.cuh"].decode(), parent["native_extension.cc"].decode()
    start = c.index("template<class Base> struct SharedEPsiPoint")
    end = c.index("constexpr size_t SharedEPsiBytes=", start)
    old_point = c[start:end]
    edits = (
        (old_point, POINT),
        ("SharedEPsiBytes=SharedBytes+8ULL*1024*sizeof(float)", "SharedEPsiBytes=SharedBytes+3072ULL*sizeof(float)"),
        ('SharedEPsiBytes==84516,"Frozen shared-bank allocation"', 'SharedEPsiBytes==64036,"Frozen compact-X bank allocation"'),
        ('"Exact shared Epsi class");', '"Exact shared Epsi class");\n static_assert(!CacheE||(Depth==4||Depth==16),"Compact-X proved depth class");'),
        ("return SharedEPsiPoint<decltype(env)>(env,epsi_bank[j*1024+threadIdx.x],epsi_bank[(Slots+j)*1024+threadIdx.x],",
         "return SharedEPsiPoint<decltype(env)>(env,epsi_bank,zs,sy,"),
        ("axes==2u||axes==4u||axes==8u);", "axes==8u&&p.physical_n[2]>=64);"),
        ("BeamzResearchSharedEPsiEnabled&&w.shared_epsi_state_independent);break;",
         "BeamzResearchSharedEPsiEnabled&&w.shared_epsi_state_independent&&p.physical_n[2]>=64);break;"),
    )
    changed = c
    for old, new in edits:
        changed = once(changed, old, new)
    inverse = changed
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    if inverse != c:
        raise ValueError("Compact-X exact inverse source failed")
    host_edits = (
        ("if(mask!=2&&mask!=4&&mask!=8)continue;const int axis=mask==2?0:mask==4?1:2;",
         "if(mask!=8||minimum[2]<64)continue;const int axis=2;"),
        ("BeamzResearchSharedEPsiEnabled&&moving_workspace_.shared_epsi_state_independent&&moving_uniform_noio_>0;",
         "BeamzResearchSharedEPsiEnabled&&moving_workspace_.shared_epsi_state_independent&&moving_uniform_noio_>0&&base_inputs_[1].dims[2]>=64;"),
        ('result["moving_shared_epsi_bytes"]=84516;', 'result["moving_shared_epsi_bytes"]=64036;'),
        ('module.attr("moving_shared_epsi_shared_bytes")=84516;', 'module.attr("moving_shared_epsi_shared_bytes")=64036;'),
        ('module.attr("moving_shared_epsi")=true;', 'module.attr("moving_shared_epsi_compact_x")=true;module.attr("moving_shared_epsi_minimum_nx")=64;\n  module.attr("moving_shared_epsi")=true;'),
    )
    host = e
    for old, new in host_edits:
        host = once(host, old, new)
    invhost = host
    for old, new in reversed(host_edits):
        invhost = once(invhost, new, old)
    if invhost != e:
        raise ValueError("Compact-X exact host inverse failed")
    files = dict(parent)
    files["asymmetric_native.cuh"], files["native_extension.cc"] = changed.encode(), host.encode()
    # Entire source point body is unchanged. New host selector conjunct is the
    # sole launch-policy change; flags/IDs/sort/slots/workspace/key stay fixed.
    marker = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1>\n__device__ __forceinline__ void SourcePendingPairedBody("
    end_marker = "constexpr size_t SourcePendingSharedBytes="
    assert c[c.index(marker):c.index(end_marker)] == changed[changed.index(marker):changed.index(end_marker)]
    return files


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical copy required")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENTS}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {"parent_module_sha256": PARENT_MODULE, "parent_hashes": PARENTS,
             "result_hashes": {n: sha(d) for n, d in files.items()}, "patcher_sha256": sha(Path(__file__).read_bytes()),
             "point_sha256": sha(POINT.encode()), "exact_consumer_and_host_inverse": True,
             "mapping": "((color*8+zs)*16+sy)*12+Packed(slot,z,y,x)%12, computed ONLY inside State after active X canonical guard",
             "admission": "existing NoIO uniform/isotropic/byte class+prepared disjoint state; physicalNx>=64; Depth4/16; oneaxisflags8 cached, allotheraxes global",
             "allocation_bytes": 64036, "same_noio_grid": True, "source5_body_unchanged": True,
             "canonical_math_header_unchanged": True, "workspace_hidden_selector_key_unchanged": True,
             "requires": ["bound compact-X CPU lifecycle/adapter oracle", "exact-context resource gate", "strict v12-matched hardware/sanitizer/performance gates"], "gpu_executed": False}
    (output.parent / "compact-x-source-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output)))


if __name__ == "__main__":
    main()
