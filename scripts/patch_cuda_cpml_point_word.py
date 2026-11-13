"""One stored uint32 point context: packed coordinate plus active bit.

Copy-only derivative of the reviewed cubic point-context experiment. Addresses
are rematerialized at each needed access; no cached base/delta/boolean members
extend their live ranges through the three component updates.
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

from scripts.patch_cuda_cpml_point_context import helper as original_helper


PARENT_CONSUMER = "9e88aa773f564514494c92af6773e17beafc7e9c51959fc6bfc45945d66a1301"


def helper():
    text = original_helper()
    start = text.index(" int packed_index,point_base,point_delta;bool point_active;")
    end = text.index(" __device__ __forceinline__ bool Eligible(", start)
    constructor = r''' uint32_t packed_active;
 __device__ __forceinline__ CubicCpmlPointContext(const Base&base,int z,int y,int x):Base(base){
  const int n=this->p.physical_n[0],coordinate=Axis==0?z:Axis==1?y:x,size=n+Phase;
  const int packed=coordinate<12?coordinate:24+coordinate-size;
  packed_active=(static_cast<uint32_t>(packed)<<1)|uint32_t(coordinate<12||coordinate>=size-12);
 }
 __device__ __forceinline__ int PackedIndex()const{return static_cast<int32_t>(packed_active)>>1;}
'''
    text = text[:start] + constructor + text[end:]
    old_offset = " __device__ __forceinline__ int Offset(int slot)const{return point_base+(slot%6==ExtraTerm?point_delta:0);}"
    new_offset = r''' __device__ __forceinline__ int Offset(int slot,int z,int y,int x)const{
  const int n=this->p.physical_n[0],packed=PackedIndex();
  if constexpr(Axis==0)return packed*(n*(n+1))+y*n+x+(slot%6==ExtraTerm?y:0);
  else if constexpr(Axis==1){const int q=z*24+packed;return q*n+x+(slot%6==ExtraTerm?q:0);}
  else return (z*n+y)*24+packed+(slot%6==ExtraTerm?z*24:0);
 }'''
    if text.count(old_offset) != 1:
        raise ValueError("Reviewed point-address offset helper changed")
    text = text.replace(old_offset, new_offset)
    text = text.replace("return point_active;", "return (packed_active&1u)!=0;")
    text = text.replace("Offset(slot)]", "Offset(slot,z,y,x)]")
    text = text.replace("if(kind>=2&&Eligible(index)){const auto&psi=", "if(kind>=2&&Eligible(index)){const int packed=PackedIndex();const auto&psi=")
    text = text.replace("psi.a+packed_index", "psi.a+packed").replace("psi.b+packed_index", "psi.b+packed").replace("psi.inv+packed_index", "psi.inv+packed")
    if any(name in text for name in ("point_base", "point_delta", "point_active", "packed_index")):
        raise ValueError("A reviewed multi-register context field remains")
    return text


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(text):
    if sha(text.encode()) != PARENT_CONSUMER:
        raise ValueError("Exact reviewed four-field point-context source required")
    old, new = original_helper(), helper()
    if text.count(old) != 1:
        raise ValueError("Actual reviewed context differs")
    original = text
    text = text.replace(old, new, 1)
    old_guard = "if(p.physical_n[0]==p.physical_n[1]&&p.physical_n[1]==p.physical_n[2])"
    new_guard = "if(p.physical_n[0]>0&&p.physical_n[0]<=(1<<20)&&p.physical_n[0]==p.physical_n[1]&&p.physical_n[1]==p.physical_n[2])"
    if text.count(old_guard) != 1:
        raise ValueError("Reviewed cubic wrapper guard changed")
    text = text.replace(old_guard, new_guard, 1)
    if text.replace(new_guard, old_guard, 1).replace(new, old, 1) != original:
        raise ValueError("One-word exact inverse failed")
    return text


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source required")
    before = (source / "src/asymmetric_native.cuh").read_text()
    after = transformed(before)
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(after)
    proof = {"schema": "beamz.one-word-cubic-cpml-point.v1", "exact_source_inverse": True,
             "parent_consumer_sha256": PARENT_CONSUMER, "result_consumer_sha256": sha(after.encode()),
             "stored_point_context_words": 1, "packed_and_active_word_type": "uint32_t",
             "integer_address_rematerialization": True,
             "selector": "reviewedcubicNoIOonehot AND0<N<=2**20; NativeSessionint32field/slab bounds remain",
             "encoding_range": "active body coordinates lie in[-16,N+30]; signedpacked fits31bits underselector",
             "original_field_graph_and_fallbacks_unchanged": True,
             "barriers_shared_allocation_and_pointer_identities_unchanged": True,
             "expected_moving_kernels": 62, "GPU_executed": False,
             "resource_claim": "One stored source-level word; actualptxas registers/stack/spills require exact-context build",
             "producer_sha256": sha(Path(__file__).read_bytes())}
    (output.parent / "point-word-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
