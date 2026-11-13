"""Batch only the last two 8x32 partial-Ey planes, for 1024-thread classes.

Private copied source only. Per-point IEEE operations, ghost participation and
source/CPML/DFT callback order within each point are unchanged. Four old-own
Ex/Ez slots become the same four pending result slots. No extra shared storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "12486a02201fcd072e1f1aabd5e66b20e1e8049176f2043afa878f9cd5ace12d"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
START = "  #pragma unroll\n  for(int j=0;j<LocalZ;++j){const int zs=zbase+j;\n   float ex[YGroups],ez[YGroups];"
END = "  if(!threadIdx.x)*phase_step=n+1;"
RL = "\n  __syncthreads(); // RL: retire last-plane H XY readers before final pending restore."
POINT = "   #pragma unroll\n   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),si=gy*LocalZ+j,key=scratch_key(ys);"
PENDING = "    pending[key]=ex[gy];pending[1024+key]=ez[gy];"
COMMENT = """    // At j0, these snapshot cells have exactly this same-XY consumer. Its
    // Hx/Hy Z reads finished above, so pending stores need no extra barrier.
"""


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def transform(source):
    if digest(source) != PARENT:
        raise ValueError("Exact frozen full v3 consumer required")
    if "BZ=7,BY=31,BX=31,CZ=8,CY=32,CX=32" not in source:
        raise ValueError("Only complete owned7x31x31/cache8x32x32 geometry is admitted")
    if source.count(START) != 1:
        raise ValueError("Reviewed E plane loop changed")
    first, last = source.index(START), source.index(END, source.index(START))
    old = source[first:last]
    if old.count(RL) != 1 or old.count(POINT) != 1 or old.count(PENDING) != 1 or old.count(COMMENT) != 1:
        raise ValueError("Reviewed point/pending/retirement boundaries changed")
    loop = old[:old.index(RL)]
    point = loop[loop.index(POINT):loop.rfind("\n  }")]
    # Explicit fixed-index copies avoid a runtime gather into retained H/E arrays.
    no_pending = point.replace(COMMENT, "").replace(PENDING, "")
    point0 = no_pending.replace("ex[gy]", "tail_ex0[gy]").replace("ez[gy]", "tail_ez0[gy]")
    point1 = no_pending.replace("ex[gy]", "tail_ex1[gy]").replace("ez[gy]", "tail_ez1[gy]")
    prefix_loop = loop.replace("for(int j=0;j<LocalZ;++j)", "for(int j=0;j<LocalZ-2;++j)", 1)
    fused = r'''  if constexpr(T==1024){
   static_assert(CZ==8&&CY==32&&CX==32&&LocalZ==8&&YGroups==1,"Proved whole-Z uniform1024 class only");
'''+prefix_loop+r'''
   float tail_ex0[YGroups],tail_ez0[YGroups],tail_ex1[YGroups],tail_ez1[YGroups];
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),s0=gy*LocalZ+LocalZ-2,s1=s0+1;
    tail_ex0[gy]=frame[Slot(zbase+LocalZ-2,ys,sx)];tail_ez0[gy]=frame[PAD+Slot(zbase+LocalZ-2,ys,sx)];
    tail_ex1[gy]=frame[Slot(zbase+LocalZ-1,ys,sx)];tail_ez1[gy]=frame[PAD+Slot(zbase+LocalZ-1,ys,sx)];
    frame[Slot(zbase+LocalZ-2,ys,sx)]=h[0][s0];frame[PAD+Slot(zbase+LocalZ-2,ys,sx)]=h[2][s0];
    frame[Slot(zbase+LocalZ-1,ys,sx)]=h[0][s1];frame[PAD+Slot(zbase+LocalZ-1,ys,sx)]=h[2][s1];
   }
   __syncthreads(); // P67: publish BOTH H planes, retire all plane5 H XY readers.
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),key=scratch_key(ys);
    frame[Slot(zbase+LocalZ-3,ys,sx)]=pending[key];frame[PAD+Slot(zbase+LocalZ-3,ys,sx)]=pending[1024+key];
   }
   {constexpr int j=LocalZ-2;const int zs=zbase+j;
'''+point0+r'''
   }
   {constexpr int j=LocalZ-1;const int zs=zbase+j;
'''+point1+r'''
   }
   __syncthreads(); // RL67: retire ALL H6/H7 XY readers before restoring either E plane.
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX);
    frame[Slot(zbase+LocalZ-2,ys,sx)]=tail_ex0[gy];frame[PAD+Slot(zbase+LocalZ-2,ys,sx)]=tail_ez0[gy];
    frame[Slot(zbase+LocalZ-1,ys,sx)]=tail_ex1[gy];frame[PAD+Slot(zbase+LocalZ-1,ys,sx)]=tail_ez1[gy];
   }
  }else{
'''+old+r'''  }
'''
    changed = source[:first] + fused + source[last:]
    if changed.replace(fused, old, 1) != source:
        raise ValueError("Exact original body cannot be recovered")
    # The real point callbacks are copied literally, except their own saved slots
    # and removal of the now-unneeded single-plane pending store.
    for point_copy, label in ((point0, "0"), (point1, "1")):
        inverse = point_copy.replace(f"tail_ex{label}[gy]", "ex[gy]").replace(f"tail_ez{label}[gy]", "ez[gy]")
        if inverse != no_pending:
            raise ValueError("Point arithmetic/publication source changed")
    return changed, {"original_region": old, "new_region": fused, "point_source_sha256": digest(point),
                     "original_region_sha256": digest(old), "new_region_sha256": digest(fused)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    src, out = args.source.resolve(), args.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if out.exists() or src == out or out == canonical or src in out.parents or out in src.parents:
        parser.error("New disjoint private source tree required")
    if digest((src / "src/asymmetric_native_cell_generated.cuh").read_bytes()) != CELL:
        raise ValueError("Canonical point physics changed; recurrence/rounding review must be repeated")
    changed, regions = transform((src / "src/asymmetric_native.cuh").read_text())
    shutil.copytree(src, out)
    (out / "src/asymmetric_native.cuh").write_text(changed)
    proof = {"parent_consumer_sha256": PARENT, "consumer_sha256": digest(changed), "cell_sha256": CELL,
             "original_region_sha256": regions["original_region_sha256"], "new_region_sha256": regions["new_region_sha256"],
             "point_source_sha256": regions["point_source_sha256"], "exact_source_inverse": True,
             "generic_fallback_byte_identical": True, "enabled_threads": 1024,
             "barriers_per_step": {"enabled": 10, "generic": 11}, "extra_source_field_floats": 2,
             "shared_bytes_change": 0,
             "collectives": "Both Hy(-X) calls remain fullmask before Owned/Valid; private Ey6/7(+X) H calls unchanged",
             "scope": "CPU-proved schedule only; actual exact-context all-entry resources and GPU numerical/sync/race gates are pending"}
    (out.parent / "tail-pair-source-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    (out.parent / "tail-pair-regions.json").write_text(json.dumps(regions, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
