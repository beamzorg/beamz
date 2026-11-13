"""Copy-only two-family shared frontier with no persistent private field arrays.

The existing ordered cell header remains byte-identical. Both H/E center reads
come from their point owner's shared word and are overwritten only by that same
owner. Opposite-family reads cross exactly the two retained family barriers.
All own trail/final publications precede PF2: future coalesced H ghost writers
must not race those shared H readers. Source-bearing kernels stay unchanged.
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

from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES, once, transformed


def sha(data):
    return hashlib.sha256(data).hexdigest()


def lower_body(body):
    old = body
    body = once(body, " float h[3][Slots],e[3][Slots];\n", "")
    body = once(body,
        "  for(int c=0;c<3;++c){h[c][j]=Initial(p,c,z,y,x);e[c][j]=Initial(p,c+3,z,y,x);frame[(c+3)*PAD+Slot(zs,sy,sx)]=e[c][j];}",
        "  for(int c=0;c<6;++c)frame[c*PAD+Slot(zs,sy,sx)]=Initial(p,c,z,y,x);")
    body = once(body,
        "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;\n   e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;",
        "   frame[Slot(zs,sy,sx)]=v.hx;frame[PAD+Slot(zs,sy,sx)]=v.hy;frame[2*PAD+Slot(zs,sy,sx)]=v.hz;")
    body = once(body,
        "      if(field<3)return h[field][j];const int c=field-3;\n      if(!dz&&!ddy&&!ddx)return e[c][j];return frame[(c+3)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
        "      return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];")
    body = once(body, "     }h[c][j]=value;frame[c*PAD+Slot(zs,sy,sx)]=value;}",
                "     }frame[c*PAD+Slot(zs,sy,sx)]=value;}")
    body = once(body,
        "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;\n",
        "")
    body = once(body,
        "  // Coalesced import writers publish only H before B1. Old shared E still\n  // has cross-warp readers; its current ghost imports stay private/staged.",
        "  // Coalesced H imports precede PF1. E ghosts remain untouched until PF1\n  // retires every old-E reader; six-field low-X staging is read only after it.")
    ghost_e = """  // PF1 retired all old-E neighbors. Z/Y and low-X E writers are disjoint.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs==hz||sy==hy){const int dz=(zs-hz)&7;
   const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,dz,dy,dx);
   frame[3*PAD+Slot(zs,sy,sx)]=v.ex;frame[4*PAD+Slot(zs,sy,sx)]=v.ey;frame[5*PAD+Slot(zs,sy,sx)]=v.ez;
  }}
  for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,ddy=1+i%BY;
   const int zs=(hz+dz)&7,ys=(hy+ddy)&15;
   #pragma unroll
   for(int c=0;c<3;++c)frame[(c+3)*PAD+Slot(zs,ys,hx)]=low_stage[(c+3)*LowXPoints+i];
  }
"""
    body = once(body, "  RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);\n", ghost_e)
    body = once(body,
        "      if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];",
        "      return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];")
    body = once(body, "     }e[c][j]=value;}", "     }frame[(c+3)*PAD+Slot(zs,sy,sx)]=value;}")
    body = once(body, """  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   #pragma unroll
   for(int c=0;c<3;++c)frame[(c+3)*PAD+Slot(zs,sy,sx)]=e[c][j];
  }
  __syncthreads(); // PF2 publishes current E and retires all current H readers.
  if(!threadIdx.x)*e_step=n+1;
""", "")
    body = once(body, """   for(int c=0;c<3;++c){
    if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][j];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=e[c][j];}
    if(n==Depth){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);
     if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
     if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=e[c][j];
    }
   }
  }}
 }
}
""", """   for(int c=0;c<3;++c){
    const float hv=frame[c*PAD+Slot(zs,sy,sx)],ev=frame[(c+3)*PAD+Slot(zs,sy,sx)];
    if(surface>=0){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=hv;p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;}
    if(n==Depth){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);
     if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=hv;
     if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=ev;
    }
   }
  }}
  // All shared-H trail/final readers retire before next coalesced H ghost writes.
  __syncthreads(); // PF2 publishes current E and retires H readers/publications.
  if(!threadIdx.x)*e_step=n+1;
 }
}
""")
    if "h[c][j]" in body or "e[c][j]" in body or "float h[3]" in body or "RetireRegisterLow" in body:
        raise ValueError("Persistent private fields or their consumer remain")
    if old.count("__syncthreads()") != body.count("__syncthreads()"):
        raise ValueError("Two-family barrier count changed")
    for phase in (0, 1):
        for component in range(3):
            expression = f"beamz::cuda::trace_native::AdvanceNativeCell<{phase},{component},false>(env,read,n,z,y,x)"
            if old.count(expression) != 1 or body.count(expression) != 1:
                raise ValueError("Actual canonical arithmetic invocation changed")
    return body


def transform(original, *, scope=3):
    if scope not in (1, 2, 3):
        raise ValueError("Initial experiment admits only bulk/NoIO scope bits1/2")
    parent, base_proof = transformed(original, scope=scope)
    begin = parent.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void PermanentFamilyBody")
    end = parent.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", begin)
    old_body = parent[begin:end]
    body = lower_body(old_body)
    changed = parent[:begin] + body + parent[end:]
    # The entire source-bearing schedule is kept, including its kernel entries.
    marker = "// Private source-only storage schedule; source-bound"
    if original[original.index(marker):original.index("inline bool MakeSource")] != changed[changed.index(marker):changed.index("inline bool MakeSource")]:
        raise ValueError("Source-bearing kernel schedule changed")
    return changed, {"base_permanent_frontier": base_proof, "scope_bits": scope,
        "permanent_private_field_floats_per_thread": 0, "parent_private_field_floats_per_thread": 24,
        "shared_bytes": 100904, "loop_barriers": 2,
        "H_and_E_centers_same_owner_shared_read_then_write": True,
        "ghost_H_prePF1_ghost_E_postPF1": True,
        "lowX_H_E_staged_prePF1_then_split_scattered": True,
        "all_trail_final_publications_prePF2": True,
        "source_bodies_and_canonical_cell_header_unchanged": True,
        "generated_all_shared_body_sha256": sha(body.encode()),
        "generated_parent_body_sha256": sha(old_body.encode()),
        "all_other_generated_parent_bytes_identical": changed[:begin] == parent[:begin] and changed[begin+len(body):] == parent[end:]}


def patch_copy(source, output, *, scope=3):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint private source required")
    for name, expected in PARENT_HASHES.items():
        if sha((source / "src" / name).read_bytes()) != expected:
            raise ValueError(f"Exact selected52865 source required: {name}")
    original = (source / "src/asymmetric_native.cuh").read_text()
    changed, proof = transform(original, scope=scope)
    proof.update({"parent_hashes": PARENT_HASHES, "consumer_sha256": sha(changed.encode()),
                  "patcher_sha256": sha(Path(__file__).read_bytes()),
                  "base_patcher_sha256": sha(Path(sys.modules[transformed.__module__].__file__).read_bytes()),
                  "gpu_executed": False,
                  "requires": ["actual all-shared ring-version/hazard oracle", "zero stack/local/spill exact-context compilation",
                               "all30 exact GPU behavior", "memcheck/synccheck/racecheck", "matched whole-workload measurement"]})
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(changed)
    (output.parent / "shared-field-frontier-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(patch_copy(args.source, args.output, scope=args.scope)))


if __name__ == "__main__":
    main()
