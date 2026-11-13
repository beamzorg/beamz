"""Emit a copied native consumer with proved canonical geometry metadata.

No live native file is edited. Only logical CPML extent/address inputs and
unreachable H forward guards change; E boundaries, arithmetic, ψ/profile
pointers, source/DFT lifecycles, ownership, and synchronization stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import derive_geometry
from scripts.cuda_space_time_ir import IllegalSchedule
from scripts.generate_cuda_native_cell import generate_native_cell

EXTENT = """  const auto&f=p.phase[family].output[term/2];const int coord=axis==0?z:axis==1?y:x;
  const int size=axis==0?f.z:axis==1?f.y:f.x;
"""
COMMON_EXTENT = """  // Canonical geometry proof: differentiated target extent is N_axis+Phase.
  const int coord=axis==0?z:axis==1?y:x;
  const int size=p.physical_n[axis]+family;
"""
DIFFERENCE = """  if constexpr(!Fast){const auto&f=InitialField(p,source);const int coordinate=axis==0?z:axis==1?y:x;const int size=axis==0?f.z:axis==1?f.y:f.x;
   if(!phase){if(coordinate+1>=size)return 0.f;}
   else if(coordinate==0||coordinate==size)return 0.f;
  }
"""
COMMON_DIFFERENCE = """  // Valid H output has extent N and forward E source extent N+1.
  // E output has extent N+1: the native zero branches at 0/N remain.
  if constexpr(!Fast){if(phase){const int coordinate=axis==0?z:axis==1?y:x;
   if(coordinate==0||coordinate==p.physical_n[axis])return 0.f;
  }}
"""


def once(source, old, new):
    if source.count(old) != 1:
        raise IllegalSchedule(f"Native geometry patch expected one anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source, *, cell_source):
    # Bind the implementation to the exact canonical traced point arithmetic.
    # A different specialization needs its own checked operator/source binding.
    cell = generate_native_cell()
    if hashlib.sha256(cell_source.encode()).hexdigest() != cell["source_sha256"]:
        raise IllegalSchedule("Native cell source does not match the proved canonical Operator")
    for admission in ("if(f.z!=minimum[0]+dz||f.y!=minimum[1]+dy||f.x!=minimum[2]+dx)return false;",
                      "if(env.Valid(c,z,y,x))", "if(env.Valid(c+3,z,y,x))"):
        if admission not in source:
            raise IllegalSchedule("Canonical shape/output-support admission anchor absent")
    source = once(source, "int nz,ny,nx,frontier,macro_step;float h_source;",
                  "int nz,ny,nx,frontier,macro_step;int physical_n[3];float h_source;")
    if source.count(EXTENT) != 2:
        raise IllegalSchedule("Expected separate Packed/LocalActive extent expressions")
    source = source.replace(EXTENT, COMMON_EXTENT)
    source = once(source, DIFFERENCE, COMMON_DIFFERENCE)
    source = once(source, " const int sizes[3]={BZ,BY,BX};int64_t tiles=1;",
                  " for(int axis=0;axis<3;++axis)out->physical_n[axis]=minimum[axis];\n"
                  " const int sizes[3]={BZ,BY,BX};int64_t tiles=1;")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cell-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input is frozen; output must be a different file")
    source = patch(args.input.read_text(), cell_source=args.cell_header.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    result = {"input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "output": str(args.output), "output_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "cell_header": str(args.cell_header), "cell_sha256": hashlib.sha256(args.cell_header.read_bytes()).hexdigest(),
              "geometry": derive_geometry(), "scope": "Metadata-only copied consumer; GPU parity/resource/performance gates required"}
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "geometry"}, indent=2))


if __name__ == "__main__":
    main()
