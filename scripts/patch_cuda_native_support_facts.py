"""Copy-only canonical support/metadata lowering for the complete moving consumer.

Bind to the actual generated Yee operator and its host shape admission. Reuse
three minimum extents across Valid, derivative walls, CPML profiles and packed
E addressing. Arithmetic, pointers, source/DFT timing and synchronization stay
unchanged. This is a resource/performance experiment, not a default selection.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import (
    SUPPORT_DELTA,
    derive_geometry,
    verify_component_shapes,
)
from scripts.cuda_space_time_ir import FIELDS, IllegalSchedule
from scripts.generate_cuda_native_cell import generate_native_cell


def support_mask(minimum, point):
    z, y, x = point
    nz, ny, nx = minimum
    if z < 0 or y < 0 or x < 0 or z > nz or y > ny or x > nx:
        return 0
    iz, iy, ix = z < nz, y < ny, x < nx
    return (
        int(iz and iy)
        | (int(iz and ix) << 1)
        | (int(iy and ix) << 2)
        | (int(ix) << 3)
        | (int(iy) << 4)
        | (int(iz) << 5)
    )


def prove_support_facts(extents=(1, 2, 3, 5)):
    geometry = derive_geometry()
    masks = walls = addresses = 0
    for minimum in itertools.product(extents, repeat=3):
        shapes = [
            tuple(n + d for n, d in zip(minimum, delta, strict=True))
            for delta in SUPPORT_DELTA
        ]
        assert verify_component_shapes(shapes) == minimum
        for point in itertools.product(*(range(-1, n + 2) for n in minimum)):
            mask = support_mask(minimum, point)
            for field, shape in enumerate(shapes):
                valid = all(0 <= c < n for c, n in zip(point, shape, strict=True))
                assert bool(mask & (1 << field)) == valid
                masks += 1
                if field >= 3 and valid:
                    c = field - 3
                    z, y, x = point
                    actual = (z * shape[1] + y) * shape[2] + x
                    ny = minimum[1] + (c != 1)
                    nx = minimum[2] + (c != 0)
                    assert actual == (z * ny + y) * nx + x
                    addresses += 1
            for derivative in geometry["derivatives"]:
                output = FIELDS.index(derivative["output"])
                if not mask & (1 << output):
                    continue
                source = FIELDS.index(derivative["source"])
                axis = derivative["axis_zyx"]
                coordinate = point[axis]
                size = shapes[source][axis]
                phase = derivative["phase"]
                original = (
                    coordinate + 1 >= size
                    if not phase
                    else coordinate == 0 or coordinate == size
                )
                lowered = (
                    False
                    if not phase
                    else coordinate == 0 or coordinate == minimum[axis]
                )
                assert original == lowered
                walls += 1
    return {
        "support_bit_checks": masks,
        "valid_derivative_wall_checks": walls,
        "packed_E_address_checks": addresses,
        "geometry": geometry,
        "scope": "Actual canonical Operator dependencies; no floating arithmetic changes",
    }


def once(text, old, new):
    if text.count(old) != 1:
        raise IllegalSchedule(f"Expected one support-fact anchor: {old!r}")
    return text.replace(old, new, 1)


def patch(text, cell_source):
    canonical = generate_native_cell()
    if hashlib.sha256(cell_source.encode()).hexdigest() != canonical["source_sha256"]:
        raise IllegalSchedule(
            "The generated cell does not match the proved Yee Operator"
        )
    if (
        "if(f.z!=minimum[0]+dz||f.y!=minimum[1]+dy||f.x!=minimum[2]+dx)return false;"
        not in text
    ):
        raise IllegalSchedule("Canonical host component-shape admission is required")
    if not all(
        guard in text
        for guard in ("if(env.Valid(c,z,y,x))", "if(env.Valid(c+3,z,y,x))")
    ):
        raise IllegalSchedule(
            "Every generated H/E cell must retain its output support guard"
        )
    text = once(
        text,
        "int nz,ny,nx,frontier,macro_step;float h_source;",
        "int nz,ny,nx,frontier,macro_step;int physical_n[3];float h_source;",
    )
    text = once(
        text,
        "  if constexpr(SupportSafe)return true;return Contains(FinalField(p,field),z,y,x);",
        """  if constexpr(SupportSafe)return true;
  const int nz=p.physical_n[0],ny=p.physical_n[1],nx=p.physical_n[2];
  if(z<0||y<0||x<0||z>nz||y>ny||x>nx)return false;
  const bool iz=z<nz,iy=y<ny,ix=x<nx;
  return field==0?(iz&&iy):field==1?(iz&&ix):field==2?(iy&&ix):field==3?ix:field==4?iy:iz;""",
    )
    old = """  const auto&f=p.phase[family].output[term/2];const int coord=axis==0?z:axis==1?y:x;
  const int size=axis==0?f.z:axis==1?f.y:f.x;"""
    if text.count(old) != 2:
        raise IllegalSchedule("Expected separate Packed and LocalActive extent anchors")
    text = text.replace(
        old,
        """  const int coord=axis==0?z:axis==1?y:x;
  const int size=p.physical_n[axis]+family;""",
    )
    text = once(
        text,
        "const int c=index-3;const auto&v=p.phase[1];const auto&f=v.output[c];const int linear=(z*f.y+y)*f.x+x;",
        "const int c=index-3;const auto&v=p.phase[1];const int linear=(z*(p.physical_n[1]+(c!=1))+y)*(p.physical_n[2]+(c!=0))+x;",
    )
    text = once(
        text,
        """  if constexpr(!SupportSafe){const auto&f=InitialField(p,source);const int coordinate=axis==0?z:axis==1?y:x;const int size=axis==0?f.z:axis==1?f.y:f.x;
   if(!phase){if(coordinate+1>=size)return 0.f;}
   else if(coordinate==0||coordinate==size)return 0.f;
  }""",
        """  // H's forward wall guard is unreachable after the output support check.
  if constexpr(!SupportSafe){if(phase){const int coordinate=axis==0?z:axis==1?y:x;
   if(coordinate==0||coordinate==p.physical_n[axis])return 0.f;
  }}""",
    )
    text = once(
        text,
        " const int sizes[3]={BZ,BY,BX};int64_t tiles=1;",
        " for(int axis=0;axis<3;++axis)out->physical_n[axis]=minimum[axis];\n const int sizes[3]={BZ,BY,BX};int64_t tiles=1;",
    )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("A private copied source is required")
    path = args.source / "src/asymmetric_native.cuh"
    proof = prove_support_facts()
    text = patch(
        path.read_text(),
        (args.source / "src/asymmetric_native_cell_generated.cuh").read_text(),
    )
    path.write_text(text)
    (args.source.parent / "support-facts-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in proof.items() if k != "geometry"}))


if __name__ == "__main__":
    main()
