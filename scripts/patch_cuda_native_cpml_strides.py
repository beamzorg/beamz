"""Copy-only lowering of host-proved canonical CPML slab addresses.

This removes descriptor row/plane loads, not profile or psi pointer identities.
The actual NativeSession admission proves every psi is a dense 24-cell slab
of its canonical target field. Slot IDs remain the canonical twelve CPML
terms, and every floating operation, pointer and scheduler remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry
from scripts.cuda_space_time_ir import IllegalSchedule

CANONICAL_CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
AXES = (1, 0, 0, 2, 2, 1)
INT_MAX = 2**31 - 1
OLD = """  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  const auto&psi=p.phase[family].psi[term];const int packed=Packed(slot,z,y,x);
  return (axis==0?packed:z)*psi.plane+(axis==1?packed:y)*psi.row+(axis==2?packed:x);"""
NEW = """  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term),c=term/2;
  const int packed=Packed(slot,z,y,x);
  // NativeSession proves canonical target shapes, dense24 slabs and int32 offsets.
  const int ny=axis==1?24:p.physical_n[1]+(family?(c!=1):(c==1));
  const int nx=axis==2?24:p.physical_n[2]+(family?(c!=0):(c==0));
  return ((axis==0?packed:z)*ny+(axis==1?packed:y))*nx+(axis==2?packed:x);"""

ADMISSION = (
    "native inputs must be C-contiguous",
    "(dimension && elements > INT32_MAX / dimension)",
    "constexpr int dz[6]={0,0,1,1,1,0},dy[6]={0,1,0,1,0,1},dx[6]={1,0,0,0,1,1};",
    "field.dims[a]!=minimum[a]+add[a]||final.dims[a]!=field.dims[a]",
    "f->row_stride!=f->dims[2]||f->plane_stride!=f->dims[1]*f->dims[2]||f->compact_slots",
    "constexpr int axes[6]={1,0,0,2,2,1};",
    'if(p.uniform_cpml_thickness!=12)throw std::invalid_argument("moving requires uniform CPML12");',
    "psi.rank!=3||psi.element_type!=kBeamzF32||p.outputs[3+term].data!=psi.data||psi.compact_slots",
    "psi.dims[d]!=(d==axis?24:target.dims[d])",
    "psi.row_stride!=psi.dims[2]||psi.plane_stride!=psi.dims[1]*psi.dims[2]",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def field_shape(minimum, field):
    return tuple(n + d for n, d in zip(minimum, SUPPORT_DELTA[field], strict=True))


def slab_shape(minimum, slot):
    family, term = divmod(slot, 6)
    shape = list(field_shape(minimum, 3 * family + term // 2))
    shape[AXES[term]] = 24
    return tuple(shape)


def admitted(minimum):
    if len(minimum) != 3 or any(type(n) is not int or n < 1 for n in minimum):
        return False
    shapes = [field_shape(minimum, i) for i in range(6)]
    shapes += [slab_shape(minimum, i) for i in range(12)]
    return all(
        all(0 < x <= INT_MAX for x in s) and math.prod(s) <= INT_MAX for s in shapes
    )


def lowered_offset(minimum, slot, point):
    family, term = divmod(slot, 6)
    axis, c = AXES[term], term // 2
    z, y, x = point
    coord = point[axis]
    packed = coord if coord < 12 else 24 + coord - minimum[axis] - family
    ny = 24 if axis == 1 else minimum[1] + (c != 1 if family else c == 1)
    nx = 24 if axis == 2 else minimum[2] + (c != 0 if family else c == 0)
    qz, qy, qx = (
        packed if axis == 0 else z,
        packed if axis == 1 else y,
        packed if axis == 2 else x,
    )
    intermediate = (qz * ny, qz * ny + qy, (qz * ny + qy) * nx)
    offset = intermediate[-1] + qx
    if any(not 0 <= v <= INT_MAX for v in (*intermediate, offset)):
        raise IllegalSchedule("Canonical address intermediate exceeds int32")
    return offset


def prove(extents=(1, 2, 11, 12, 13, 23, 24, 25, 31, 64, 128, 512)):
    import numpy as np

    geometry = derive_geometry()
    terms = {x["term"]: x for x in geometry["cpml_terms"]}
    for family in (0, 1):
        for term in range(6):
            t = terms[f"psi_{'HE'[family]}{term}"]
            assert t["axis_zyx"] == AXES[term]
            assert t["output"] == "HE"[family] + "xyz"[term // 2]
    minima = list(itertools.product(extents, repeat=3))
    minima += [(1289, 1289, 1289)]
    for axis in range(3):
        for extent in [INT_MAX // 48, INT_MAX // 48 - 1, INT_MAX // 48 - 2]:
            shape = [1, 1, 1]
            shape[axis] = extent
            minima.append(tuple(shape))
    count = near_limit = maximum = 0
    for minimum in minima:
        if not admitted(minimum):
            continue
        for slot in range(12):
            family, term = divmod(slot, 6)
            axis = AXES[term]
            target = field_shape(minimum, 3 * family + term // 2)
            shape = slab_shape(minimum, slot)
            samples = [
                sorted(
                    {
                        q
                        for q in [0, 1, 11, 12, n // 2, n - 13, n - 12, n - 2, n - 1]
                        if 0 <= q < n
                    }
                )
                for n in target
            ]
            for point in itertools.product(*samples):
                coordinate = point[axis]
                if not (coordinate < 12 or coordinate >= target[axis] - 12):
                    continue
                # Independent oracle uses the actual target/slab shape and NumPy's
                # bounds-checked C-order indexer, not the lowered stride formula.
                q = list(point)
                q[axis] = (
                    coordinate if coordinate < 12 else coordinate - target[axis] + 24
                )
                expected = int(np.ravel_multi_index(tuple(q), shape, order="C"))
                original = q[0] * shape[1] * shape[2] + q[1] * shape[2] + q[2]
                actual = lowered_offset(minimum, slot, point)
                assert actual == expected == original
                assert actual < math.prod(shape) <= INT_MAX
                count += 1
                near_limit += actual >= INT_MAX - 10000
                maximum = max(maximum, actual)
    return {
        "address_cases": count,
        "near_int32_limit_cases": near_limit,
        "maximum_checked_offset": maximum,
        "geometry_sha256": geometry["geometry_sha256"],
        "scope": "Valid active canonical field points, dense24 slabs; exact integer address, no FP or pointer identity changes",
    }


def transform(consumer, extension, cell):
    if digest(cell) != CANONICAL_CELL:
        raise IllegalSchedule(
            "Canonical cell source is not the reviewed twelve-term ABI"
        )
    missing = [x for x in ADMISSION if x not in extension.decode()]
    if missing:
        raise IllegalSchedule(f"Missing actual host admission: {missing}")
    text = consumer.decode()
    for anchor in [
        "if(env.Valid(c,z,y,x))",
        "if(env.Valid(c+3,z,y,x))",
        "out->physical_n[axis]=minimum[axis]",
    ]:
        if anchor not in text:
            raise IllegalSchedule(f"Missing canonical support/extent binding: {anchor}")
    if text.count(OLD) != 1:
        raise IllegalSchedule("Expected exactly one original PsiOffset body")
    changed = text.replace(OLD, NEW, 1)
    assert changed.replace(NEW, OLD, 1).encode() == consumer
    return changed.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--proof", action="store_true")
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        output.exists()
        or output == canonical
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        p.error("Output must be a new disjoint noncanonical source tree")
    c = source / "src/asymmetric_native.cuh"
    e = source / "src/native_extension.cc"
    cell = source / "src/asymmetric_native_cell_generated.cuh"
    changed = transform(c.read_bytes(), e.read_bytes(), cell.read_bytes())
    report = {
        "parent_consumer_sha256": digest(c.read_bytes()),
        "consumer_sha256": digest(changed),
        "extension_sha256": digest(e.read_bytes()),
        "cell_sha256": digest(cell.read_bytes()),
        "inverse_source_binding": True,
        "host_admission_anchors": list(ADMISSION),
    }
    if a.proof:
        report["integer_oracle"] = prove()
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_bytes(changed)
    (output.parent / "cpml-strides-proof.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "host_admission_anchors"})
    )


if __name__ == "__main__":
    main()
