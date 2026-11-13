"""Copied startup-address lowering under canonical shapes and C-contiguity.

Only the previously certified unchecked cache loads change. Native Describe
already rejects non-C-contiguous input and constructs exact row/plane strides.
The old checked startup fallback and all arithmetic/phases remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry


def prove_addresses():
    count = 0
    for minimum in itertools.product((1, 2, 5, 9), repeat=3):
        for field, delta in enumerate(SUPPORT_DELTA):
            shape = tuple(n + d for n, d in zip(minimum, delta, strict=True))
            volume = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
            ny = minimum[1] + (field in (1, 3, 5))
            nx = minimum[2] + (field in (0, 4, 5))
            for z, y, x in itertools.product(*(range(n) for n in shape)):
                original = z * shape[1] * shape[2] + y * shape[2] + x
                lowered = (z * ny + y) * nx + x
                assert original == lowered == volume[z, y, x]
                count += 1
    return {
        "address_checks": count,
        "geometry": derive_geometry(),
        "scope": "All six actual canonical supports, C strides and valid coordinates; integer offsets fit int32 by Native Describe",
    }


def patch(consumer, extension):
    admission = (
        'throw std::invalid_argument("native inputs must be C-contiguous")',
        "result.descriptor.row_stride = static_cast<int>(result.descriptor.dims[2]);",
        "result.descriptor.dims[1] * result.descriptor.dims[2]",
        "elements > INT32_MAX / dimension",
        "final.dims[a]!=field.dims[a]",
        "moving requires dense canonical field pitches",
    )
    if not all(anchor in extension for anchor in admission):
        raise ValueError("Native contiguous/int32 stride admission must remain exact")
    if "int physical_n[3]" not in consumer or "InitialUnchecked" not in consumer:
        raise ValueError("Canonical support and certified initial-box parent required")
    if (
        "if(f.z!=minimum[0]+dz||f.y!=minimum[1]+dy||f.x!=minimum[2]+dx)return false;"
        not in consumer
    ):
        raise ValueError("Actual canonical field-shape admission is required")
    old = """__device__ __forceinline__ float InitialUnchecked(const Parameters&p,int field,int z,int y,int x){
 const auto&f=InitialField(p,field);return f.data[z*f.plane+y*f.row+x];
}"""
    new = """__device__ __forceinline__ float InitialUnchecked(const Parameters&p,int field,int z,int y,int x){
 const auto&f=InitialField(p,field);
 const int ny=p.physical_n[1]+(field==1||field==3||field==5);
 const int nx=p.physical_n[2]+(field==0||field==4||field==5);
 return f.data[(z*ny+y)*nx+x];
}"""
    if consumer.count(old) != 1:
        raise ValueError("Expected one exact certified unchecked startup helper")
    return consumer.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("Private copied source required")
    header = args.source / "src/asymmetric_native.cuh"
    extension = args.source / "src/native_extension.cc"
    parent = header.read_text()
    result = patch(parent, extension.read_text())
    header.write_text(result)
    proof = prove_addresses()
    proof.update(
        parent_consumer_sha256=hashlib.sha256(parent.encode()).hexdigest(),
        consumer_sha256=hashlib.sha256(result.encode()).hexdigest(),
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
    )
    (args.source.parent / "initial-address-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in proof.items() if k != "geometry"}))


if __name__ == "__main__":
    main()
