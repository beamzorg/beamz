"""Copy-only depth32 instantiation of the exact accepted NoIO1024 consumer.

The native host planner already supports32. Extend only the API/Prepare gates
and template dispatch; keep all existing4/16 bodies, storage and FP equations.
This changes workspace/history size and certification coverage at runtime and
therefore requires full memory admission, continuation and numerical gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_CONSUMER = "3ed4a4f837df1ca60f409a243c0dd8a63778c0b2b983bac0ee895b56da04d6aa"
PARENT_EXTENSION = "844217c56d96bbe827bcfa36522075d7f84658bd238f739dbf757858904ea596"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
PREPARE_OLD = "(w.depth!=4&&w.depth!=16)"
PREPARE_NEW = "(w.depth!=4&&w.depth!=16&&w.depth!=32)"
DISPATCH = " if(w.depth==16){if(layout==2)return DispatchClass<16,2>(stream,p,w);return DispatchClass<16,1>(stream,p,w);}"
EXTRA = " if(w.depth==32){if(layout==2)return DispatchClass<32,2>(stream,p,w);return DispatchClass<32,1>(stream,p,w);}"
GATE_OLD = 'if(depth!=4&&depth!=16)throw std::invalid_argument("compiled native moving depth must be4 or16");'
GATE_NEW = 'if(depth!=4&&depth!=16&&depth!=32)throw std::invalid_argument("compiled native moving depth must be4/16/32");'
ATTRIBUTE = '\n  module.attr("moving_compiled_depths")=nb::make_tuple(4,16,32);'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transform(consumer, extension, cell, planner):
    for name, data, expected in [
        ("consumer", consumer, PARENT_CONSUMER),
        ("extension", extension, PARENT_EXTENSION),
        ("cell", cell, CELL),
    ]:
        if digest(data) != expected:
            raise ValueError(f"Exact accepted4941 {name} required")
    if "if(depth!=4&&depth!=8&&depth!=16&&depth!=32)" not in planner.decode():
        raise ValueError("Actual native host planner does not admit32")
    c, e = consumer.decode(), extension.decode()
    if c.count(PREPARE_OLD) != 1 or c.count(DISPATCH) != 1 or e.count(GATE_OLD) != 2:
        raise ValueError("Native depth gate or dispatch multiplicity changed")
    c = c.replace(PREPARE_OLD, PREPARE_NEW, 1).replace(
        DISPATCH, DISPATCH + "\n" + EXTRA, 1
    )
    e = e.replace(GATE_OLD, GATE_NEW).replace(
        "NB_MODULE(_cuda_native, module) {",
        "NB_MODULE(_cuda_native, module) {" + ATTRIBUTE,
        1,
    )
    inverse_c = c.replace(DISPATCH + "\n" + EXTRA, DISPATCH, 1).replace(
        PREPARE_NEW, PREPARE_OLD, 1
    )
    inverse_e = e.replace(ATTRIBUTE, "", 1).replace(GATE_NEW, GATE_OLD)
    if inverse_c.encode() != consumer or inverse_e.encode() != extension:
        raise ValueError("Existing4/16 code cannot be reconstructed bytewise")
    return c.encode(), e.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
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
    planner = source / "src/native_moving.cc"
    changed_c, changed_e = transform(
        c.read_bytes(), e.read_bytes(), cell.read_bytes(), planner.read_bytes()
    )
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_bytes(changed_c)
    (output / "src/native_extension.cc").write_bytes(changed_e)
    proof = {
        "parent_consumer_sha256": digest(c.read_bytes()),
        "consumer_sha256": digest(changed_c),
        "parent_extension_sha256": digest(e.read_bytes()),
        "extension_sha256": digest(changed_e),
        "planner_sha256": digest(planner.read_bytes()),
        "cell_sha256": digest(cell.read_bytes()),
        "inverse_source_exact": True,
        "compiled_depths": [4, 16, 32],
        "expected_moving_kernels": 75,
        "scope": "Depth32 gate/instantiation only; newworkspace/history/certificationcoverage require all30 correctness and memory gate",
    }
    (output.parent / "depth32-source-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
