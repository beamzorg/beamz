"""Copy-only CPML store-pointer reuse under the existing twelve-alias guard.

This is a resource experiment, not a runtime promotion. The underlying input
allocations are mutable; const belongs to the read view. Ordinary coherent
LoadLocal accesses remain unchanged. No arithmetic, offset, or schedule changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
DESCRIPTOR = "62d4fa66106469e7a89c1848361ac72fab6aef9b01bb8cb09dca0fbcafa17da4"
PARENTS = {
    "f0657105ea5a57219d48a7a067d1ef53f6246676579e722ffd0670a14d3d8f49":
        "8f122cdbb850374f44c8e59c3282274b744ebff0ef5a907540a245a53a730bcf",
    "b514c317c82a4e680d304745ec5a8ffb74f9be47bdd4874b3742954521f10906":
        "c7bd59160aa12fc86b0a54bc4f355d4dbd45c81b7276fb51f2e2e25f9e2cc9a0",
}
OLD = "p.phase[slot/6].psi[slot%6].output[PsiOffset(slot,z,y,x)]=value;"
NEW = "const_cast<float*>(p.phase[slot/6].psi[slot%6].input)[PsiOffset(slot,z,y,x)]=value;"
LOAD = "return p.phase[slot/6].psi[slot%6].input[PsiOffset(slot,z,y,x)];"
ALIAS_GUARD = (
    "for(int term=0;term<6;++term){if(h.inputs[31+term].data!="
    "h.outputs[3+term].data||e.inputs[31+term].data!="
    "e.outputs[3+term].data)return false;}"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(consumer: bytes, extension: bytes, cell: bytes, descriptor: bytes):
    parent = sha(consumer)
    if parent not in PARENTS or sha(extension) != PARENTS[parent]:
        raise ValueError("Exact reviewed Obs9f6a or joint-phase parent required")
    if sha(cell) != CELL or sha(descriptor) != DESCRIPTOR:
        raise ValueError("Unreviewed canonical cell or descriptor mapping")
    text = consumer.decode()
    if text.count(ALIAS_GUARD) != 1 or text.count(LOAD) != 1:
        raise ValueError("Missing twelve-pointer alias admission or coherent load")
    if text.count(OLD) != 1 or NEW in text:
        raise ValueError("Expected exactly one original StoreLocal pointer")
    changed = text.replace(OLD, NEW, 1).encode()
    if changed.decode().replace(NEW, OLD, 1).encode() != consumer:
        raise ValueError("Inverse source binding failed")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        output.exists()
        or source == output
        or source in output.parents
        or output in source.parents
        or output == canonical
        or canonical in output.parents
    ):
        parser.error("Output must be a new disjoint, noncanonical source tree")
    names = (
        "asymmetric_native.cuh",
        "native_extension.cc",
        "asymmetric_native_cell_generated.cuh",
        "compact_full_edge.cuh",
    )
    inputs = [(source / "src" / name).read_bytes() for name in names]
    changed = transform(*inputs)
    shutil.copytree(source, output)
    (output / "src" / names[0]).write_bytes(changed)
    report = {
        "input_sha256": dict(zip(names, map(sha, inputs), strict=True)),
        "consumer_sha256": sha(changed),
        "inverse_source_binding": True,
        "existing_twelve_alias_guard": True,
        "ordinary_coherent_load_unchanged": True,
        "scope": "Only StoreLocal pointer reuse; no arithmetic/offset/schedule changes",
        "runtime_validated": False,
    }
    (output / "psi-input-store-source-proof.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
