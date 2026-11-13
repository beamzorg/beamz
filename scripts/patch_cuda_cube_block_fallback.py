"""Copy-only T256 repair for exactly cube Layout2/isotropic/NoIO fallback.

No supported input is rejected. All other class thread mappings and every
source/field/CPML operation stay unchanged; full zero-local gate remains required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "0a36f36809b4f6b3dab6ed606bc66d348b21f22ca2b64015581686789ad6f705"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(raw):
    if sha(raw) != PARENT:
        raise ValueError("Exact frozen cube-v1 consumer required")
    text = raw.decode()
    old = text
    edits = [
        (
            "?1024:512,Slots=PAD/T,ZStep=T/(CY*CX);",
            "?1024:((Layout==2&&IsotropicH&&Policy==1)?256:512),Slots=PAD/T,ZStep=T/(CY*CX);",
        ),
        (
            "__launch_bounds__((Fast?1024:512),1) void Asymmetric(",
            "__launch_bounds__((Fast?1024:((Layout==2&&IsotropicH&&Policy==1)?256:512)),1) void Asymmetric(",
        ),
        (
            "<<<count,(Fast?1024:512),SharedBytes,stream>>>",
            "<<<count,(Fast?1024:((Layout==2&&IsotropicH&&Policy==1)?256:512)),SharedBytes,stream>>>",
        ),
    ]
    for a, b in edits:
        if text.count(a) != 1:
            raise ValueError(f"Expected one thread selector:{a!r}")
        text = text.replace(a, b, 1)
    inverse = text
    for a, b in reversed(edits):
        inverse = inverse.replace(b, a, 1)
    assert inverse == old
    return text.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src, out = a.parent.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or out in (src, canonical)
        or src in out.parents
        or out in src.parents
        or canonical in out.parents
    ):
        p.error("New disjoint private output required")
    if sha((src / "src/asymmetric_native_cell_generated.cuh").read_bytes()) != CELL:
        p.error("Canonical cell changed")
    changed = transformed((src / "src/asymmetric_native.cuh").read_bytes())
    shutil.copytree(src, out)
    (out / "src/asymmetric_native.cuh").write_bytes(changed)
    proof = {
        "parent_consumer_sha256": PARENT,
        "consumer_sha256": sha(changed),
        "canonical_cell_sha256": CELL,
        "producer_sha256": sha(Path(__file__).read_bytes()),
        "exact_inverse_three_selectors": True,
        "only_class": "Layout2&&IsotropicH&&Policy1&&!Fast",
        "depths": [4, 16],
        "thread_count": 256,
        "slots": 16,
        "zstep": 1,
        "cache_xy": 256,
        "unchanged": [
            "all other56 mappings",
            "field/CPML/source operations",
            "all four barriers",
            "source5B body",
            "admission and all fallbacks",
            "shared54628/source60120",
        ],
        "expected_kernels": 58,
        "gpu_executed": False,
    }
    (out.parent / "block-fallback-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
