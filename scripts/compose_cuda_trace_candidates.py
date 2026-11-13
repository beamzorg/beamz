"""Compose two reviewed copied-source experiments against their exact base.

This records a clean textual three-way merge, never a semantic equivalence or
speed claim. Actual compile/resource, independent-state and sanitizer gates are
still mandatory for the composed executable. Production sources are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compose(base, left, right, output):
    base, left, right, output = (Path(x).resolve() for x in (base, left, right, output))
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents:
        raise ValueError("New noncanonical output required")
    for tree in (base, left, right):
        if tree in output.parents or output in tree.parents:
            raise ValueError("Disjoint output required")
    names = {str(p.relative_to(tree)) for tree in (base, left, right)
             for p in tree.rglob("*") if p.is_file()}
    changes, result = [], {}
    for name in sorted(names):
        before = (base / name).read_bytes() if (base / name).is_file() else None
        a = (left / name).read_bytes() if (left / name).is_file() else None
        b = (right / name).read_bytes() if (right / name).is_file() else None
        method = "identical"
        if a == b:
            value = a
        elif a == before:
            value, method = b, "right-only"
        elif b == before:
            value, method = a, "left-only"
        else:
            if any(x is None for x in (before, a, b)):
                raise ValueError("Conflicting file creation/deletion: " + name)
            r = subprocess.run(["git", "merge-file", "--stdout", str(left / name),
                                str(base / name), str(right / name)], capture_output=True)
            if r.returncode:
                raise ValueError("Non-clean composition: " + name)
            value, method = r.stdout, "clean-three-way"
        if value is not None:
            result[name] = value
        if value != before:
            changes.append({"file": name, "base_sha256": digest(before) if before is not None else None,
                            "left_sha256": digest(a) if a is not None else None,
                            "right_sha256": digest(b) if b is not None else None,
                            "result_sha256": digest(value) if value is not None else None,
                            "method": method})
    shutil.copytree(base, output)
    for p in output.rglob("*"):
        if p.is_file() and str(p.relative_to(output)) not in result:
            p.unlink()
    for name, data in result.items():
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    proof = {"schema": "beamz.trace-candidate-composition.v1", "base": str(base),
             "left": str(left), "right": str(right), "changes": changes,
             "conflicts": 0, "semantic_equivalence_not_established_by_text_merge": True,
             "result_files": {n: digest(v) for n, v in result.items()},
             "gpu_executed": False, "producer_sha256": digest(Path(__file__).read_bytes())}
    (output.parent / "composition.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "left", "right", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    proof = compose(a.base, a.left, a.right, a.output)
    print(json.dumps({"changed_files": len(proof["changes"]), "conflicts": 0}))
