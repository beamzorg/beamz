"""Emit admitted runtime-scalar/profile cells from the canonical typed IR.

The default native cell generator and its bef1... artifact are untouched.
This header is private to an explicitly certified NoIO/UniformE/IsotropicH
consumer. Coefficient VALUES remain runtime inputs, including resolution/dt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_readonly_profile_bindings import EqualAxisProfileClass, bind_profiles
from scripts.cuda_space_time_ir import yee_operator
from scripts.cuda_stencil_trace import CoefficientRead, coefficient_reads
from scripts.cuda_trace_runtime_uniform import (
    UniformElectricRegionClass,
    specialize_runtime_e,
)
from scripts.cuda_trace_specialize import MagneticScalarClass, specialize_scalar_class
from scripts.generate_cuda_native_cell import generate_native_cell


def verify_rounded_rewrite(canonical, specialized):
    """Validate every expression edge, allowing only the three admitted leaf rules."""
    if len(canonical.updates) != len(specialized.updates) or canonical.tensors != specialized.tensors:
        raise ValueError("Runtime facts changed output/state topology")
    counts = {"mul_rn": 0, "fma_rn": 0, "round_bf16": 0, "add": 0, "sub": 0, "neg": 0}

    def check(old, new):
        if old.op == "coefficient":
            read = old.args[0]
            if read.name in {f"decay_H{c}" for c in "xyz"}:
                if new.op != "constant" or new.args != (1.,):
                    raise ValueError("Admitted H decay must remain exact positive1")
                return
            if read.name in {f"source_H{c}" for c in "xyz"}:
                if new.op != "coefficient" or new.args != (CoefficientRead("source_Hx", ()),):
                    raise ValueError("Admitted H source must remain one runtime scalar")
                return
            if read.name in {f"source_E{c}" for c in "xyz"}:
                if new.op != "coefficient" or new.args != (CoefficientRead(read.name, ()),):
                    raise ValueError("Admitted E source must remain its component runtime scalar")
                return
        if old.op != new.op or len(old.args) != len(new.args):
            raise ValueError("Rounding/operator order changed")
        if old.op == "constant":
            if struct.pack("!f", old.args[0]) != struct.pack("!f", new.args[0]):
                raise ValueError("Unadmitted exact FP32 constant changed")
            return
        if old.op in counts:
            counts[old.op] += 1
        for a, b in zip(old.args, new.args, strict=True):
            if hasattr(a, "op"):
                check(a, b)
            elif a != b:
                raise ValueError("Unadmitted expression leaf changed")

    for old, new in zip(canonical.updates, specialized.updates, strict=True):
        if old.output != new.output:
            raise ValueError("Runtime facts changed evolving output identity")
        check(old.value, new.value)
    return {"all_expression_edges_checked": True, "per_output_tree_operation_counts": counts,
            "scope": "Exact node/operand ordering modulo admitted coefficient leaves; not compiler instruction ordering"}


def generate(*, profiles=False):
    canonical = yee_operator(cpml_axes=(0, 1, 2))
    operator, magnetic = specialize_scalar_class(canonical, MagneticScalarClass())
    operator, electric = specialize_runtime_e(operator, UniformElectricRegionClass())
    bindings = bind_profiles(operator, EqualAxisProfileClass()) if profiles else None
    result = generate_native_cell(operator, profile_bindings=bindings)
    namespace = "trace_native_runtime_profiles" if profiles else "trace_native_runtime"
    source = result["source"].replace("trace_native", namespace)
    source = source.replace("scripts/generate_cuda_native_cell.py", "scripts/generate_cuda_runtime_fact_cell.py")
    result.update(source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  namespace=namespace, magnetic_runtime_class=magnetic,
                  electric_runtime_region_class=electric,
                  rounded_rewrite=verify_rounded_rewrite(canonical, operator))
    coefficients = {r for update in operator.updates for r in coefficient_reads(update.value)}
    result["leaf_storage"] = {"runtime_scalar_sources": sorted(r.name for r in coefficients if r.name.startswith("source_") and not r.axes),
                              "spatial_electric_source_leaves": sum(r.name.startswith("source_E") and bool(r.axes) for r in coefficients),
                              "mutable_psi_outputs": sum(u.output.field.startswith("psi_") for u in operator.updates),
                              "readonly_profile_alias": profiles}
    result["dispatch_requires"] = [
        "Layout1, Policy1, Fast=false, IsotropicH=true, UniformE=true",
        "Native immutable three-component E region certificate covers all supported owned trajectories",
        "Native scalar H class verifies decay1 and exact equal finite scalar source bits",
    ]
    if profiles:
        result["dispatch_requires"].append("Bitwise equal same-phase/same-axis a/b/inv profiles from private upload snapshots")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = generate(profiles=args.profiles)
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != result["source"].encode():
            raise SystemExit("Generated admitted runtime cell is stale")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result["source"])
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({k: v for k, v in result.items() if k != "source"}, indent=2) + "\n")
    print(json.dumps({"source_sha256": result["source_sha256"], "leaf_storage": result["leaf_storage"]}))


if __name__ == "__main__":
    main()
