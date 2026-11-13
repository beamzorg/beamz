"""Proof-carrying spatial coefficient to runtime-scalar IR specialization.

Kernel identity contains the class and transformed graph, never coefficient
bits, resolution or scene data. A packet/region witness independently proves
that every supported owned E update reads the supplied scalar bits. The native
consumer uses its equivalent immutable C++ trajectory certificate before
dispatch; the ordinary cell remains the fallback.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from scripts.cuda_material_tile_facts import bit_lut
from scripts.cuda_native_geometry_ir import verify_component_shapes
from scripts.cuda_space_time_ir import IllegalSchedule, Operator, validate_operator
from scripts.cuda_stencil_trace import CoefficientRead, Expr, Update, serialize_updates


@dataclass(frozen=True)
class UniformElectricRegionClass:
    schema: str = "beamz.uniform-runtime-E-source-region.v1"


@dataclass(frozen=True)
class UniformElectricWitness:
    origin: tuple[int, int, int]
    owned: tuple[int, int, int]
    depth: int
    field_shapes: tuple[tuple[int, int, int], ...]
    scalar_bits: tuple[int, int, int]
    bounds: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    source_hashes: tuple[tuple[str, str], ...]
    supported_points: tuple[int, int, int]
    schema: str = UniformElectricRegionClass().schema


def _array_hash(array):
    prefix = json.dumps({"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True)
    digest = hashlib.sha256(prefix.encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def certify_region(inputs, origin, owned, depth, runtime_sources):
    """Bounded row-wise decode; no dense material or whole-domain graph needed."""
    if len(inputs) < 43:
        raise IllegalSchedule("Prepared packet lacks electric tables/codes")
    if (len(origin) != 3 or len(owned) != 3 or any(type(a) is not int or a < 0 for a in origin)
            or any(type(a) is not int or a < 1 for a in owned) or type(depth) is not int or depth < 1):
        raise IllegalSchedule("Integer region origin/owned/depth required")
    fields = tuple(np.asarray(a) for a in inputs[:6])
    shapes = tuple(tuple(int(n) for n in a.shape) for a in fields)
    verify_component_shapes(shapes)
    if any(a.dtype != np.dtype("float32") or not a.flags.c_contiguous for a in fields):
        raise IllegalSchedule("Canonical contiguous FP32 field supports required")
    values = np.asarray(runtime_sources)
    if values.shape != (3,) or values.dtype != np.dtype("float32") or not np.isfinite(values).all():
        raise IllegalSchedule("Three finite runtime FP32 electric sources required")
    bits = tuple(int(x) for x in values.view(np.uint32))
    bounds, hashes, points = [], [], []
    for c in range(3):
        shape = shapes[c + 3]
        count = int(np.prod(shape, dtype=np.int64))
        table, packed = np.asarray(inputs[37 + c]), np.asarray(inputs[40 + c])
        if (count >= 2**31 or packed.dtype != np.dtype("int32") or packed.ndim != 1
                or packed.size != (count + 3) // 4 or not packed.flags.c_contiguous
                or not table.flags.c_contiguous):
            raise IllegalSchedule("Canonical int32 byte-packed material required")
        lut = bit_lut(table)
        lower = tuple(min(size, max(0, o - depth)) for size, o in zip(shape, origin, strict=True))
        upper = tuple(min(size, max(0, o + b - 1)) for size, o, b in zip(shape, origin, owned, strict=True))
        words = packed.view(np.uint32)
        xs = np.arange(lower[2], upper[2], dtype=np.int64)
        for z in range(lower[0], upper[0]):
            for y in range(lower[1], upper[1]):
                linear = (z * shape[1] + y) * shape[2] + xs
                codes = (words[linear >> 2] >> (8 * (linear & 3))) & 255
                if np.any(lut[codes] != bits[c]):
                    raise IllegalSchedule(f"Electric source {c} is not bit-uniform over the complete trajectory box")
        bounds.append((lower, upper))
        points.append(int(np.prod(tuple(max(0, hi - lo) for lo, hi in zip(lower, upper, strict=True)))))
        hashes.append((_array_hash(table), _array_hash(packed)))
    if not any(points):
        raise IllegalSchedule("Region has no supported E update")
    return UniformElectricWitness(tuple(origin), tuple(owned), depth, shapes, bits,
                                  tuple(bounds), tuple(hashes), tuple(points))


def verify_region(witness, inputs, runtime_sources):
    if not isinstance(witness, UniformElectricWitness):
        raise IllegalSchedule("Typed uniform electric witness required")
    actual = certify_region(inputs, witness.origin, witness.owned, witness.depth, runtime_sources)
    if actual != witness:
        raise IllegalSchedule("Immutable uniform-region witness became stale")


def specialize_runtime_e(operator, fact):
    """Rewrite only coefficient leaves; no arithmetic folding or reassociation."""
    if fact != UniformElectricRegionClass():
        raise IllegalSchedule("Unknown runtime scalar region class")
    validate_operator(operator)
    names = {f"source_E{c}" for c in "xyz"}
    memo, used = {}, set()

    def transform(node):
        if id(node) in memo:
            return memo[id(node)]
        if node.op in ("coefficient", "parameter"):
            name = node.args[0].name if node.op == "coefficient" else node.args[0]
            if name in names:
                if node.op == "coefficient":
                    read = node.args[0]
                    if read.dtype != "float32" or any(read.offset) or read.axes not in ((), (0, 1, 2)):
                        raise IllegalSchedule("Runtime region coefficient has an unsupported footprint")
                used.add(name)
                result = Expr("coefficient", (CoefficientRead(name, ()),))
                memo[id(node)] = result
                return result
        result = Expr(node.op, tuple(transform(arg) if isinstance(arg, Expr) else arg for arg in node.args))
        memo[id(node)] = result
        return result

    result = Operator(tuple(Update(u.output, transform(u.value)) for u in operator.updates), operator.tensors)
    if used != names:
        raise IllegalSchedule("Exactly three canonical electric source leaves required")
    if [u.reads for u in result.updates] != [u.reads for u in operator.updates]:
        raise IllegalSchedule("Runtime coefficient specialization changed evolving-state dependencies")
    evidence = {"class": asdict(fact), "substituted": sorted(used),
                "runtime_leaves": [{"name": name, "axes": [], "dtype": "float32"} for name in sorted(used)],
                "operator": serialize_updates(result.updates),
                "arithmetic": "Every nonleaf node and operand order unchanged; no folding, regrouping, storage or clock change",
                "dispatch_requires": "Immutable host region certificate matches runtime scalar bits over ALL3 supported owned E trajectory boxes"}
    evidence["specialization_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return result, evidence
