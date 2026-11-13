"""Independent traced warp-layout footprints and address-sector enumeration.

This is a CPU cost/proof model, not generated CUDA or a speed prediction. It
derives field publication faces from ordered canonical Yee reads. Sector
counts are ideal distinct32B sectors per scalar warp instruction: caches,
compiler instruction formation, replay and timing are deliberately unmodeled.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS, validate_stream_signature, yee_operator
from scripts.cuda_stencil_trace import serialize_updates

SHAPE = (8, 16, 32)
OWNED = (7, 15, 31)
SURFACE = 735
LOW_X = 105
LAYOUTS = ((1, 32), (2, 16), (4, 8), (8, 4), (16, 2))


@dataclass(frozen=True)
class Layout:
    warp_y: int
    warp_x: int

    def __post_init__(self):
        if (self.warp_y, self.warp_x) not in LAYOUTS:
            raise ValueError("Admitted complete32-lane warp rectangle required")

    def point(self, tid, z):
        warp, lane = divmod(tid, 32)
        warps_x = 32 // self.warp_x
        return z, (warp // warps_x) * self.warp_y + lane // self.warp_x, (warp % warps_x) * self.warp_x + lane % self.warp_x

    def owner(self, point):
        _, y, x = point
        warp = (y // self.warp_y) * (32 // self.warp_x) + x // self.warp_x
        lane = (y % self.warp_y) * self.warp_x + x % self.warp_x
        return warp * 32 + lane


def canonical_reads():
    op = yee_operator(cpml_axes=(0, 1, 2))
    validate_stream_signature(op)
    return op, {u.output.field: tuple(sorted(r for r in u.reads if r.field in FIELDS)) for u in op.updates if u.output.field in FIELDS}


def required_axes(reads, layout=None):
    result = {f: {} for f in FIELDS}
    for dst, values in reads.items():
        for r in values:
            if r.field[0] == dst[0]:
                continue
            for axis, delta in enumerate(r.offset):
                if axis not in (1, 2) or not delta:
                    continue
                if layout is not None and (layout.warp_y if axis == 1 else layout.warp_x) == SHAPE[axis]:
                    # A full cache axis lies inside one warp, including its
                    # wrap-around neighbor. It never needs CTA publication.
                    continue
                previous = result[r.field].setdefault(axis, delta)
                if previous != delta or delta not in (-1, 1):
                    raise ValueError("One unit directed face per field/axis required")
    return result


def boundary(axis, delta, layout):
    width = layout.warp_y if axis == 1 else layout.warp_x
    return 0 if delta == 1 else width - 1


def footprint(field, layout, requirements):
    faces = requirements[field]
    return {p for p in itertools.product(*(range(s) for s in SHAPE))
            if any(p[a] % (layout.warp_y if a == 1 else layout.warp_x) == boundary(a, d, layout) for a, d in faces.items())}


def physical(point, version):
    return tuple(-version - 1 + ((p + version + 1) % s) for p, s in zip(point, SHAPE, strict=True))


def publication_model(layout):
    op, reads = canonical_reads()
    requirements = required_axes(reads, layout)
    fields = []
    for field in FIELDS:
        axes = requirements[field]
        y = SHAPE[0] * (SHAPE[1] // layout.warp_y) * SHAPE[2] if 1 in axes else 0
        x = SHAPE[0] * SHAPE[1] * (SHAPE[2] // layout.warp_x) if 2 in axes else 0
        intersection = SHAPE[0] * (SHAPE[1] // layout.warp_y) * (SHAPE[2] // layout.warp_x) if y and x else 0
        words = y + x - intersection
        if len(footprint(field, layout, requirements)) != words:
            raise AssertionError("Enumerated and factored publication sizes differ")
        fields.append({"field": field, "axis_direction": axes, "Y_face_words": y, "X_face_words": x,
                       "intersection_words": intersection, "allocated_words": words})
    words = sum(f["allocated_words"] for f in fields)
    return {"layout": asdict(layout), "threads": 512, "private_field_words_per_thread": 48,
            "fields": fields, "shared_field_words": words, "shared_field_bytes": words * 4,
            "low_X_stage_bytes": 6 * LOW_X * 4, "base_pointer_bytes": 8 * 8,
            "two_phase_clock_and_window_bytes": 4 * 4,
            "shared_total_bytes": words * 4 + 6 * LOW_X * 4 + 8 * 8 + 4 * 4,
            "operator_sha256": hashlib.sha256(json.dumps(serialize_updates(op.updates), sort_keys=True).encode()).hexdigest()}


def prove_routes(layout, depth=16, negative=None):
    _, reads = canonical_reads()
    requirements = required_axes(reads, layout)
    published = {f: footprint(f, layout, requirements) for f in FIELDS}
    if negative == "opposite-face":
        published = {f: {p for p in itertools.product(*(range(s) for s in SHAPE))
                         if any(p[a] % (layout.warp_y if a == 1 else layout.warp_x) == boundary(a, -d, layout) for a, d in requirements[f].items())}
                     for f in FIELDS}
    elif negative == "omit-union-intersection":
        for f in FIELDS:
            if len(requirements[f]) == 2:
                published[f] = {p for p in published[f] if not all(p[a] % (layout.warp_y if a == 1 else layout.warp_x) == boundary(a, d, layout) for a, d in requirements[f].items())}
    elif negative not in (None, "wrong-forward-version"):
        raise ValueError("Unknown deliberate negative mutation")
    routes = {"same_thread": 0, "same_warp": 0, "shared_cross_warp": 0}
    seen = {f: set() for f in FIELDS}
    checks = 0
    for version in range(1, depth + 1):
        heads = tuple((-version - 1) % s for s in SHAPE)
        for point in itertools.product(*(range(s) for s in SHAPE)):
            if any(p == h for p, h in zip(point, heads, strict=True)):
                continue
            tid = layout.owner(point)
            for dst, values in reads.items():
                for r in values:
                    source = tuple((p + d) % s for p, d, s in zip(point, r.offset, SHAPE, strict=True))
                    source_version = version - 1 + r.version
                    if negative == "wrong-forward-version" and dst.startswith("H") and any(r.offset):
                        source_version += 1
                    actual = physical(source, source_version)
                    expected = tuple(p + d for p, d in zip(physical(point, version), r.offset, strict=True))
                    if actual != expected:
                        raise AssertionError((negative, "Incorrect logical-version/physical coordinate", version, dst, r, point, source, actual, expected))
                    source_tid = layout.owner(source)
                    route = "same_thread" if source_tid == tid else "same_warp" if source_tid // 32 == tid // 32 else "shared_cross_warp"
                    if route == "same_warp":
                        lane = tid % 32
                        delta_y, delta_x = r.offset[1:]
                        if delta_y:
                            peer = (lane + (layout.warp_x if delta_y > 0 else 32 - layout.warp_x)) & 31
                        elif delta_x:
                            peer = (lane & (32 - layout.warp_x)) | ((lane + (1 if delta_x > 0 else layout.warp_x - 1)) & (layout.warp_x - 1))
                        else:
                            raise AssertionError("Same-warp distinct-thread route has no XY offset")
                        if peer != source_tid % 32:
                            raise AssertionError("Lowered C++ peer formula disagrees with actual canonical source owner")
                    routes[route] += 1
                    if route == "shared_cross_warp":
                        if source not in published[r.field]:
                            raise AssertionError((negative, "Unpublished actual cross-warp producer", version, dst, r, point, source))
                        seen[r.field].add(source)
                    checks += 1
    # All16 actual ring phases suffice to expose every allocated producer face.
    if depth >= 16 and negative is None and seen != published:
        raise AssertionError("Conservative full-ring face coverage differs from actual read union")
    return {"layout": asdict(layout), "versions": [1, depth], "actual_read_checks": checks,
            "route_counts": routes, "all_actual_cross_warp_reads_published": True,
            "source_physical_coordinate_and_version_exact": True,
            "within_warp_CPP_shuffle_peer_formulas_exact": True,
            "field_read_union_words": {f: len(v) for f, v in seen.items()},
            "scope": "Address/version/ownership and publisher footprint; barrier visibility and actual FP CUDA execution are separate gates"}


def eval_peer(expression, lane):
    """Evaluate only the small integer grammar used by actual CUDA peer code."""
    tree = ast.parse(expression, mode="eval")
    operations = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
                  ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b}

    def value(node):
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id == "lane":
            return lane
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](value(node.left), value(node.right))
        raise ValueError("Unknown CUDA shuffle peer expression grammar")

    return value(tree)


def audit_lowering(source, output, warp_y=4):
    """Bind independent ownership proof to actual emitted CUDA peer/header code."""
    source, output = Path(source), Path(output)
    layout = Layout(warp_y, 32 // warp_y)
    text = (source / "src/asymmetric_native.cuh").read_text()
    begin = text.index("__device__ __forceinline__ void PermanentFamilyBody")
    end = text.index("__device__ __forceinline__ void UniformBody", begin)
    body = text[begin:end]
    wy, wx = layout.warp_y, layout.warp_x
    mapping = f"const int sx=((threadIdx.x/32)&{wy-1})*{wx}+(lane&{wx-1}),sy=(threadIdx.x/{32*wy})*{wy}+(lane/{wx}),zbase=0;"
    if mapping not in body or "constexpr int T=512,Slots=8,ZStep=1;" not in body:
        raise ValueError("Actual CUDA point/lane/Z ownership differs from independent proof")
    matches = re.findall(r"const float (\w+)=__shfl_sync\(0xffffffffu,([eh])\[(\d)\]\[j\],([^;]+)\);", body)
    expected = {"ex_y": ("e", "0", 1, 1), "ez_y": ("e", "2", 1, 1),
                "ey_x": ("e", "1", 2, 1), "ez_x": ("e", "2", 2, 1),
                "hx_y": ("h", "0", 1, -1), "hz_y": ("h", "2", 1, -1),
                "hy_x": ("h", "1", 2, -1), "hz_x": ("h", "2", 2, -1)}
    if len(matches) != 8 or body.count("__shfl_sync(") != 8 or {r[0] for r in matches} != set(expected):
        raise ValueError("Exactly eight unconditional full-mask field peer shuffles required")
    for name, family, component, expression in matches:
        want_family, want_component, axis, delta = expected[name]
        if (family, component) != (want_family, want_component):
            raise ValueError("Actual shuffle field input differs from traced read")
        for lane in range(32):
            y, x = divmod(lane, wx)
            y = (y + (delta if axis == 1 else 0)) % wy
            x = (x + (delta if axis == 2 else 0)) % wx
            if eval_peer(expression, lane) != y * wx + x:
                raise ValueError(f"Actual C++ shuffle peer differs:{name}/lane{lane}")
        guard = "if(zs!=hz&&sy!=hy&&sx!=hx)" if family == "e" else "if(zs!=ehz&&sy!=ehy&&sx!=ehx)"
        if body.index("const float " + name + "=") > body.index(guard):
            raise ValueError("Full-mask shuffle moved after divergent owned predicate")
    model = publication_model(layout)
    if f"PermanentFamilySharedBytes=={model['shared_total_bytes']}" not in text:
        raise ValueError("Actual total shared allocation differs from traced face count")
    header_path = source / "src/warp_frontier_layout_generated.cuh"
    header = header_path.read_text()
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "actual-frontier-index.cc", output / "actual-frontier-index"
    program = '#define __device__\n#define __forceinline__ inline\n#include <cstdio>\n#include "warp_frontier_layout_generated.cuh"\nint main(){using namespace beamz::cuda::moving_native;for(int f=0;f<6;++f)for(int z=0;z<8;++z)for(int y=0;y<16;++y)for(int x=0;x<32;++x)std::printf("%d\\n",WarpFrontierIndex(f,z,y,x));}\n'
    cpp.write_text(program)
    command = ["c++", "-std=c++17", "-O2", "-I" + str(source / "src"), str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    values = np.fromstring(run.stdout, dtype=np.int64, sep="\n").reshape(6, 8, 16, 32)
    _, reads = canonical_reads()
    requirements = required_axes(reads, layout)
    occupied = set()
    for f, field in enumerate(FIELDS):
        expected_points = footprint(field, layout, requirements)
        for point in itertools.product(*(range(s) for s in SHAPE)):
            index = int(values[(f, *point)])
            if (index >= 0) != (point in expected_points):
                raise ValueError("Actual packed shared header misses/adds a traced producer")
            if index >= 0:
                if index in occupied or index >= model["shared_field_words"]:
                    raise ValueError("Actual shared producer index aliases or exceeds allocation")
                occupied.add(index)
    if occupied != set(range(model["shared_field_words"])):
        raise ValueError("Actual shared bank includes holes or missing producers")
    result = {"schema": "beamz.actual-warp-layout-lowering-audit.v1", "passed": True,
              "layout": asdict(layout), "actual_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
              "actual_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "actual_generated_header_sha256": hashlib.sha256(header.encode()).hexdigest(),
              "shuffle_peer_cases": 8 * 32, "header_index_cases": 6 * 8 * 16 * 32,
              "shared_producer_words_exactly_once": len(occupied), "shared_total_bytes": model["shared_total_bytes"],
              "command": command, "no_CUDA_execution": True}
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def instruction_points(layout=None):
    """128 scalar-plane warp instructions cover4096 cache points exactly."""
    if layout is None:
        # Selected NoIO/bulk1024-thread interleaved4Z-slot mapping.
        points = [[(tid // 512 + 2 * j, (tid // 32) % 16, tid % 32)
                   for tid in range(w * 32, (w + 1) * 32)]
                  for j in range(4) for w in range(32)]
    else:
        points = [[layout.point(tid, z) for tid in range(w * 32, (w + 1) * 32)]
                  for z in range(8) for w in range(16)]
    return np.asarray(points, np.int64)


def sector_summary(words, valid=None, tags=None):
    """Exact ideal sector sets per provided scalar warp instruction group."""
    words = np.asarray(words, np.int64)
    if valid is None:
        valid = np.ones(words.shape, bool)
    if tags is None:
        tags = np.zeros(words.shape, np.int64)
    encoded_words = np.where(valid, words + (np.asarray(tags, np.int64) << 40), -1)
    encoded_sectors = np.where(valid, (words >> 3) + (np.asarray(tags, np.int64) << 40), -1)

    def counts(values):
        sorted_values = np.sort(values, axis=1)
        changed = np.concatenate((np.ones((values.shape[0], 1), bool), sorted_values[:, 1:] != sorted_values[:, :-1]), axis=1)
        return np.sum((sorted_values >= 0) & changed, axis=1)

    unique_words = counts(encoded_words)
    sectors = counts(encoded_sectors)
    return {"active_lane_accesses": int(np.sum(valid)), "distinct_warp_word_accesses": int(unique_words.sum()),
            "issued_sector_lower_bound": int(sectors.sum()),
            "sector_floor_if_each_instruction_were_contiguous": int(np.sum((unique_words + 7) // 8)),
            "CTA_unique_sector_addresses": int(np.unique(encoded_sectors[valid]).size),
            "active_warp_instructions": int(np.count_nonzero(unique_words))}


def surface(z, y, x):
    return np.where(z == 6, y * 31 + x,
                    np.where(y == 14, 15 * 31 + z * 31 + x,
                             np.where(x == 30, 15 * 31 + 6 * 31 + z * 14 + y, -1)))


def trail_sectors(layout, version, field, *, imports=False):
    points = instruction_points(layout)
    relative = (points + version + 1) % np.asarray(SHAPE)
    z, y, x = (relative[..., a] for a in range(3))
    if imports:
        valid = (z == 0) | (y == 0)
        tags = (z == 0).astype(np.int64) | ((y == 0).astype(np.int64) << 1) | ((x == 0).astype(np.int64) << 2)
        offset = surface(np.where(z > 0, z - 1, 6), np.where(y > 0, y - 1, 14), np.where(x > 0, x - 1, 30))
    else:
        valid = (z > 0) & (y > 0) & (x > 0)
        tags = None
        offset = surface(z - 1, y - 1, x - 1)
        valid &= offset >= 0
    words = (version - 1) * 6 * SURFACE + field * SURFACE + offset
    return sector_summary(words, valid, tags)


def low_x_stage_sectors(version, field):
    index = np.arange(128).reshape(4, 32)
    z, y, x = index // 15, index % 15, np.full(index.shape, 30)
    words = (version - 1) * 6 * SURFACE + field * SURFACE + surface(z, y, x)
    return sector_summary(words, index < LOW_X)


def field_sectors(layout, field_shape, origin, *, version=0, owned=False, packed_codes=False):
    points = instruction_points(layout)
    relative = (points + version + 1) % np.asarray(SHAPE)
    physical_points = np.asarray(origin) - version - 1 + relative
    z, y, x = (physical_points[..., a] for a in range(3))
    valid = np.all((physical_points >= 0) & (physical_points < np.asarray(field_shape)), axis=2)
    if owned:
        valid &= np.all(relative > 0, axis=2)
    words = (z * field_shape[1] + y) * field_shape[2] + x
    if packed_codes:
        words >>= 2
    return sector_summary(words, valid)


def psi_sectors(layout, domain, family, term, origin, version):
    """One mutable FP32ψ scalar load OR store at actual CPML12 coordinates."""
    axis = (1, 0, 0, 2, 2, 1)[term]
    target = tuple(domain + d for d in SUPPORT_DELTA[3 * family + term // 2])
    points = instruction_points(layout)
    relative = (points + version + 1) % np.asarray(SHAPE)
    coordinates = np.asarray(origin) - version - 1 + relative
    valid = np.all(relative > 0, axis=2)
    valid &= np.all((coordinates >= 0) & (coordinates < np.asarray(target)), axis=2)
    derivative_extent = domain + family
    normal = coordinates[..., axis]
    active = (normal < 12) | (normal >= derivative_extent - 12)
    valid &= active
    packed = np.array(coordinates, copy=True)
    packed[..., axis] = np.where(normal < 12, normal, 24 + normal - derivative_extent)
    slab_shape = list(target)
    slab_shape[axis] = 24
    words = (packed[..., 0] * slab_shape[1] + packed[..., 1]) * slab_shape[2] + packed[..., 2]
    return sector_summary(words, valid)


def cpml_coalescing_model(layout, domains=(512, 564)):
    """All low/high axis tile origins touching actual activeψ at each phase."""
    result = []
    for domain in domains:
        rows = {axis: [] for axis in range(3)}
        for family, term, version in itertools.product(range(2), range(6), range(1, 17)):
            axis = (1, 0, 0, 2, 2, 1)[term]
            extent = domain + family
            for b in range((domain + 16 + OWNED[axis] - 1) // OWNED[axis]):
                origin = b * OWNED[axis]
                low, high = origin - version, origin - version + OWNED[axis] - 1
                low_active = max(low, 0) <= min(high, 11)
                high_active = max(low, extent - 12) <= min(high, extent - 1)
                if not low_active and not high_active:
                    continue
                point = [56, 60, 124]
                point[axis] = origin
                rows[axis].append(psi_sectors(layout, domain, family, term, point, version))
        result.append({"domain": domain,
                       "per_derivative_axis": {"zyx"[axis]: sum_summaries(values) for axis, values in rows.items()},
                       "all12terms_low_high_origins_load_OR_store": sum_summaries([r for values in rows.values() for r in values]),
                       "scope": "One load OR one store of each active packedFP32ψ; actualphase extent N+family andCPML12; transverse origin(56,60,124), normal origins vary over all active low/high tiles"})
    return result


def sum_summaries(rows):
    result = {key: sum(r[key] for r in rows) for key in rows[0]}
    # CTAunique addresses may repeat across sample contexts; this is a sum, not
    # the union of all sectors in an entire simulation or a DRAM traffic count.
    return result


def coalescing_model(layout, domains=(512, 564), origin_samples=8):
    contexts = []
    for n in domains:
        startup, final, packed = [], [], []
        for by, bx in itertools.product(range(origin_samples), repeat=2):
            origin = (56, (4 + by) * 15, (4 + bx) * 31)
            for c, delta in enumerate(SUPPORT_DELTA):
                shape = tuple(n + d for d in delta)
                startup.append(field_sectors(layout, shape, origin))
                final.append(field_sectors(layout, shape, origin, version=16, owned=True))
                if c >= 3:
                    packed.append(field_sectors(layout, shape, origin, version=16, owned=True, packed_codes=True))
        contexts.append({"domain": n, "actual_tile_origin_samples": origin_samples ** 2,
                         "startup_FP32_field_loads": sum_summaries(startup),
                         "final_owned_FP32_field_stores": sum_summaries(final),
                         "owned_byte_packed_E_code_word_loads": sum_summaries(packed)})
    pub = [trail_sectors(layout, n, c) for n in range(1, 17) for c in range(6)]
    imp = [trail_sectors(layout, n, c, imports=True) for n in range(1, 17) for c in range(6)]
    stage = [low_x_stage_sectors(n, c) for n in range(1, 17) for c in range(6)]
    return {"layout": "selected1024thread32laneX" if layout is None else asdict(layout),
            "dense_field_contexts": contexts, "16step_all6_trail_publication": sum_summaries(pub),
            "16step_all6_ZY_ghost_import": sum_summaries(imp), "16step_all6_coalesced_lowX_stage": sum_summaries(stage),
            "packed_mutable_CPML12_psi": cpml_coalescing_model(layout, domains),
            "notes": ["Exact distinct32B sectors of the modeled scalar warp instructions; not measured memory traffic or time",
                      "Different neighbor trail allocations are distinct tags; allocation strides16*6*735FP32 are32B aligned",
                      "Component735FP32 strides and timestep4410FP32 strides retain their actual sector misalignment",
                      "Dense scalar field pitches use actual staggered N/N+1 supports; origins follow owned strides7/15/31",
                      "Byte-packed E code words count sample_index>>2; data-dependent table lookups are omitted",
                      "Compiler lowering, cache reuse, replay, packed table lookup, support guards and kernel overlap require hardware measurements"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--origin-samples", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if not 1 <= a.origin_samples <= 8:
        p.error("Use1..8 actual interior tile-origin samples per axis")
    layouts = [Layout(*shape) for shape in LAYOUTS]
    proofs = [prove_routes(layout, depth=2 if a.quick else 16) for layout in layouts]
    negatives = []
    for mutation in ("opposite-face", "omit-union-intersection", "wrong-forward-version"):
        try:
            prove_routes(Layout(4, 8), negative=mutation)
        except AssertionError as error:
            negatives.append({"mutation": mutation, "rejected": True, "witness": str(error)})
        else:
            raise AssertionError("Unsafe route escaped:" + mutation)
    result = {"schema": "beamz.traced-warp-layout-cost.v1", "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "publication_models": [publication_model(layout) for layout in layouts], "route_proofs": proofs,
              "negative_controls": negatives, "coalescing_models": [coalescing_model(layout, origin_samples=a.origin_samples) for layout in (None, *layouts)],
              "GPU_executed": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "actual_route_checks": sum(r["actual_read_checks"] for r in proofs),
                      "layouts": [(m["layout"], m["shared_total_bytes"]) for m in result["publication_models"]]}))


if __name__ == "__main__":
    main()
