"""Source-backed analytic payload/resource model; no GPU or measured DRAM claim.

Counts requested scalar accesses at the traced-cell interface. Compiler CSE,
cache reuse, coalescing, sectors, metadata and spills change actual DRAM traffic.
Exact finite-domain counts factor rectangular intervals across the tile grid.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

try:
    from scripts.cuda_stencil_trace import trace_yee_step
    from scripts.prove_cuda_systolic_schedule import component_shapes
except ModuleNotFoundError:
    from cuda_stencil_trace import trace_yee_step
    from prove_cuda_systolic_schedule import component_shapes


SOURCES = {
    "paper": "https://arxiv.org/pdf/2502.20610",
    "sm86": "https://docs.nvidia.com/cuda/ampere-tuning-guide/",
    "h100_specs": "https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306",
}
TILES = {"probe": (4, 8, 32), "balanced": (8, 8, 16),
         "paper_active": (15, 15, 31), "sm86_power2_cache": (7, 7, 15),
         "half_shared": (15, 15, 15), "asymmetric": (5, 15, 31)}


def geometry(tile):
    v = math.prod(tile)
    edge = sum(tile)
    area = sum(v // b for b in tile)
    outgoing = v - math.prod(b - 1 for b in tile)
    padded = math.prod(b + 1 for b in tile)
    low = padded - v
    # All formulas count an interior tile, away from physical walls.
    requested = 12 * outgoing + 6 * area - 2 * edge
    unique = 6 * outgoing + 4 * edge - 6 + 2 * area
    published = 6 * outgoing
    broad = 6 * low + 3 * area
    tangential = 6 * low + 2 * area
    circular = 6 * low
    return {
        "owned_zyx": tile, "cache_zyx": tuple(b + 1 for b in tile),
        "owned_cells": v, "padded_cells": padded, "retained_intersection": v,
        "outgoing_high_union": outgoing, "incoming_low_union": low,
        "face_area_sum": area, "edge_length_sum": edge,
        "requested_global_read_floats": requested,
        "unique_incoming_field_version_positions": unique,
        "publication_floats": published,
        "broad_face_read_floats": broad,
        "tangential_face_read_floats": tangential,
        "circular_current_face_read_floats": circular,
        "requested_field_bytes_per_update": 4 * (requested + published) / v,
        "unique_field_payload_bytes_per_update": 4 * (unique + published) / v,
        "broad_field_payload_bytes_per_update": 4 * (broad + published) / v,
        "tangential_field_payload_bytes_per_update": 4 * (tangential + published) / v,
        "circular_field_payload_bytes_per_update": 4 * (circular + published) / v,
        "owned_six_fp32_shared_bytes": 24 * v,
        "padded_six_fp32_shared_bytes": 24 * padded,
        "half_shared_one_family_bytes": 12 * padded,
        "permanent_shared_xz_bytes": 16 * padded,
        "y_component_state_scalars_at512": 2 * math.ceil(padded / 512),
        "half_shared_state_register_lower_bound_at256": 6 * math.ceil(padded / 256),
        "half_shared_state_register_lower_bound_at512": 6 * math.ceil(padded / 512),
        "padded_six_field_six_material_fp32_bytes": 48 * padded,
        "padded_fp32_fields_plus_three_byte_codes": 27 * padded,
        "padded_six_fields_plus_twelve_psi_fp32_bytes": 72 * padded,
    }


def verify_interior(tile):
    g = geometry(tile)
    requests = 0
    unique = set()
    phases = {"H": {"requests": 0, "unique": set()}, "E": {"requests": 0, "unique": set()}}
    for update in trace_yee_step():
        phase = update.output.field[0]
        for p in itertools.product(*(range(n) for n in tile)):
            for read in update.reads:
                q = tuple(a + b + read.version - 1 for a, b in zip(p, read.offset, strict=True))
                if any(a < 0 or a >= b for a, b in zip(q, tile, strict=True)):
                    key = read.field, read.version, q
                    requests += 1
                    unique.add(key)
                    phases[phase]["requests"] += 1
                    phases[phase]["unique"].add(key)
    assert requests == g["requested_global_read_floats"]
    assert len(unique) == g["unique_incoming_field_version_positions"]
    return {"requests": requests, "unique": len(unique),
            "phases": {p: {"requests": q["requests"], "unique": len(q["unique"])} for p, q in phases.items()}}


def grid_for(n, steps, tile):
    return tuple((n + steps + b) // b for b in tile)


def axis_sum(extent, b, count, version, lo=0, hi=None):
    hi = b if hi is None else hi
    return sum(max(0, min(hi, version + extent - o) - max(lo, version - o))
               for o in range(0, b * count, b))


def domain(shape, tile, grid, version, lo=0, high_delta=0):
    return math.prod(axis_sum(e, b, count, version, lo, b + high_delta)
                     for e, b, count in zip(shape, tile, grid, strict=True))


def read_request_count(dst, src, delta, relative, tile, grid, n, initial=False):
    valid_sums, local_sums = [], []
    for d, s, shift, rel, b, count in zip(dst, src, delta, relative, tile, grid, strict=True):
        valid = local = 0
        for o in range(0, b * count, b):
            low = max(0, n - o, n - o - shift)
            high = min(b, n + d - o, n + s - o - shift)
            valid += max(0, high - low)
            local += max(0, min(high, b - rel) - max(low, -rel))
        valid_sums.append(valid)
        local_sums.append(local)
    return math.prod(valid_sums) - (0 if initial else math.prod(local_sums))


def pure_faces(shape, tile, grid, version, component, tangential=False):
    result = 0
    for axis in range(3):
        if tangential and component == 2 - axis:
            continue
        counts = []
        for a, (extent, b, count) in enumerate(zip(shape, tile, grid, strict=True)):
            if a == axis:
                counts.append(sum(0 <= o - 1 - version < extent for o in range(0, b * count, b)))
            else:
                counts.append(axis_sum(extent, b, count, version))
        result += math.prod(counts)
    return result


def rank_histogram(grid):
    hist = [1]
    for length in grid:
        next_hist = []
        running = 0
        for i in range(len(hist) + length - 1):
            if i < len(hist):
                running += hist[i]
            if 0 <= i - length < len(hist):
                running -= hist[i - length]
            next_hist.append(running)
        hist = next_hist
    assert sum(hist) == math.prod(grid)
    return hist


def finite_case(n, steps, tile, bandwidth_gbs, material_channels=6):
    shapes = component_shapes((n, n, n))
    fields = tuple("H" + c for c in "xyz") + tuple("E" + c for c in "xyz")
    grid = grid_for(n, steps, tile)
    histogram = rank_histogram(grid)
    volume = math.prod(tile)
    count = math.prod(grid)
    requests = initial_requests = valid_publications = old_halo = current_all_h = current_tangential_h = current_halo = 0
    padded_initial = sum(domain(shapes[f], tile, grid, 0, lo=-1) for f in fields)
    owned_initial = sum(domain(shapes[f], tile, grid, 0) for f in fields)
    equations = trace_yee_step()
    for nstep in range(1, steps + 1):
        for update in equations:
            for read in update.reads:
                rel = tuple(a + read.version - 1 for a in read.offset)
                value = read_request_count(shapes[update.output.field], shapes[read.field],
                                           read.offset, rel, tile, grid, nstep,
                                           initial=nstep == 1 and read.version == 0)
                requests += value
                initial_requests += value if nstep == 1 and read.version == 0 else 0
        for c, f in enumerate(fields):
            owned = domain(shapes[f], tile, grid, nstep)
            valid_publications += owned - domain(shapes[f], tile, grid, nstep, high_delta=-1)
            old_halo += domain(shapes[f], tile, grid, nstep - 1, lo=-1) - domain(shapes[f], tile, grid, nstep - 1)
            current_halo += domain(shapes[f], tile, grid, nstep, lo=-1) - owned
            if c < 3:
                current_all_h += pure_faces(shapes[f], tile, grid, nstep, c)
                current_tangential_h += pure_faces(shapes[f], tile, grid, nstep, c, tangential=True)
    final_stores = sum(math.prod(shapes[f]) for f in fields)
    # Actual prototype publishes zero-filled INVALID physical slots too.
    # Ring reuse reduces allocation size, not this number of stores.
    publications = count * steps * 6 * geometry(tile)["outgoing_high_union"]
    base_updates = n ** 3 * steps
    broad_requests = owned_initial + old_halo + current_all_h
    tangent_requests = owned_initial + old_halo + current_tangential_h
    circular_requests = padded_initial + current_halo
    material_initial = domain((n, n, n), tile, grid, 0, lo=-1)
    material_faces = sum(domain((n, n, n), tile, grid, v, lo=-1) - domain((n, n, n), tile, grid, v)
                         for v in range(1, steps + 1))
    e_code_payload = sum(domain(shapes["E" + c], tile, grid, 0, lo=-1)
                         + sum(domain(shapes["E" + c], tile, grid, v, lo=-1)
                               - domain(shapes["E" + c], tile, grid, v)
                               for v in range(1, steps + 1)) for c in "xyz")
    scenarios = {}
    for name, reads in (("current_direct_requested", requests), ("broad_face_requested", broad_requests),
                        ("tangential_face_requested", tangent_requests), ("circular_face_requested", circular_requests)):
        byte_count = 4 * (reads + publications + final_stores)
        scenarios[name] = {"field_read_floats": reads, "payload_bytes": byte_count,
                           "bytes_per_basecell_step": byte_count / base_updates,
                           "all_payload_at_assumed_bandwidth_ms": byte_count / (bandwidth_gbs * 1e6)}
    resident_cells = math.prod(b + 1 for b in tile)
    g = geometry(tile)
    max_frontier = max(histogram)
    direct_trail = count * steps * 6 * g["outgoing_high_union"] * 4
    ring_trail = 4 * max_frontier * steps * 6 * g["outgoing_high_union"] * 4
    four_field_banks = 4 * 6 * (n + 1) ** 3 * 4
    two_field_banks = four_field_banks // 2
    cpml_axes = (1, 0, 0, 2, 2, 1)
    packed_psi_bank = 4 * sum(24 * math.prod(shapes[family + "xyz"[t // 2]][a]
                                             for a in range(3) if a != axis)
                              for family in "HE" for t, axis in enumerate(cpml_axes))
    byte_code_arrays = sum(math.prod(shapes["E" + c]) for c in "xyz")
    # Exact envelope activity, separable by version; includes all six fields'
    # maximal N+1 cube. It is an overapproximation of component-specific activity.
    active_box_versions = 0
    for v in range(1, steps + 1):
        active_box_versions += math.prod(sum(o <= n + v and o + b - 1 >= v
                                             for o in range(0, b * num, b))
                                         for b, num in zip(tile, grid, strict=True))
    return {
        "n": n, "steps": steps, "tile_zyx": tile, "grid_zyx": grid,
        "boxes": count, "rank_launches": len(histogram), "max_rank_frontier": max_frontier,
        "cta_wave_efficiency_at_82_sms_one_cta": count / (82 * sum(math.ceil(v / 82) for v in histogram)),
        "valid_component_updates": final_stores * steps,
        "visited_component_slots": count * steps * volume * 6,
        "visited_over_valid": count * steps * volume * 6 / (final_stores * steps),
        "active_envelope_box_versions": active_box_versions,
        "all_box_versions": count * steps,
        "initial_requested_global_floats": initial_requests,
        "circular_initial_field_load_floats": padded_initial,
        "circular_current_halo_load_floats": current_halo,
        "circular_startup_final_bytes": 4 * (padded_initial + final_stores),
        "publication_floats": publications, "final_field_stores": final_stores,
        "valid_publication_floats": valid_publications,
        "invalid_zero_publication_bytes": 4 * (publications - valid_publications),
        "scenarios": scenarios,
        "material_fp32_load_only_payload_bytes": 4 * material_channels * (material_initial + material_faces),
        "three_byte_material_code_payload_bytes": e_code_payload,
        "six_e_coefficient_fp32_load_only_payload_bytes": 2 * 4 * e_code_payload,
        "direct_trail_bytes": direct_trail, "four_rank_trail_bytes": ring_trail,
        "four_dense_field_banks_bytes": four_field_banks,
        "two_dense_field_banks_bytes": two_field_banks,
        "harness_payload_direct_bytes": four_field_banks + direct_trail,
        "harness_payload_ring_bytes": four_field_banks + ring_trail,
        "two_bank_solver_ring_bytes": two_field_banks + ring_trail,
        "owned_one_field_bank_ring_bytes": two_field_banks // 2 + ring_trail,
        "packed_cpml12_psi_bytes_per_bank": packed_psi_bank,
        "global_psi_inplace_payload_bytes_per_macro": 2 * packed_psi_bank * steps,
        "global_psi_inplace_bytes_per_basecell_step": 2 * packed_psi_bank / n ** 3,
        "three_e_byte_code_array_bytes": byte_code_arrays,
        "two_field_two_psi_banks_codes_ring_bytes_excluding_monitors": two_field_banks + 2 * packed_psi_bank + byte_code_arrays + ring_trail,
        "owned_one_field_one_psi_codes_ring_bytes_excluding_monitors": two_field_banks // 2 + packed_psi_bank + byte_code_arrays + ring_trail,
        "ring_harness_under_24gib_with_256mib_reserve": four_field_banks + ring_trail + 256 * 2 ** 20 < 24 * 2 ** 30,
        "padded_fp32_field_bytes": resident_cells * 24,
        "warning": "Payload estimates, NOT measured DRAM; no CPML/source/DFT/metadata/spill/sector traffic included",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=(256, 384, 512, 640))
    p.add_argument("--steps", type=int, nargs="+", default=(8,))
    p.add_argument("--bandwidth-gbs", type=float, default=700.0,
                   help="Explicit assumed payload bandwidth; not a hardware measurement or runtime prediction")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(*a.sizes, *a.steps, a.bandwidth_gbs) <= 0:
        p.error("Positive dimensions/steps/bandwidth required")
    audited = Path(__file__).resolve().parents[1] / "scripts/cuda_moving_box_probe.cu"
    repo = Path(__file__).resolve().parents[1]
    source_paths = ("scripts/cuda_moving_box_probe.cu", "scripts/cuda_moving_face_cache.cuh",
                    "scripts/cuda_moving_circular_cache.cuh", "scripts/cuda_stencil_trace.py",
                    "cuda/src/yee_cell_generated.cuh")
    result = {"scope": __doc__, "sources": SOURCES,
              "audited_cuda_sha256": hashlib.sha256(audited.read_bytes()).hexdigest(),
              "audited_sources_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                                         for name in source_paths if (repo / name).is_file()},
              "assumed_payload_bandwidth_gbs": a.bandwidth_gbs,
              "material_float_channels_assumption": 6,
              "geometry": {name: geometry(tile) for name, tile in TILES.items()},
              "interior_ir_enumeration": {name: verify_interior(tile) for name, tile in TILES.items()},
              "cases": [finite_case(n, t, tile, a.bandwidth_gbs) for n in a.sizes for t in a.steps for tile in TILES.values()]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(case["n"], case["steps"], case["tile_zyx"],
              round(case["scenarios"]["current_direct_requested"]["bytes_per_basecell_step"], 4),
              round(case["scenarios"]["circular_face_requested"]["bytes_per_basecell_step"], 4),
              round(case["harness_payload_ring_bytes"] / 2 ** 30, 3))


if __name__ == "__main__":
    main()
