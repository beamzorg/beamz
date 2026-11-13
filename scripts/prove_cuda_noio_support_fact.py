"""Trace-derived Cartesian proof of the native bit32 support certificate.

Enumeration is one dimensional per axis because all six actual supports are
Cartesian products. It visits every cache slot/head, phase/read, initial load,
and possible packet coordinate; no sampled 3-D corners substitute for reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry
from scripts.cuda_space_time_ir import FIELDS, validate_stream_signature, yee_operator
from scripts.cuda_stencil_trace import serialize_updates
from scripts.patch_cuda_noio_support_fact import PARENT, transformed


CACHE = (8, 16, 32)
OWNED = tuple(x - 1 for x in CACHE)


def axis_certificate(origin, owned, depth, minimum):
    low, high = origin - depth, origin + owned - 2
    return low > 0 and high + 1 < minimum


def prove(sizes=None, depths=(4, 8, 16, 32)):
    sizes = tuple(range(25, 97)) + (136, 272, 512, 564, 680) if sizes is None else tuple(sizes)
    operator = yee_operator(cpml_axes=(0, 1, 2))
    validate_stream_signature(operator)
    geometry = derive_geometry(operator)
    updates = [u for u in operator.updates if u.output.field in FIELDS]
    reads = {a: sorted({(operator.specs[u.output.field].phase, FIELDS.index(u.output.field), FIELDS.index(r.field), r.offset[a])
                        for u in updates for r in u.reads if r.field in FIELDS}) for a in range(3)}
    counts = {key: 0 for key in ("safe_axis_tiles", "startup_component_checks", "owned_component_checks", "traced_field_read_checks", "import_component_checks", "native_wall_checks", "safe_CPML_axis_tiles", "bit32_negative_counterexamples", "wrong_low_comparison_counterexamples", "wrong_high_comparison_counterexamples")}
    phases = set()
    heads = {a: set() for a in range(3)}
    for minimum in sizes:
        for depth in depths:
            for axis, (cache, owned) in enumerate(zip(CACHE, OWNED, strict=True)):
                # Real grid origins plus exact symbolic comparison boundaries.
                # The latter need not be multiples of B for the current four
                # depths; label them explicitly rather than misreporting them
                # as reachable native tile counterexamples.
                origins = sorted(set(range(0, minimum + depth + owned, owned)) | {depth, minimum - owned + 1})
                for origin in origins:
                    low, high = origin - depth, origin + owned - 2
                    safe = axis_certificate(origin, owned, depth, minimum)
                    # Real host startup and final imported envelopes. Clearing
                    # bit32 unconditionally or weakening either comparison
                    # admits points outside at least one actual component.
                    bad_support = origin - depth - 1 < 0 or origin + owned - 1 >= minimum
                    if not safe:
                        counts["bit32_negative_counterexamples"] += bad_support
                        counts["wrong_low_comparison_counterexamples"] += (low >= 0 and high + 1 < minimum and bad_support)
                        counts["wrong_high_comparison_counterexamples"] += (low > 0 and high < minimum and bad_support)
                        continue
                    counts["safe_axis_tiles"] += 1
                    counts["safe_CPML_axis_tiles"] += low < 12 or high >= minimum - 12
                    # The proof uses minimum actual component extents, so the
                    # other two dimensions may be independent rectangular N.
                    assert 0 <= origin - depth - 1 <= low <= high < origin + owned - 1 <= minimum - 1
                    # A safe tile cannot start late: origin < minimum and the
                    # actual window first=max(1,origin-(minimum+1)+1)=1.
                    assert max(1, origin - (minimum + 1) + 1) == 1
                    for slot in range(cache):
                        coordinate = origin + (-1 if slot == cache - 1 else slot)
                        for field in range(6):
                            assert 0 <= coordinate < minimum + SUPPORT_DELTA[field][axis]
                            counts["startup_component_checks"] += 1
                    for version in range(1, depth + 1):
                        head = (-version - 1) & (cache - 1)
                        heads[axis].add(head)
                        for slot in range(cache):
                            relative = (slot - head) & (cache - 1)
                            coordinate = origin - version - 1 + relative
                            # Includes all union ghosts and all stage-low-X
                            # packet coordinates, independent of which axis
                            # caused the ghost membership in full 3-D.
                            for field in range(6):
                                assert 0 <= coordinate < minimum + SUPPORT_DELTA[field][axis]
                                counts["import_component_checks"] += 1
                            if slot == head:
                                continue
                            assert low <= coordinate <= high
                            for field in range(6):
                                assert 0 <= coordinate < minimum + SUPPORT_DELTA[field][axis]
                                counts["owned_component_checks"] += 1
                            for phase, output, source, offset in reads[axis]:
                                phases.add(phase)
                                assert 0 <= coordinate < minimum + SUPPORT_DELTA[output][axis]
                                neighbor = coordinate + offset
                                assert 0 <= neighbor < minimum + SUPPORT_DELTA[source][axis]
                                counts["traced_field_read_checks"] += 1
                                if phase and offset:
                                    assert coordinate != 0 and coordinate != minimum
                                    counts["native_wall_checks"] += 1
    assert phases == {0, 1}
    assert all(heads[a] == set(range(CACHE[a])) for a in range(3))
    assert all(counts[k] > 0 for k in counts)
    return {"schema": "beamz.native-interior-cartesian-support-proof.v1", **counts,
            "dimensions_tested": list(sizes), "depths_tested": list(depths),
            "axis_cache_heads": {str(a): sorted(v) for a, v in heads.items()},
            "phase_read_classes_per_axis": {str(a): len(v) for a, v in reads.items()},
            "origin_enumeration": "all actual multiples of B plus symbolic origins depth and N-B+1; wrong-comparison negatives may be off-grid, bit32 negatives include actual grid boundaries",
            "canonical_field_support_delta": SUPPORT_DELTA,
            "operator_sha256": hashlib.sha256(json.dumps(serialize_updates(operator.updates), sort_keys=True).encode()).hexdigest(),
            "geometry_sha256": geometry["geometry_sha256"],
            "universal_integer_argument": {
                "host": "low=origin-depth>0 AND high+1=origin+B-1<minimum",
                "owned": "for 1<=n<=depth and 1<=relative<=B: coordinate=origin-n-1+relative in[low,high], hence0<coordinate<minimum-1",
                "startup": "slot==B?-1:slot gives coordinate in[origin-1,origin+B-1] subset[0,minimum-1]",
                "ghost_packets": "0<=relative<=B gives coordinate in[origin-depth-1,origin+B-2] subset[0,minimum-1]",
                "trace_reads": "actual axial offsets0,+1(H),-1(E), so every source in[low-1,high+1] subset[0,minimum-1]",
                "Cartesian_product": "each actual canonical field support is product[0,N_axis+delta), delta>=0; independent axis proofs establish full rectangular six-field support",
                "native_E_walls": "owned coordinate strictly between0 andN_axis, so original wall guard never fires",
                "CPML": "certificate allows low<12 or high>=N-12; local state/profile/ordered FP operations remain unchanged"},
            "errors": 0, "GPU_executed": False,
            "scope": "universal integer envelope argument plus exhaustive separable actual cache heads/reads/ghosts on listed sizes; no runtime speed claim"}


def validate_source(source, output, sizes=None):
    source, output = Path(source), Path(output)
    files = {name: (source / "src" / name).read_bytes() for name in PARENT}
    # Bind the actual emitted source to the exact parent transformation while
    # retaining all other real ABI/admission/host-certificate file hashes.
    original = dict(files)
    original["asymmetric_native.cuh"] = (Path(__file__).resolve().parents[1] / ".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh").read_bytes()
    actual, transformation = transformed(original)
    if files != actual:
        raise ValueError("Actual support-fact source or bound ABI differs")
    report = prove(sizes)
    report.update(result_hashes={n: hashlib.sha256(v).hexdigest() for n, v in files.items()},
                  transformation=transformation, producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(validate_source(a.source, a.output), indent=2))
