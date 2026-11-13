"""Actual C++ scalar-mux bits plus traced NoIO ownership/version oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import FIELDS, validate_stream_signature, yee_operator
from scripts.cuda_stencil_trace import serialize_updates
from scripts.patch_cuda_noio_mux import HELPERS, PARENT, mux_body, transformed
from scripts.prove_cuda_half_y_noio import Hazard, Topology


HARNESS = r'''
#include <bit>
#include <cstdint>
#include <cstdio>
#define __device__
#define __forceinline__ inline
@HELPERS@
int main(){
 size_t select_checks=0,update_checks=0,signed_zero_checks=0,nan_checks=0,wrong_select=0,wrong_update=0,errors=0;
 const uint32_t special[]={0u,0x80000000u,0x7f800000u,0xff800000u,0x7fc00001u,0xffc00001u,0x7f800001u,0xff800001u,1u,0x80000001u,0x3f800000u,0xbf800000u};
 for(uint32_t seed=0;seed<4096;++seed){float bank[4];
  for(int k=0;k<4;++k){uint32_t bits=seed<12?special[(seed+k)%12]:(seed+uint32_t(k+1)*0x5319u)*2654435761u;bank[k]=std::bit_cast<float>(bits);}
  for(int j=0;j<4;++j){const uint32_t want=std::bit_cast<uint32_t>(bank[j]);
   errors+=std::bit_cast<uint32_t>(NoIOSelect4(bank,j))!=want;++select_checks;
   signed_zero_checks+=(want&0x7fffffffu)==0;nan_checks+=(want&0x7fffffffu)>0x7f800000u;
   wrong_select+=std::bit_cast<uint32_t>(NoIOSelect4(bank,(j+1)&3))!=want;
   const uint32_t newbits=want^0x80000001u;const float value=std::bit_cast<float>(newbits);
   float actual[4],bad[4];for(int k=0;k<4;++k)actual[k]=bad[k]=bank[k];
   NoIOUpdate4(actual,j,value);NoIOUpdate4(bad,(j+1)&3,value);
   for(int k=0;k<4;++k){const uint32_t expected=k==j?newbits:std::bit_cast<uint32_t>(bank[k]);
    errors+=std::bit_cast<uint32_t>(actual[k])!=expected;++update_checks;
    wrong_update+=std::bit_cast<uint32_t>(bad[k])!=expected;
   }
  }
 }
 errors+=!wrong_select||!wrong_update||!signed_zero_checks||!nan_checks;
 std::printf("{\"select_bit_checks\":%zu,\"update_and_unselected_word_bit_checks\":%zu,\"signed_zero_checks\":%zu,\"NaN_payload_checks\":%zu,\"wrong_select_counterexamples\":%zu,\"wrong_update_counterexamples\":%zu,\"errors\":%zu}\n",select_checks,update_checks,signed_zero_checks,nan_checks,wrong_select,wrong_update,errors);
 return errors!=0;
}
'''


def select4(values, j):
    return values[0] if j == 0 else values[1] if j == 1 else values[2] if j == 2 else values[3]


def update4(values, j, value):
    for k in range(4):
        if j == k:
            values[k] = value


def token_case(version, seed=13, negative=None):
    op = yee_operator(cpml_axes=(0, 1, 2))
    validate_stream_signature(op)
    reads = {FIELDS.index(u.output.field): tuple(r for r in u.reads if r.field in FIELDS)
             for u in op.updates if u.output.field in FIELDS}
    topology = Topology(16, 1024)
    points = [topology.point(tid, j) for tid in range(1024) for j in range(4)]
    if len(set(points)) != 4096:
        raise Hazard("Four-word bank mapping is not bijective")
    for tid in range(1024):
        for j in range(4):
            assert topology.owner(topology.point(tid, j)) == (tid, j)
    heads = tuple((-version - 1) % d for d in topology.cache)
    banks = [[[[None] * 4 for _ in range(1024)] for _ in range(3)] for _ in range(2)]
    frame = {}
    checks = updates = 0

    def physical(point, step):
        return tuple(64 - step - 1 + ((c + step + 1) % size) for c, size in zip(point, topology.cache, strict=True))

    def token(field, step, point):
        return field, step, physical(point, step)

    def ghost(point):
        return any(c == h for c, h in zip(point, heads, strict=True))

    def load(field, point, j_override=None):
        tid, j = topology.owner(point)
        if j_override is not None:
            j = j_override
        return select4(banks[field // 3][field % 3][tid], j)

    def store(field, point, value, wrong=False):
        tid, j = topology.owner(point)
        update4(banks[field // 3][field % 3][tid], (j + 1) % 4 if wrong else j, value)

    for point in points:
        for field in range(6):
            store(field, point, token(field, version - 1, point))
        frame[point] = [load(f, point) for f in (3, 4, 5)]
    order = list(range(32))
    rng = random.Random(seed)
    for phase in (0, 1):
        if phase:
            # All import/staging/publication operations retain original static
            # unrolling. They still address each same owner/static bank slot.
            for point in points:
                if ghost(point):
                    for field in range(6):
                        store(field, point, token(field, version, point))
                frame[point] = [load(f, point) for f in (0, 1, 2)]
        rng.shuffle(order)
        for warp in order:
            # Actual rolled SIMT point order: the same j across a whole warp.
            for j in range(4):
                for tid in range(warp * 32, warp * 32 + 32):
                    point = topology.point(tid, j)
                    if ghost(point):
                        continue
                    for field in range(phase * 3, phase * 3 + 3):
                        for read in reads[field]:
                            source = FIELDS.index(read.field)
                            other = tuple((c + d) % size for c, d, size in zip(point, read.offset, topology.cache, strict=True))
                            private = source // 3 == phase or not any(read.offset)
                            index = (j + 1) & 3 if private and negative == "wrong-select-slot" else None
                            actual = load(source, point, index) if private else frame[other][source % 3]
                            expected = (source, version - 1 + read.version,
                                        tuple(c + d for c, d in zip(physical(point, version), read.offset, strict=True)))
                            # yee_operator is traced at relative time0, hence
                            # H's own/oldE version0 and currentH/Eown version1/0.
                            if actual != expected:
                                raise Hazard((negative, phase, j, tid, field, read, actual, expected))
                            checks += 1
                        store(field, point, token(field, version, point), negative == "wrong-update-slot")
                        updates += 1
    for point in points:
        for field in range(6):
            if load(field, point) != token(field, version, point):
                raise Hazard((negative, "wrong final bank", field, point))
    return {"version": version, "seed": seed, "ownership_bijection_checks": 4096,
            "actual_traced_field_read_checks": checks, "owned_mux_updates": updates, "result": "pass"}


def validate(source, output, versions=range(1, 33), seeds=(13, 37)):
    source, output = Path(source), Path(output)
    actual = {n: (source / "src" / n).read_bytes() for n in PARENT}
    parent = dict(actual)
    parent["asymmetric_native.cuh"] = (Path(__file__).resolve().parents[1] / ".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh").read_bytes()
    expected, patch = transformed(parent)
    if actual != expected:
        raise ValueError("Actual emitted mux source or original ABI differs")
    clone = mux_body(parent["asymmetric_native.cuh"].decode())
    consumer = actual["asymmetric_native.cuh"].decode()
    if consumer.count(HELPERS) != 1 or consumer.count(clone) != 1:
        raise ValueError("Actual helper/body differs from token-bound lowering")
    output.mkdir(parents=True, exist_ok=True)
    program = HARNESS.replace("@HELPERS@", HELPERS)
    cpp, binary = output / "mux-oracle.cc", output / "mux-oracle"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    bits = json.loads(subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout)
    cases = [token_case(n, seed) for n in versions for seed in seeds]
    negatives = {}
    for name in ("wrong-select-slot", "wrong-update-slot"):
        try:
            token_case(1, negative=name)
        except Hazard as error:
            negatives[name] = repr(error)
        else:
            raise AssertionError("Unsafe mux escaped token proof:" + name)
    operator = yee_operator(cpml_axes=(0, 1, 2))
    result = {"schema": "beamz.noio-scalar-mux-proof.v1", "actual_CXX_bits": bits, "token_cases": cases,
              "traced_field_read_checks": sum(x["actual_traced_field_read_checks"] for x in cases),
              "negative_controls_rejected": negatives, "source_patch": patch,
              "actual_consumer_sha256": hashlib.sha256(actual["asymmetric_native.cuh"]).hexdigest(),
              "oracle_cpp_sha256": hashlib.sha256(program.encode()).hexdigest(),
              "operator_sha256": hashlib.sha256(json.dumps(serialize_updates(operator.updates), sort_keys=True).encode()).hexdigest(),
              "command": command, "GPU_executed": False,
              "scope": "Actual emitted float helpers preserve all four selected/unselected word bits; traced Cartesian field read/owner/version proof across interleaved register slots and both phases; unchanged source binds CPML and barrier graph; no compiler-register or speed claim"}
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    report = validate(a.source, a.output)
    print(json.dumps({"actual_CXX_bits": report["actual_CXX_bits"], "token_cases": len(report["token_cases"]), "traced_field_read_checks": report["traced_field_read_checks"], "negative_controls_rejected": list(report["negative_controls_rejected"])}))
