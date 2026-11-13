"""Source-bound admission for speculative native cached-field derivatives.

Only the frozen NoIO joint consumer is admitted. Its derivative callbacks read
fixed private slots or three bounded shared planes; they cannot touch global
profiles, psi, source/DFT state, atomics or diagnostic counters. The enclosing
H/E barriers keep the opposite shared family immutable throughout each phase.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from scripts.cuda_space_time_ir import (
    IllegalSchedule,
    validate_stream_signature,
    yee_operator,
)

PARENT = "b514c317c82a4e680d304745ec5a8ffb74f9be47bdd4874b3742954521f10906"
H_READ = "if(field<3)return h[field][j];const int c=field-3; if(!dz&&!ddy&&!ddx)return e[c][j];return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];"
E_READ = "if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];"


@dataclass(frozen=True)
class CachedFieldReadFact:
    consumer_sha256: str
    callbacks_sha256: tuple[str, str]
    slot_sha256: str
    difference_sha256: str
    schema: str = "beamz.NoIO-physical-validity-independent-cached-field-reads.v1"


EXPECTED = CachedFieldReadFact(PARENT,
    ("51a5f1eed7ce8f42bc83abbb3ee507c50228cfa806934f4ecc037529c8629f6e",
     "a43f926198b9f633360b73475d07a42c453839af09f67ac81cd242ea6a73dae3"),
    "7e36514e9cc72e973229e49af0a96799cb5af6e851d084343d5c3745e7b3d8a6",
    "562588b033aeeeb1a5a01ba0586c42a34c4659729ff7d8d176573a01d1658de6")


def verify_fact(fact):
    if fact != EXPECTED:
        raise IllegalSchedule("Cached read speculation requires exact callback/Slot/Difference admission")


def certify(consumer):
    checksum = hashlib.sha256(consumer.encode()).hexdigest()
    if checksum != PARENT:
        raise IllegalSchedule("Exact reviewed cached native consumer required")
    callbacks = re.findall(r"const auto read=\[&\] __device__\(int field,int,int dz,int ddy,int ddx\)\{(.*?)\n     \};", consumer, re.S)
    if len(callbacks) != 2:
        raise IllegalSchedule("Exactly two cached field callback definitions required")
    normalized = tuple(" ".join(text.split()) for text in callbacks)
    if normalized != (H_READ, E_READ):
        raise IllegalSchedule("Read callback no longer contains only admitted private/shared reads")
    slot_first = consumer.index("__device__ __forceinline__ int WrapZ(")
    slot_last = consumer.index("__device__ __forceinline__ float& At(", slot_first)
    slot = consumer[slot_first:slot_last]
    difference_first = consumer.index(" __device__ __forceinline__ float Difference(")
    difference_last = consumer.index(" __device__ __forceinline__ float Finish(", difference_first)
    difference = consumer[difference_first:difference_last]
    if any(token in difference for token in ("LoadLocal", "StoreLocal", "__ldg", "atomic", "Finish")):
        raise IllegalSchedule("Difference callback has an inadmissible data side effect")
    fact = CachedFieldReadFact(checksum, tuple(hashlib.sha256(text.encode()).hexdigest() for text in normalized),
                               hashlib.sha256(slot.encode()).hexdigest(), hashlib.sha256(difference.encode()).hexdigest())
    verify_fact(fact)
    return fact


def prove_addresses(operator=None):
    """All cache slots/actual opposite field reads, with either admitted swizzle."""
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    validate_stream_signature(operator)
    reads = []
    for phase in (0, 1):
        reads.append(sorted({read for u in operator.updates if operator.specs[u.output.field].phase == phase
                             for read in u.reads if operator.specs[read.field].role == "field"
                             and operator.specs[read.field].phase != phase}))
    checked = 0
    for swizzle in (False, True):
        for tid in range(1024):
            sy, sx, zbase = tid % 512 // 32, tid % 32, tid // 512
            for j in range(4):
                zs = zbase + 2 * j
                for phase in (0, 1):
                    for read in reads[phase]:
                        component = "xyz".index(read.field[-1])
                        if not any(read.offset):
                            assert 0 <= component < 3 and 0 <= j < 4
                        else:
                            dz, dy, dx = read.offset
                            z, y, x = (zs + dz) & 7, (sy + dy) & 15, (sx + dx) & 31
                            if swizzle:
                                x ^= y | ((z & 1) << 4)
                            slot = (z * 16 + y) * 32 + x
                            assert 0 <= slot < 4096
                            assert component * 4096 <= component * 4096 + slot < (component + 1) * 4096
                        checked += 1
    return {"checked_cached_reads": checked, "threads": 1024, "slots": 4, "shared_family_floats": 3 * 4096,
            "both_swizzles": True,
            "scope": "All fixed cache addresses remain bounded independently of physical output Valid; CUDA phase barriers/source binding are separate admission facts"}
