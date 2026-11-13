"""Permute a frozen asymmetric shared cache to spread X-face bank accesses.

Only storage addresses change. Registers, field equations, global trails,
barriers and allocations retain their original contracts. The proof checks
the full physical cache permutation and all rotating low-X packet mappings.
"""
import argparse
import collections
from pathlib import Path


def prove():
    def address(z, y, x):
        z, y, x = z % 6, y & 15, x & 31
        return (z * 16 + y) * 32 + (x ^ (y | ((z & 1) << 4)))

    slots = {address(z, y, x) for z in range(6) for y in range(16) for x in range(32)}
    assert slots == set(range(3072)), "Shared mapping is not a permutation"
    old_max, new_max = 0, 0
    for head_z in range(6):
        for head_y in range(16):
            for head_x in range(32):
                for begin in range(0, 75, 32):
                    banks = []
                    for i in range(begin, min(begin + 32, 75)):
                        z, y = head_z + 1 + i // 15, head_y + 1 + i % 15
                        banks.append(address(z, y, head_x) & 31)
                    old_max = max(old_max, len(banks))
                    new_max = max(new_max, max(collections.Counter(banks).values()))
    assert old_max == 32 and new_max <= 2
    return {"cache_points": 3072, "injective": True, "full_rotating_packet_cases": 3072,
            "maximum_low_x_shared_bank_multiplicity": {"original": old_max, "permuted": new_max},
            "compute_warp_mapping": "For fixed Z,Y, XOR with a constant permutes all32 banks",
            "scope": "Address/permutation and static packet-bank proof, not timing or CUDA race proof"}


def patch(header):
    live = Path(__file__).resolve().parent / "cuda_moving_asymmetric_cache.cuh"
    if header.resolve() == live.resolve():
        raise ValueError("Patch a frozen copy, not the agent-owned live helper")
    text = header.read_text()
    old = "return shared[(2*family+component/2)*PAD+Slot(z,y,x)];"
    if text.count(old) != 1:
        raise ValueError("Expected the unmodified asymmetric At mapping")
    text = text.replace(old, "const int zz=WrapZ(z),yy=y&15,xx=x&31;\n"
                        " return shared[(2*family+component/2)*PAD+(zz*CY+yy)*CX+"
                        "(xx^(yy|((zz&1)<<4)))];")
    proof = prove()
    header.write_text(text)
    return proof


if __name__ == "__main__":
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--header", type=Path)
    a = p.parse_args()
    print(json.dumps(patch(a.header) if a.header else prove(), indent=2))
