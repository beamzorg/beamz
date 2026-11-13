"""Compile the ACTUAL copied host annotation loop and compare an independent fact."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_safe_initial_fact import certify


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must be new")
    args.output_dir.mkdir(parents=True)
    src = args.source.resolve() / "src"
    extension = (src / "native_extension.cc").read_text()
    first = extension.index("    moving_safe_initial_tiles_=0;")
    last = extension.index("    BeamzNativeMovingSparseObservation sparse;", first)
    actual_loop = extension[first:last]
    program = r'''#include "native_moving.h"
#include <cstdint>
#include <iostream>
struct Plan {std::vector<uint8_t>tile_flags;int counts[3],owned[3];};
struct Input {int64_t dims[3];};
int main(){int cases;std::cin>>cases;while(cases--){
 Plan moving_plan_;Input base_inputs_[6];size_t moving_safe_initial_tiles_;
 for(int&a:moving_plan_.counts)std::cin>>a;
 for(int&a:moving_plan_.owned)std::cin>>a;
 for(auto&f:base_inputs_)for(auto&a:f.dims)std::cin>>a;
 const int cells=moving_plan_.counts[0]*moving_plan_.counts[1]*moving_plan_.counts[2];
 for(int i=0;i<cells;++i){int v;std::cin>>v;moving_plan_.tile_flags.push_back(uint8_t(v));}
 ACTUAL_LOOP
 std::cout<<moving_safe_initial_tiles_;for(auto flag:moving_plan_.tile_flags)std::cout<<' '<<int(flag);std::cout<<'\n';
}}
'''.replace("ACTUAL_LOOP", actual_loop)
    cpp, binary = args.output_dir / "host.cpp", args.output_dir / "host"
    cpp.write_text(program)
    command = ["g++", "-std=c++17", "-O0", "-I", str(src), str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    cases = []
    delta = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
    contexts = (0, 1, 16, 17, 2, 4, 8, 32, 18, 64, 65, 80, 81)
    for n, mode in itertools.product(((62, 64, 96), (64, 96, 128), (17, 29, 47)), (0, 1)):
        supports = [tuple(a + b for a, b in zip(n, d, strict=True)) for d in delta]
        owned = (7, 15, 31)
        counts = tuple((size + 16 + b - 1) // b + 1 for size, b in zip(n, owned, strict=True))
        tiles = list(itertools.product(*(range(c) for c in counts)))
        flags = [0 if not mode else contexts[i % len(contexts)] for i in range(len(tiles))]
        expected = []
        admitted = 0
        for tile, flag in zip(tiles, flags, strict=True):
            low = flag & 63
            policy = 0 if not low else 1 if not low & 17 else 2 if low == 1 else 3 if low == 16 else 4 if low == 17 else 5
            safe = certify(supports, tile, policy=policy)
            expected.append(flag | (128 if safe else 0))
            admitted += safe
        cases.append({"counts": counts, "owned": owned, "supports": supports,
                      "flags": flags, "expected": [admitted, *expected]})
    payload = [str(len(cases))]
    for case in cases:
        payload.append(" ".join(map(str, (*case["counts"], *case["owned"],
                                          *(v for shape in case["supports"] for v in shape), *case["flags"]))))
    output = subprocess.check_output([str(binary)], input="\n".join(payload) + "\n", text=True).splitlines()
    if len(output) != len(cases):
        raise AssertionError("Host proof output count differs")
    for case, line in zip(cases, output, strict=True):
        if list(map(int, line.split())) != case["expected"]:
            raise AssertionError("Actual host annotation differs from independent support fact")
    report = {"scope": __doc__, "passed": True, "cases": len(cases),
              "tile_annotations_checked": sum(len(c["flags"]) for c in cases),
              "certified_counts": [c["expected"][0] for c in cases],
              "actual_host_loop_sha256": digest(actual_loop),
              "native_extension_sha256": digest((src / "native_extension.cc").read_bytes()),
              "native_policy_header_sha256": digest((src / "native_moving.h").read_bytes()),
              "compiler_command": command, "host_binary_sha256": digest(binary.read_bytes()),
              "proof_source_sha256": digest(Path(__file__).read_bytes())}
    (args.output_dir / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
