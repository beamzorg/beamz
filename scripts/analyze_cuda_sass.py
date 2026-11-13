"""Record static CUDA code size/control-flow budgets for frozen candidates.

These are disassembled instruction sites, not executed instructions, bandwidth
or a diagnosis of instruction-cache misses. Useful-work/time and profiler
evidence remain separate acceptance gates.
"""
import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path


def analyze(cuobjdump, binary, pattern):
    output = subprocess.run([str(cuobjdump), "--dump-sass", str(binary)],
                            check=True, capture_output=True, text=True).stdout
    result = []
    for block in output.split("Function : ")[1:]:
        name = block.splitlines()[0].strip()
        if not re.search(pattern, name):
            continue
        # Ampere SASS has 16-byte instructions; the separate encoding-only
        # continuation line does not match an opcode and is not double-counted.
        sites = re.findall(r"/\*([0-9a-f]+)\*/\s+(?:@!?U?P(?:\d+|T)\s+)?([A-Z][A-Z0-9_.]*)", block)
        counts = collections.Counter(op for _, op in sites)
        result.append({"name": name, "instruction_sites": len(sites),
                       "instruction_site_bytes": 16 * len(sites),
                       "code_extent_bytes": max((int(pc, 16) + 16 for pc, _ in sites), default=0),
                       "branch_sites": sum(counts[x] for x in ("BRA", "BRX", "JMP", "JMX")),
                       "opcodes": dict(sorted(counts.items()))})
    return {"schema": "beamz.cuda-static-code-budget.v1", "binary": str(binary.resolve()),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "cuobjdump": str(cuobjdump.resolve()),
            "scope": "Static sites; not dynamic execution or a cache-miss measurement",
            "kernels": result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuobjdump", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--pattern", default="Mstep|Moving|moving_d2")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = analyze(a.cuobjdump, a.binary, a.pattern)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"kernels": len(r["kernels"]), "output": str(a.output)}))
