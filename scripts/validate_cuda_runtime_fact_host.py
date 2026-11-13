"""Run the actual admitted IR-generated cell against the independent CPU oracle.

This uses the existing dense logical CPML/PEC/source oracle; it does not prove
native packed addressing, CUDA inter-thread ordering or runtime performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import StreamShape
from scripts.generate_cuda_runtime_fact_cell import generate
from scripts.validate_cuda_streamed_host import host_source


def validate(output, *, profiles=False, inactive=False, zero=False, metallic=0):
    generated = generate(profiles=profiles)
    shape = StreamShape((2, 2, 3), 3, 128)
    source = host_source(generated, shape, cpml=True, native=True, metallic=metallic,
                         inactive=inactive, zero_fields=zero, native_cell=True, scalar_h_class=True)
    source = source.replace("beamz::cuda::trace_native::AdvanceNativeCell", f'beamz::cuda::{generated["namespace"]}::AdvanceNativeCell')
    anchor = " if(kind<2)return kind?(.03f+.002f*f+.0001f*((z+2*y+3*x)%5)):(.99f-.0003f*(f%3));"
    if source.count(anchor) != 1:
        raise ValueError("Independent oracle coefficient anchor changed")
    source = source.replace(anchor, " if(kind==1&&f>=3)return .03f+.002f*f;\n" + anchor, 1)
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "host.cc", output / "host"
    cpp.write_text(source)
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("C++20 compiler required")
    command = [compiler, "-std=c++20", "-O1", "-pthread", "-ffp-contract=off", str(cpp), "-o", str(binary)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (output / "build.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    run = subprocess.run([str(binary.resolve())], capture_output=True, text=True, timeout=90)
    (output / "run.log").write_text(run.stdout + run.stderr)
    report = {"scope": __doc__, "generated_sha256": generated["source_sha256"],
              "host_sha256": hashlib.sha256(source.encode()).hexdigest(), "command": command,
              "profiles": profiles, "inactive": inactive, "zero_fields": zero, "pec_mask": metallic,
              "run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    run.check_returncode()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--inactive", action="store_true")
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--pec-mask", type=int, default=0)
    args = parser.parse_args()
    result = validate(args.output_dir, profiles=args.profiles, inactive=args.inactive,
                      zero=args.zero, metallic=args.pec_mask)
    print(result["stdout"].strip())


if __name__ == "__main__":
    main()
