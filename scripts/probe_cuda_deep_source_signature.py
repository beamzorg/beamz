"""CPU compiler diagnostic: one native D4 core, generic vs admitted source mask20.

No GPU execution. Absent source groups are proven empty by the host admission
wrapper; their calls can be removed without changing any floating operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

try:
    from scripts.cuda_deep_temporal_warp import HELPERS, generate
except ModuleNotFoundError:
    from cuda_deep_temporal_warp import HELPERS, generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--native-source", type=Path, required=True)
    p.add_argument("--nvcc", required=True)
    p.add_argument("--scalar-h", action="store_true", help="Admit scalar H coefficients and lower their reads directly")
    p.add_argument("--interior", action="store_true", help="Compile unchecked full interior tile lowering; resource diagnostic only")
    p.add_argument("--generic-only", action="store_true")
    p.add_argument("--typed-byte", action="store_true")
    p.add_argument("--source20-only", action="store_true")
    p.add_argument("--field-frames", type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    native = a.native_source.resolve()
    base = "DeepWarpD4M0F32" + ("Interior" if a.interior else "") + ("BytePacked" if a.typed_byte else "") + ("DoubleFrame" if a.field_frames == 2 else "")
    source, info = generate((64, 8, 25), 4, 0, interior=a.interior, byte_packed=a.typed_byte, field_frames=a.field_frames)
    close = "\n}}} // namespace beamz::cuda::deep\n"
    kernel = source[len(HELPERS) : -len(close)]
    pattern = r"if\(!constrained_(\d) && active_sources\[(\d)\]\) next_(\d)=AddFusedHSource<(\d)>\([^;]+;"
    assert len(re.findall(pattern, kernel)) == 24
    sections = []
    wrappers = []
    for signature in ((20,) if a.source20_only else ((63,) if a.generic_only else (63, 20))):
        code = re.sub(
            pattern,
            lambda m, signature=signature: (
                m[0]
                if signature & (1 << int(m[2]))
                else "/* Empty source group removed by host-proven signature. */"
            ),
            kernel,
        )
        if a.scalar_h:
            segments = code.split("const auto &p=")
            replaced = 0
            for i, segment in enumerate(segments[1:], 1):
                if segment.startswith("h;"):
                    segment, count = re.subn(
                        r"Read\(p\.field\.inputs\[(6|7|8|9|10|11)\],z,y,x\)",
                        lambda m: f"(*static_cast<const float *>(p.field.inputs[{m[1]}].data))",
                        segment,
                    )
                    segments[i] = segment
                    replaced += count
            assert replaced == 24, replaced
            code = "const auto &p=".join(segments)
        name = base + "Signature" + str(signature) + ("ScalarH" if a.scalar_h else "")
        code = code.replace(base, name)
        sections.append(code)
        wrappers.append(
            f"""extern "C" cudaError_t CompileSignature{signature}(cudaStream_t stream,const BeamzProgramLaunch &p,const beamz::cuda::deep::Observation &o){{
+ int actual=0;for(int c=0;c<6;++c)if(p.source_groups[3+c].coefficients.dims[0])actual|=1<<c;
+ if({signature}!=63 && actual!={signature})return cudaErrorInvalidValue;
+ {'for(int i=6;i<12;++i)if(p.h_ab.inputs[i].rank!=0 || p.h_ab.inputs[i].element_type!=kBeamzF32)return cudaErrorInvalidValue;' if a.scalar_h else ''}
+ {'for(int c=0;c<3;++c)if(p.e_ab.inputs[9+c].rank!=1 || p.e_ab.outputs[c].dims[0]*p.e_ab.outputs[c].dims[1]*p.e_ab.outputs[c].dims[2]>2147483647LL)return cudaErrorInvalidValue;' if a.typed_byte else ''}
+ auto h=MakePairCpmlPhase(p.h_ab),e=MakePairCpmlPhase(p.e_ab);PairFieldOutputs final{{}};FusedHSources hs{{}},es{{}};PairCpmlGrid grid{{}};
+ for(int c=0;c<3;++c){{hs.groups[c]=p.source_groups[3+c];es.groups[c]=p.source_groups[6+c];final.fields[c]=p.h_ab.outputs[c];final.fields[3+c]=p.e_ab.outputs[c];}}
+ beamz::cuda::deep::{name}<true><<<dim3(1),480,{info['shared_bytes']},stream>>>(h,e,final,hs,es,o,grid);return cudaPeekAtLastError();
+ }}""".replace("\n+", "\n")
        )
    header = a.output_dir / "diagnostic.cuh"
    header.write_text(HELPERS + "\n".join(sections) + close)
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "cuda/src/deep_temporal_observation.cuh",
        a.output_dir / "deep_temporal_observation.cuh",
    )
    cu = a.output_dir / "compile.cu"
    cu.write_text(
        f'#include "{native / "update.cu"}"\n#include "{header.resolve()}"\n'
        + "\n".join(wrappers)
    )
    command = [
        a.nvcc,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=true",
        "-lineinfo",
        "-arch=sm_86",
        "-Xptxas=-v",
        "-I",
        str(native),
        "-c",
        str(cu),
        "-o",
        str(a.output_dir / "compile.o"),
    ]
    r = subprocess.run(command, capture_output=True, text=True, timeout=600)
    log = r.stdout + r.stderr
    (a.output_dir / "build.log").write_text(log)
    report = {
        "scope": "CPU-only one-core-mask compiler/resource diagnostic; no GPU execution",
        "resource_plan": info,
        "command": command,
        "source_sha256": hashlib.sha256(cu.read_bytes()).hexdigest(),
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "exit_code": r.returncode,
        "scalar_h": a.scalar_h,
        "unchecked_full_interior": a.interior,
        "typed_byte_packed": a.typed_byte,
        "shared_field_frames": a.field_frames,
        "kernels": [],
    }
    for chunk in log.split("ptxas info    : Function properties for ")[1:]:
        name = chunk.splitlines()[0]
        if "Signature" not in name:
            continue
        report["kernels"].append(
            {
                "signature": int(re.search(r"Signature(20|63)", name)[1]),
                "registers": int(re.search(r"Used (\d+) registers", chunk)[1]),
                "stack_spills": list(
                    map(
                        int,
                        re.search(
                            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
                            chunk,
                        ).groups(),
                    )
                ),
            }
        )
    (a.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    r.check_returncode()


if __name__ == "__main__":
    main()
