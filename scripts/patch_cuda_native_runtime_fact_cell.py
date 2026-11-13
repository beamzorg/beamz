"""Route only certified NoIO native points through the runtime-fact IR header.

Private copied consumers only. Native callbacks, source/DFT/psi clocks, complete
field schedule, fallback point header and host certification remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_cuda_runtime_fact_cell import generate

PARENTS = {
    "3ed4a4f837df1ca60f409a243c0dd8a63778c0b2b983bac0ee895b56da04d6aa": "4941-NoIO1024",
    "a111536eba602a638aeec536ae58af5728f64af8bad0609c7b64e4f672369c62": "compact-NoIO-v1",
}
CANONICAL_CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Runtime fact-cell anchor changed: {old}")
    return text.replace(old, new, 1)


def patch(text):
    checksum = hashlib.sha256(text.encode()).hexdigest()
    if checksum not in PARENTS:
        raise ValueError("Reviewed NoIO1024 or compact parent required")
    text = once(text, '#include "asymmetric_native_cell_generated.cuh"',
                '#include "asymmetric_native_cell_generated.cuh"\n#include "asymmetric_native_cell_runtime_generated.cuh"')
    marker = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>\n__device__ __forceinline__ void UniformBody"
    selector = """template<int Phase,int Component,bool UseRuntimeFactCell,class Env,class Read>
__device__ __forceinline__ float AdvanceRuntimeSelected(Env&env,Read read,int n,int z,int y,int x){
 if constexpr(UseRuntimeFactCell)return beamz::cuda::trace_native_runtime::AdvanceNativeCell<Phase,Component,false>(env,read,n,z,y,x);
 else return beamz::cuda::trace_native::AdvanceNativeCell<Phase,Component,false>(env,read,n,z,y,x);
}

"""
    replacement = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,bool UseRuntimeFactCell=false>\n__device__ __forceinline__ void UniformBody"
    text = once(text, marker, selector + replacement)
    text = once(text, " constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n extern __shared__ float frame[];",
                ' static_assert(!UseRuntimeFactCell||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE),"Runtime fact cell requires the certified NoIO class");\n constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n extern __shared__ float frame[];')
    for phase in (0, 1):
        for c in range(3):
            old = f"beamz::cuda::trace_native::AdvanceNativeCell<{phase},{c},false>(env,read,n,z,y,x)"
            text = once(text, old, f"AdvanceRuntimeSelected<{phase},{c},UseRuntimeFactCell>(env,read,n,z,y,x)")
    return once(text, "UniformBody<Depth,1,false,true,1,true>(p,tiles);",
                "UniformBody<Depth,1,false,true,1,true,true>(p,tiles);")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("Only private copied source may be patched")
    src = args.source / "src"
    canonical = src / "asymmetric_native_cell_generated.cuh"
    if hashlib.sha256(canonical.read_bytes()).hexdigest() != CANONICAL_CELL:
        raise ValueError("Canonical physics header changed")
    consumer = src / "asymmetric_native.cuh"
    parent = consumer.read_bytes().decode()
    result = patch(parent)
    generated = generate()
    header = src / "asymmetric_native_cell_runtime_generated.cuh"
    header.write_text(generated["source"])
    consumer.write_text(result)
    extension = src / "native_extension.cc"
    extension_source = extension.read_bytes().decode()
    extension_source = once(extension_source, "NB_MODULE(_cuda_native, module) {",
                            'NB_MODULE(_cuda_native, module) {\n  module.attr("moving_runtime_fact_cell")=true;\n'
                            f'  module.attr("moving_runtime_fact_cell_sha256")="{generated["source_sha256"]}";')
    extension.write_text(extension_source)
    manifest = {k: v for k, v in generated.items() if k != "source"}
    manifest.update(parent_consumer_sha256=hashlib.sha256(parent.encode()).hexdigest(),
                    consumer_sha256=hashlib.sha256(result.encode()).hexdigest(),
                    extension_sha256=hashlib.sha256(extension_source.encode()).hexdigest(),
                    canonical_cell_sha256=CANONICAL_CELL,
                    fallback="Every other body uses original canonical header; no host class or schedule change")
    (args.source.parent / "runtime-fact-cell-proof.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"consumer_sha256": manifest["consumer_sha256"], "generated_cell_sha256": generated["source_sha256"],
                      "parent": PARENTS[manifest["parent_consumer_sha256"]]}))


if __name__ == "__main__":
    main()
