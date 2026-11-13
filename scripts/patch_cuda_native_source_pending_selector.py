"""Bind an immutable hidden host selector for matched source-hybrid controls.

The CUDA TU sees only an external declaration, so both paths and all kernels
remain compiled. A control DSO changes only the C++ definition to false while
retaining every CUDA object, creating a same-code source on/off comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENTS = {
    "src/asymmetric_native.cuh": "85c6df826aa799280672c5120be3935282566c9f50d3c4bafbc1077c7375cc0e",
    "src/native_extension.cc": "a7e5971796c3781f647bcb8a59427c47d1745fb42db2623c93a7689f0b41d40b",
    "src/native_moving.h": "9f923d4eecbe250f30f1ba76530c368804032045622cd1a00aa4033ad4233d09",
    "src/native_moving_kernel.cu": "09139fca2de14e000b4d52a77e1f18c81ad7815d526a9068278fbc0e58681d1e",
    "CMakeLists.txt": "f18fb2d7182a0fae93278e5f1b59af9d69cc98f86fee02ccaf122794f8f0c981",
}
DECLARATION = 'extern const bool BeamzResearchSourcePendingEnabled __attribute__((visibility("hidden")));'
DEFINITION = 'const bool BeamzResearchSourcePendingEnabled=true;'
OFF_DEFINITION = 'const bool BeamzResearchSourcePendingEnabled=false;'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one selector anchor: {old!r}")
    return text.replace(old, new, 1)


def transformed(parent):
    for name, expected in PARENTS.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact frozen e8af hybrid-v10 parent required: {name}")
    files = {name: data.decode() for name, data in parent.items()}
    h = files["src/native_moving.h"]
    h = once(h, "struct BeamzNativeMovingWorkspace {",
             "// Immutable per-DSO host research selector; no device reads/public setter.\n" + DECLARATION + "\n\nstruct BeamzNativeMovingWorkspace {")
    files["src/native_moving.h"] = h
    c = files["src/asymmetric_native.cuh"]
    c = once(c, "if(BeamzNativeMovingSourceMask20Class(Layout,IsotropicH,presence)){",
             "if(BeamzResearchSourcePendingEnabled&&BeamzNativeMovingSourceMask20Class(Layout,IsotropicH,presence)){")
    files["src/asymmetric_native.cuh"] = c
    e = files["src/native_extension.cc"]
    e = once(e, "namespace nb = nanobind;", DEFINITION + "\n\nnamespace nb = nanobind;")
    e = once(e, 'result["moving_source_mask20_dispatch_active"]=source_class&&',
             'result["moving_source_pending_selector_enabled"]=BeamzResearchSourcePendingEnabled;\n    result["moving_source_mask20_dispatch_active"]=BeamzResearchSourcePendingEnabled&&source_class&&')
    anchor = "    append_bytes(&moving_workspace_.source_presence_mask,sizeof(uint32_t));"
    e = once(e, anchor, anchor + "\n    const bool source_selector=BeamzResearchSourcePendingEnabled;append_bytes(&source_selector,sizeof(bool));")
    anchor = '  module.attr("moving_source_pending_mask20_1024")=true;'
    e = once(e, anchor, anchor + '\n  module.attr("moving_source_pending_selector_enabled")=BeamzResearchSourcePendingEnabled;')
    files["src/native_extension.cc"] = e
    return {name: text.encode() for name, text in files.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents
            or output in source.parents or output == canonical or canonical in output.parents):
        p.error("New disjoint noncanonical copy required")
    parent = {name: (source/name).read_bytes() for name in PARENTS}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output/name).write_bytes(data)
    r = {"parent_module_sha256": "e8af10093b72958b7081ce0f3868cbf89758e87b5013f6fa03fccacafbe0645d",
         "parent_hashes": PARENTS, "result_hashes": {name: sha(data) for name, data in files.items()},
         "patcher_sha256": sha(Path(__file__).read_bytes()),
         "scope": "Only hidden host selector/metadata/key; kernel definitions/storage/math unchanged, matched off via CXX-only false definition"}
    (output.parent/"selector-source-proof.json").write_text(json.dumps(r, indent=2)+"\n")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
