"""Admit/instantiate depth8 on the exact frozen v13 moving parent only.

No arithmetic, storage, geometry, scheduling or existing4/16 dispatch changes.
The generic CPU planner already admits8; only native API/Prepare/template
admission is extended. This writes a private copied source tree only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PARENT_CONSUMER_SHA = "cca7202977dfb765000881b587de287c132bb7a0f14cb05131366f13b09e642c"
PARENT_EXTENSION_SHA = "aa47630581d70cbe9418228d7e08bcf044a668360c910ca74c4f19414aef5cd7"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def patch_consumer(text):
    if digest(text) != PARENT_CONSUMER_SHA:
        raise ValueError("Exact frozen v13 consumer required")
    old = "(w.depth!=4&&w.depth!=16)"
    if text.count(old) != 1:
        raise ValueError("Prepare depth gate changed")
    text = text.replace(old, "(w.depth!=4&&w.depth!=8&&w.depth!=16)", 1)
    anchor = " if(w.depth==16){if(layout==2)return DispatchClass<16,2>(stream,p,w);return DispatchClass<16,1>(stream,p,w);}"
    if text.count(anchor) != 1:
        raise ValueError("Existing depth16 dispatch changed")
    return text.replace(anchor, " if(w.depth==8){if(layout==2)return DispatchClass<8,2>(stream,p,w);return DispatchClass<8,1>(stream,p,w);}\n" + anchor, 1)


def patch_extension(text):
    if digest(text) != PARENT_EXTENSION_SHA:
        raise ValueError("Exact frozen v13 native extension required")
    old = 'if(depth!=4&&depth!=16)throw std::invalid_argument("compiled native moving depth must be4 or16");'
    if text.count(old) != 2:
        raise ValueError("Expected both host resource/API depth gates")
    text = text.replace(old, 'if(depth!=4&&depth!=8&&depth!=16)throw std::invalid_argument("compiled native moving depth must be4/8/16");')
    anchor = "NB_MODULE(_cuda_native, module) {"
    if text.count(anchor) != 1:
        raise ValueError("Native module identity changed")
    return text.replace(anchor, anchor + '\n#if BEAMZ_CUDA_NATIVE_MOVING\n  module.attr("moving_compiled_depths")=nb::make_tuple(4,8,16);\n#endif', 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("Only private copied sources are allowed")
    consumer, extension = (args.source / "src" / name for name in ("asymmetric_native.cuh", "native_extension.cc"))
    # Validate both before any write so a mismatched parent leaves no half patch.
    c, e = patch_consumer(consumer.read_bytes().decode()), patch_extension(extension.read_bytes().decode())
    consumer.write_text(c)
    extension.write_text(e)
    print(json.dumps({"parent_consumer_sha256": PARENT_CONSUMER_SHA,
                      "parent_extension_sha256": PARENT_EXTENSION_SHA,
                      "consumer_sha256": digest(c), "extension_sha256": digest(e),
                      "compiled_depths": [4, 8, 16], "new_moving_kernel_specializations": 24,
                      "scope": "Depth8-only admission/template extension; exact v13 physics/storage/cohorts unchanged"}))


if __name__ == "__main__":
    main()
