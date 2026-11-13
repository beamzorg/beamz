"""Copy-only paired source-plane schedule on the immutable selector hybrid."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT = "c598caa27aa259920b49acee1b760f03606ae6305e49fa740234ee6bc8273601"
PARENT_EXTENSION = "3d2b2fad15b477ae8f981880712bdba71cbf86842f3d54c6a3afdbc268872dbd"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents
            or output in source.parents or output == canonical or canonical in output.parents):
        p.error("New disjoint noncanonical output required")
    c = (source / "src/asymmetric_native.cuh").read_bytes()
    e = (source / "src/native_extension.cc").read_bytes()
    if sha(c) != PARENT or sha(e) != PARENT_EXTENSION:
        p.error("Exact frozen v11On source required")
    text = c.decode()
    start = text.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1>\n__device__ __forceinline__ void SourcePendingBody(")
    end = text.index("constexpr size_t SourcePendingSharedBytes=", start)
    pair = Path(__file__).with_name("cuda_native_source_pending_pair_body.cuh").read_bytes()
    body = pair.decode()
    changed = text[:start] + body + "\n" + text[end:]
    oldcall = "SourcePendingBody<Depth,1,false,true,Policy,false,1024,20>(p,ids);"
    newcall = "SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20>(p,ids);"
    if changed.count(oldcall) != 1:
        p.error("Unexpected source-entry binding")
    changed = changed.replace(oldcall, newcall, 1)
    metadata = '    result["moving_source_pending_shared_bytes"]=58680;'
    assert e.decode().count(metadata) == 1
    ext = e.decode().replace(metadata, '    result["moving_source_pending_barriers"]=5;\n' + metadata, 1)
    anchor = '  module.attr("moving_source_pending_mask20_1024")=true;'
    assert ext.count(anchor) == 1
    ext = ext.replace(anchor, anchor + '\n  module.attr("moving_source_pending_barriers")=5;\n  module.attr("moving_source_pending_body_sha256")="' + sha(pair) + '";', 1)
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(changed)
    (output / "src/native_extension.cc").write_text(ext)
    report = {"parent_module_sha256": "3ae787722dde65c5f537b77e0e8e7dca1fbadbf449eef1117cc038a6b0cf3591",
              "parent_consumer_sha256": PARENT, "consumer_sha256": sha(changed.encode()),
              "extension_sha256": sha(ext.encode()), "paired_body_sha256": sha(pair),
              "original_source_body_sha256": sha(text[start:end].encode()),
              "all_other_source_unchanged": True,
              "scope": "Source-only paired5barriers, sameguards/hostselector/sourceFP/queues; original52source unchanged; requiresresource/numeric/sync gates"}
    (output.parent / "paired-source-proof.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
