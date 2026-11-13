"""Experimental complete cube geometry on the exact paired v12 source.

This changes the host ownership/trails and every moving field consumer together.
The independent cube read oracle is a required source-review input, not a claim
that this produced CUDA has passed resources or hardware validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_v12_noio_axis_bundle import PARENT_HASHES, PARENT_MODULE
except ModuleNotFoundError:
    from patch_cuda_v12_noio_axis_bundle import PARENT_HASHES, PARENT_MODULE


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one frozen cube anchor: {old!r}")
    return text.replace(old, new, 1)


def transform(parent):
    for name, expected in PARENT_HASHES.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact a939 source required: {name}")
    c, h, e = (parent[name].decode() for name in ("asymmetric_native.cuh", "native_moving.h", "native_extension.cc"))
    old_c = c
    c = once(c, "BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,Threads=512;",
             'BZ=15,BY=15,BX=15,CZ=16,CY=16,CX=16,Threads=512;\nstatic_assert(!BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE,"Cube requires the proved unswizzled slot map");')
    c = once(c, "z=WrapZ(z);y&=15;x&=31;", "z=WrapZ(z);y&=(CY-1);x&=(CX-1);")
    # Only geometry-bearing field bodies are rewritten. Packed-material &31,
    # sparse observation masks and field-component masks remain untouched.
    start = c.index("template<int T,int Slots>\n")
    end = c.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>", start)
    before_uniform = c[start:end]
    uniform = before_uniform
    for old, new in (("threadIdx.x/512", "threadIdx.x/(CY*CX)"), ("T/512", "T/(CY*CX)"),
                     ("threadIdx.x/32", "threadIdx.x/CX"), ("&7", "&(CZ-1)"),
                     ("&15", "&(CY-1)"), ("&31", "&(CX-1)"),
                     ("==7?", "==CZ-1?"), ("==15?", "==CY-1?"), ("==31?", "==CX-1?")):
        uniform = uniform.replace(old, new)
    c = c[:start] + uniform + c[end:]
    start = c.index("// Two-plane publication groups;")
    end = c.index("inline bool MakeSource(", start)
    before_paired = c[start:end]
    paired = before_paired
    edits = (
        ('CZ==8&&CY==16&&CX==32&&T==1024&&LocalZ==4&&YGroups==1', 'CZ==16&&CY==16&&CX==16&&T==1024&&ZGroups==4&&LocalZ==4&&YGroups==1'),
        ("threadIdx.x/32", "threadIdx.x/CX"),
        ("if constexpr(ZGroups==2)", "if constexpr(ZGroups>1)"),
        ("((zbase/LocalZ)^1)", "((zbase/LocalZ+ZGroups-1)%ZGroups)"),
        ("__shfl_sync(0xffffffffu,h[1][si],(sx+31)&31)", "__shfl_sync(0xffffffffu,h[1][si],(sx+CX-1)&(CX-1),CX)"),
        ("&31", "&(CX-1)"),
        ("sx==31?", "sx==CX-1?"),
        ('SourcePendingSharedBytes==58680', 'SourcePendingSharedBytes==60120'),
    )
    for old, new in edits:
        if old not in paired:
            raise ValueError(f"Missing paired cube anchor {old!r}")
        paired = paired.replace(old, new)
    c = c[:start] + paired + c[end:]
    # Replacing whole reviewed regions gives an unambiguous exact inverse even
    # when generalized expressions already existed elsewhere in the parent.
    inverse = c.replace(paired, before_paired, 1).replace(uniform, before_uniform, 1)
    inverse = once(inverse, "z=WrapZ(z);y&=(CY-1);x&=(CX-1);", "z=WrapZ(z);y&=15;x&=31;")
    inverse = once(inverse,
                   'BZ=15,BY=15,BX=15,CZ=16,CY=16,CX=16,Threads=512;\nstatic_assert(!BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE,"Cube requires the proved unswizzled slot map");',
                   "BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,Threads=512;")
    if inverse != old_c:
        raise ValueError("Cube consumer inverse failed")
    h = once(h, "kOwnedZ=7,kOwnedY=15,kOwnedX=31", "kOwnedZ=15,kOwnedY=15,kOwnedX=15")
    if h.count("{7,15,31}") != 2:
        raise ValueError("Both workspace and host-plan ownership must change")
    h = h.replace("{7,15,31}", "{15,15,15}")
    e = once(e, 'result["moving_source_pending_shared_bytes"]=58680;result["moving_original_shared_bytes"]=51748;',
             'result["moving_source_pending_shared_bytes"]=60120;result["moving_original_shared_bytes"]=54628;')
    e = once(e, '  module.attr("moving_source_pending_mask20_1024")=true;',
             '  module.attr("moving_cube_geometry")=true;module.attr("moving_cached_zyx")=nb::make_tuple(16,16,16);\n  module.attr("moving_source_pending_mask20_1024")=true;')
    files = dict(parent)
    files["asymmetric_native.cuh"], files["native_moving.h"], files["native_extension.cc"] = c.encode(), h.encode(), e.encode()
    return files


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transform(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {"parent_module_sha256": PARENT_MODULE, "parent_hashes": PARENT_HASHES,
             "result_hashes": {name: sha(data) for name, data in files.items()},
             "patcher_sha256": sha(Path(__file__).read_bytes()), "exact_consumer_inverse": True,
             "cached": [16, 16, 16], "owned": [15, 15, 15],
             "uniform_shared_bytes": 54628, "source_shared_bytes": 60120,
             "source_private_field_bound": 20, "source_barriers": 5,
             "source_z_groups": 4, "source_hy_shuffle_width": 16,
             "canonical_cell_byte_identical": True,
             "scope": "Whole moving host plan + all uniform/source field consumers change geometry together; ordinary CUDA tails/cell math unchanged",
             "requires": ["independent actual source/oracle binding", "all58 zero-local resource gate", "numerical and sanitizer gates", "same-physics measured performance"],
             "gpu_executed": False}
    (output.parent / "cube-source-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output)))
