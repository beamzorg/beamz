"""Rebuild only the C++ NativeSession object against frozen CUDA device objects.

For host-only research changes. Every device-consumed source must stay bytewise
identical. The new object must contain no CUDA device/fatbinary sections. Parent
objects and module remain untouched; exact commands/object hashes are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ALLOWED = {"src/native_extension.cc", "src/native_cohort_dependencies.h"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--ninja", type=Path, required=True)
    a = p.parse_args()
    parent = a.parent.resolve()
    candidate = a.candidate.resolve()
    build = parent / "build"
    source = candidate / "source"
    out = candidate / "build"
    out.mkdir(exist_ok=False)
    m = json.loads((parent / "manifest.json").read_text())
    parent_module = Path(m["module"])
    assert sha(parent_module) == m["module_sha256"]
    changed = []
    for name, old in m["source_hashes"].items():
        assert sha(parent / "source" / name) == old, ("Parent source changed", name)
        if sha(source / name) != old:
            changed.append(name)
    assert changed and set(changed) <= ALLOWED, changed
    lines = subprocess.check_output(
        [str(a.ninja.resolve()), "-C", str(build), "-t", "commands", "_cuda_native"],
        text=True,
    ).splitlines()
    compile_lines = [
        x
        for x in lines
        if "native_extension.cc.o" in x and " -c " in x and "/usr/bin/c++" in x
    ]
    assert len(compile_lines) == 1
    cc = shlex.split(compile_lines[0])
    old_obj = "CMakeFiles/_cuda_native.dir/src/native_extension.cc.o"
    new_obj = out / "native_extension.cc.o"
    cc = [
        x.replace(str(parent / "source"), str(source)).replace(old_obj, str(new_obj))
        for x in cc
    ]
    link_lines = [
        x
        for x in lines
        if "native_extension.cc.o" in x and " -shared " in x and "/usr/bin/c++" in x
    ]
    assert len(link_lines) == 1
    segments = link_lines[0].split("&&")
    matches = [shlex.split(x) for x in segments if "/usr/bin/c++" in x]
    assert len(matches) == 1
    link = matches[0]
    module = out / parent_module.name
    for i, x in enumerate(link):
        if x == old_obj:
            link[i] = str(new_obj)
        elif x.startswith("-Wl,--dependency-file="):
            link[i] = "-Wl,--dependency-file=" + str(out / "link.d")
    index = link.index("-o") + 1
    link[index] = str(module)
    assert not any("nvcc" in x or x in ("&&", ";") for x in cc + link)
    provenance = {
        "schema": "beamz.host-only-relink.v1",
        "parent_module": str(parent_module),
        "parent_module_sha256": sha(parent_module),
        "changed_host_sources": changed,
        "compile_argv": cc,
        "link_argv": link,
        "cwd": str(build),
        "retained_objects": {
            str((build / x).resolve()): sha(build / x)
            for x in link
            if x.endswith((".o", ".a")) and x != str(new_obj)
        },
        "source_hashes": {
            str(x.relative_to(source)): sha(x)
            for x in sorted(source.rglob("*"))
            if x.is_file()
        },
    }
    (candidate / "host-relink-before.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    with (candidate / "host-build.log").open("w") as log:
        subprocess.run(cc, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
        sections = subprocess.check_output(["readelf", "-S", str(new_obj)], text=True)
        assert ".nv_fatbin" not in sections and ".nvFatBinSegment" not in sections
        subprocess.run(
            link, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    assert sha(parent_module) == m["module_sha256"], "Parent module mutated"
    provenance.update(
        module=str(module),
        module_sha256=sha(module),
        new_host_object_sha256=sha(new_obj),
        no_new_cuda_sections=True,
        compile_exit_code=0,
        gpu_executed=False,
    )
    (candidate / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: provenance[k]
                for k in (
                    "module",
                    "module_sha256",
                    "changed_host_sources",
                    "no_new_cuda_sections",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
