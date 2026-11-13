"""Relink the frozen native host wrapper with bounded uploads and lazy Mstep plans.

CUDA objects and their fatbins are reused byte for byte. Only the constructor's
ownership and upload policy changes; the existing ordinary method materializes
its deferred inputs before execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one patch site: {old[:100]}")
    return text.replace(old, new)


def build(parent, output):
    parent, output = parent.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Candidate output must be new")
    source, previous = parent / "source", parent / "build"
    output.mkdir(parents=True)
    staged = output / "source"
    shutil.copytree(source, staged)
    path = staged / "src/native_extension.cc"
    original = path.read_text()
    text = once(
        original,
        "const bool defer_scratch=workspace_count_&&i>=size_t(graph_count_+6)&&i<size_t(graph_count_+12);",
        """// Moving reconstructs its own plan; ordinary Mstep materializes this
      // independently deferred plan together with its existing lazy scratch.
      const bool defer_plan=workspace_count_&&i>=size_t(monitor_offset_+16);
      const bool defer_scratch=workspace_count_&&
          ((i>=size_t(graph_count_+6)&&i<size_t(graph_count_+12))||defer_plan);""",
    )
    text = once(
        text,
        "      base_inputs_.push_back(descriptor);",
        """      base_inputs_.push_back(descriptor);
      // A mapped input is staging storage. Finish this bounded upload before
      // dropping its file pages, while private snapshots retain their bytes.
      Check(cudaStreamSynchronize(stream_.value), "finish bounded input upload");
      const auto original_input=inputs[i];
      // Never discard private modifications in copy-on-write caller mappings.
      const auto mode=nb::getattr(original_input,"mode",nb::none());
      if(nb::isinstance<nb::str>(mode)&&nb::cast<std::string>(mode)=="r"&&nb::hasattr(original_input,"_mmap")){
        const auto mapping=original_input.attr("_mmap");
        if(!mapping.is_none()&&nb::hasattr(mapping,"madvise")){
          const auto mmap=nb::module_::import_("mmap");
          if(nb::hasattr(mmap,"MADV_DONTNEED"))mapping.attr("madvise")(mmap.attr("MADV_DONTNEED"));
        }
      }""",
    )
    text = once(
        text,
        '    size_t deferred_bytes=0;for(int index:deferred_scratch_)deferred_bytes+=host_[index].bytes;\n    result["deferred_scratch_bytes"]=deferred_bytes;result["allocated_scratch_bytes"]=scratch_bytes-deferred_bytes;',
        """    size_t deferred_bytes=0,deferred_plan_bytes=0;
    for(int index:deferred_scratch_){
      if(index>=monitor_offset_+16)deferred_plan_bytes+=host_[index].bytes;
      else deferred_bytes+=host_[index].bytes;
    }
    result["deferred_plan_bytes"]=deferred_plan_bytes;
    result["bounded_mapped_upload"]=true;
    result["deferred_scratch_bytes"]=deferred_bytes;
    result["allocated_scratch_bytes"]=scratch_bytes-deferred_bytes;""",
    )
    path.write_text(text)
    ninja = Path(
        json.loads((parent / "configure-command.json").read_text())[0]
    ).with_name("ninja")
    object_name = "CMakeFiles/_cuda_native.dir/src/native_extension.cc.o"
    module_name = "_cuda_native.cpython-311-x86_64-linux-gnu.so"

    def command(target):
        lines = subprocess.check_output(
            [str(ninja), "-C", str(previous), "-t", "commands", target], text=True
        ).splitlines()
        return shlex.split(lines[-1])

    compile_command = command(object_name)
    for i, value in enumerate(compile_command):
        if value == str(source / "src/native_extension.cc"):
            compile_command[i] = str(path)
        elif value == "-I" + str(source / "src"):
            compile_command[i] = "-I" + str(staged / "src")
        elif value.startswith("CMakeFiles/"):
            compile_command[i] = str(output / Path(value).name)
    link = command(module_name)
    if link[:2] != [":", "&&"] or link[-2:] != ["&&", ":"]:
        raise ValueError("Unexpected CMake linker envelope")
    link = link[2:-2]
    objects = {}
    for i, value in enumerate(link):
        if "--dependency-file=" in value:
            link[i] = "-Wl,--gc-sections"
        elif value == object_name:
            link[i] = str(output / "native_extension.cc.o")
        elif value.startswith("CMakeFiles/") or value == "libnanobind-static.a":
            file = previous / value
            objects[str(file)] = digest(file)
            link[i] = str(file)
        elif value == module_name:
            link[i] = str(output / module_name)
    proof = dict(
        parent=str(parent),
        parent_module_sha256=digest(previous / module_name),
        parent_host_sha256=digest(source / "src/native_extension.cc"),
        host_sha256=digest(path),
        reused_objects=objects,
        compile_command=compile_command,
        link_command=link,
        cuda_kernels_changed=False,
        status="building",
    )
    (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    with (output / "build.log").open("w") as log:
        subprocess.run(
            compile_command,
            cwd=previous,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            link, cwd=previous, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    for label, module in [
        ("parent", previous / module_name),
        ("candidate", output / module_name),
    ]:
        subprocess.run(
            [
                "objcopy",
                "--dump-section",
                f".nv_fatbin={output / label}.fatbin",
                str(module),
            ],
            check=True,
        )
    if digest(output / "parent.fatbin") != digest(output / "candidate.fatbin"):
        raise ValueError("CUDA fatbin changed during host-only relink")
    proof.update(
        status="complete",
        module=str(output / module_name),
        module_sha256=digest(output / module_name),
        cuda_fatbin_sha256=digest(output / "candidate.fatbin"),
        cuda_fatbin_identical=True,
    )
    (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    (output / module_name).chmod(0o444)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = build(arguments.parent, arguments.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "module", "module_sha256", "cuda_fatbin_identical")
            }
        )
    )
