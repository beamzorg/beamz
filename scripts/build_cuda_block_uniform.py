"""Extend a frozen bounded native candidate with exact block-material facts.

Only the moving CUDA translation unit and two native host objects are rebuilt.
All other objects, flags, field ownership, and the bounded memory policy are
retained. The installed CUDA backend is never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one block-uniform patch site: {old[:100]}")
    return text.replace(old, new)


def patch(source):
    src = source / "src"
    path = src / "native_moving.cc"
    text = path.read_text()
    begin = text.index("BeamzNativeMovingUniformE BeamzCertifyNativeMovingUniformE(")
    end = text.index("// Certificate bits are immutable", begin)
    text = text[:begin] + '#include "native_block_material_facts.inc"\n\n' + text[end:]
    path.write_text(text)
    shutil.copyfile(Path(__file__).with_name("cuda_native_block_material_facts.inc"),
                    src / "native_block_material_facts.inc")
    path = src / "native_extension.cc"
    text = once(path.read_text(),
        "const bool compact_enabled=(moving_workspace_.data_flags&beamz::cuda::moving::kBytePackedE)&&(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH);",
        "const bool compact_enabled=bool(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH);")
    text = once(text, '  module.attr("moving_noio_uniform_e")=true;',
                '  module.attr("moving_block_uniform_e")=true;\n  module.attr("moving_noio_uniform_e")=true;')
    path.write_text(text)
    path = src / "asymmetric_native.cuh"
    text = path.read_text()
    text = text.replace("(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE)",
                        "((Policy==1||Policy==2)&&IsotropicH&&UniformE)")
    text = text.replace("static_assert(!CachedProfiles||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE)",
                        "static_assert(!CachedProfiles||(!Fast&&IsotropicH&&Policy==1&&UniformE)")
    text = once(text, "if constexpr(Layout==1&&Policy==1&&IsotropicH){",
                "if constexpr(Policy==1&&IsotropicH){")
    text = once(text, "}else if constexpr(Layout==1&&Policy==2&&IsotropicH){",
                "}else if constexpr(Policy==2&&IsotropicH){")
    # Constant E kernels retain Layout for terminal trail ownership/imports.
    for name in ("AsymmetricNoIOUniform1024", "AsymmetricObserverUniform1024"):
        text = once(text, f"template<int Depth>\n__global__ __launch_bounds__(1024,1) void {name}(",
                    f"template<int Depth,int Layout=1>\n__global__ __launch_bounds__(1024,1) void {name}(")
        text = text.replace(f"{name}<Depth>", f"{name}<Depth,Layout>")
    text = text.replace("UniformBody<Depth,1,false,true,1,true,", "UniformBody<Depth,Layout,false,true,1,true,")
    text = once(text, "UniformBody<Depth,1,false,true,2,true>(p,tiles);",
                "UniformBody<Depth,Layout,false,true,2,true>(p,tiles);")
    for name, end in (("LaunchNoIO", "LaunchObservers"), ("LaunchObservers", "LaunchSourcePending")):
        a = text.index(f"inline cudaError_t {name}(")
        b = text.index(f"inline cudaError_t {end}(", a)
        region = text[a:b]
        region = once(region, "if constexpr(Layout==1&&IsotropicH){", "if constexpr(IsotropicH){")
        if name == "LaunchNoIO":
            region = region.replace("if(FixedNoIOCubeEligible<", "if(Layout==1&&FixedNoIOCubeEligible<")
        text = text[:a] + region + text[b:]
    path.write_text(text)


def build(parent, output, mode_source=False, rect_shape=None, join_mode_policies=False):
    parent, output = parent.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Candidate output must be new")
    proof = json.loads((parent / "proof.json").read_text())
    previous = Path(proof["parent"]) / "build"
    previous_source = Path(proof["parent"]) / "source"
    source = output / "source"
    output.mkdir(parents=True)
    shutil.copytree(parent / "source", source)
    patch(source)
    if mode_source:
        from patch_cuda_mode_source import patch as patch_mode_source
        patch_mode_source(source)
    if rect_shape:
        from patch_cuda_rect_material import patch as patch_rect
        patch_rect(source, rect_shape)
    if join_mode_policies:
        if not mode_source:
            raise ValueError("Mode-policy joining requires the traced mode-source kernel")
        from patch_cuda_mode_cohorts import patch as patch_cohorts
        patch_cohorts(source)
    ninja = Path(json.loads((Path(proof["parent"]) / "configure-command.json").read_text())[0]).with_name("ninja")
    targets = {
        "CMakeFiles/beamz_cuda_moving.dir/src/native_moving_kernel.cu.o": "native_moving_kernel.cu.o",
        "CMakeFiles/_cuda_native.dir/src/native_moving.cc.o": "native_moving.cc.o",
    }
    commands = []
    for target, obj in targets.items():
        command = shlex.split(subprocess.check_output(
            [str(ninja), "-C", str(previous), "-t", "commands", target], text=True).splitlines()[-1])
        command = [v.replace(str(previous_source), str(source)) for v in command]
        command = [str(output / obj) + v[len(target):] if v.startswith(target) else v for v in command]
        commands.append(command)
    host = [v.replace(str(parent / "source"), str(source))
            .replace(str(parent / "native_extension.cc.o"), str(output / "native_extension.cc.o"))
            for v in proof["compile_command"]]
    commands.append(host)
    module = output / Path(proof["module"]).name
    replacements = {str(previous / target): str(output / obj) for target, obj in targets.items()}
    replacements[str(parent / "native_extension.cc.o")] = str(output / "native_extension.cc.o")
    replacements[proof["module"]] = str(module)
    link = [replacements.get(v, v) for v in proof["link_command"]]
    manifest = dict(parent=str(parent), parent_module_sha256=digest(proof["module"]),
        mode_source=mode_source, rect_shape_zyx=rect_shape, join_mode_policies=join_mode_policies,
        status="building", commands=commands, link_command=link,
        source_hashes={str(p.relative_to(source)): digest(p) for p in (source / "src").glob("*") if p.is_file()})
    (output / "proof.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "build.log").open("w") as log:
        for command in commands + [link]:
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest.update(status="complete", module=str(module), module_sha256=digest(module))
    (output / "proof.json").write_text(json.dumps(manifest, indent=2) + "\n")
    module.chmod(0o444)
    return {k: manifest[k] for k in ("status", "module", "module_sha256")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode-source", action="store_true")
    parser.add_argument("--rect-shape", type=int, nargs=3, metavar=("NZ", "NY", "NX"))
    parser.add_argument("--join-mode-policies", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.output, args.mode_source, args.rect_shape, args.join_mode_policies)))
