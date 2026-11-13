"""Build one bounded temporal engine with general source and geometry lowering.

The canonical CUDA source is cuda/src/temporal/kernel.cuh. No source mask,
contribution count, grid dimension or benchmark name selects a kernel family.
The parent supplies the already-validated native ABI and bounded host ownership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main() -> int:
    """Stage the shared compiler source and build an immutable native module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-kernels", type=Path,
                        help="Reuse identical kernel/planner objects for a wrapper-only change")
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Candidate output must be new")
    root = Path(__file__).resolve().parents[1]
    proof = json.loads((parent / "proof.json").read_text())
    previous, previous_source = Path(proof["parent"]) / "build", Path(proof["parent"]) / "source"
    source = output / "source"
    output.mkdir(parents=True)
    shutil.copytree(parent / "source", source)
    src = source / "src"
    (src / "asymmetric_native.cuh").unlink()
    shutil.copyfile(root / "cuda/src/temporal/kernel.cuh", src / "temporal_native.cuh")
    p = src / "native_moving_kernel.cu"
    p.write_text(p.read_text().replace('#include "asymmetric_native.cuh"', '#include "temporal_native.cuh"'))
    p = src / "native_moving.cc"
    text = p.read_text()
    a = text.index("BeamzNativeMovingUniformE BeamzCertifyNativeMovingUniformE(")
    b = text.index("// Certificate bits are immutable", a)
    p.write_text(text[:a] + '#include "native_block_material_facts.inc"\n\n' + text[b:])
    shutil.copyfile(root / "scripts/cuda_native_block_material_facts.inc", src / "native_block_material_facts.inc")
    p = src / "native_moving.h"
    text = p.read_text()
    a = text.index("// Immutable per-DSO host research selector;")
    b = text.index("struct BeamzNativeMovingWorkspace", a)
    text = text[:a] + text[b:]
    a = text.index("inline constexpr bool BeamzNativeMovingSourceMask20Class(")
    b = text.index("// CUDA consumer owns", a)
    p.write_text(text[:a] + text[b:])
    p = src / "native_extension.cc"
    text = p.read_text().replace("const bool BeamzResearchSourcePendingEnabled=true;\n", "").replace("const bool BeamzResearchSourcePlanePairEnabled=true;\n", "")
    shutil.copyfile(root / "scripts/cuda_native_observation_masks.inc", src / "native_observation_masks.inc")
    text = text.replace('#include "native_macro_slab.h"', '#include "native_macro_slab.h"\n#include "native_observation_masks.inc"')
    text = text.replace("BeamzBuildNativeMovingSparseObservation(", "BeamzBuildNativeMovingMaskedObservation(")
    shutil.copyfile(root / "scripts/cuda_native_zero_reset.inc", src / "native_zero_reset.inc")
    text = text.replace('#include "native_observation_masks.inc"', '#include "native_observation_masks.inc"\n#include "native_zero_reset.inc"')
    text = text.replace("  std::set<int> persistent_;", "  std::set<int> persistent_;\n  std::set<int> zero_reset_;")
    a = text.index("      host_.push_back(Describe(inputs[i]")
    b = text.index("      auto memory =", a)
    original = text[a:b]
    original = original.replace("host_.push_back(Describe(", "host_.push_back(zero_initial ? std::move(view) : Describe(")
    text = text[:a] + '''      auto view = Describe(inputs[i], numpy_, false);
      const bool zero_initial = (persistent_.count(i) || defer_scratch) && BeamzNativeAllPositiveZero(view.data, view.bytes);
      if (zero_initial) zero_reset_.insert(int(i));
''' + original + text[b:]
    reset = '''        Check(cudaMemcpyAsync(base_inputs_[index].data, host_[index].data,
                              host_[index].bytes, cudaMemcpyHostToDevice,
                              stream_.value), "reset native state");'''
    if text.count(reset) != 1:
        raise ValueError("Native reset contract changed")
    text = text.replace(reset, '''        if (zero_reset_.count(index))
          Check(cudaMemsetAsync(base_inputs_[index].data, 0, host_[index].bytes, stream_.value), "reset captured zero state");
        else
''' + reset)
    text = text.replace('if(host.bytes)std::memcpy(const_cast<void*>(destination.data),host.data,host.bytes);',
        'if(host.bytes){if(zero_reset_.count(output_aliases_[i]))std::memset(const_cast<void*>(destination.data),0,host.bytes);else std::memcpy(const_cast<void*>(destination.data),host.data,host.bytes);}')
    text = text.replace('if(expected.bytes)std::memcpy(const_cast<void*>(destination.data),expected.data,expected.bytes);',
        'if(expected.bytes){if(zero_reset_.count(output_aliases_[output_index]))std::memset(const_cast<void*>(destination.data),0,expected.bytes);else std::memcpy(const_cast<void*>(destination.data),expected.data,expected.bytes);}')
    text = text.replace('    for(int index:retire)if(host_[index].bytes)Check(', '''    for(int index:retire)if(zero_reset_.erase(index)){
      auto array=numpy_.attr("empty")(host_[index].shape,"dtype"_a=host_[index].dtype);
      host_[index]=Describe(array,numpy_,false);
    }
    for(int index:retire)if(host_[index].bytes)Check(''')
    text = text.replace('if(host_[index].bytes)Check(cudaMemcpyAsync(memory->data,host_[index].data,host_[index].bytes,cudaMemcpyHostToDevice,stream_.value),"materialize deferred Mstep scratch");',
        'if(host_[index].bytes){if(zero_reset_.count(index))Check(cudaMemsetAsync(memory->data,0,host_[index].bytes,stream_.value),"materialize captured zero scratch");else Check(cudaMemcpyAsync(memory->data,host_[index].data,host_[index].bytes,cudaMemcpyHostToDevice,stream_.value),"materialize deferred Mstep scratch");}')
    a = text.index("    const int source_layout=")
    b = text.index('    result["moving_source_pending_barriers"]', a)
    text = text[:a] + '''    std::array<int64_t,3> source_tiles{};
    for(size_t rank=0;rank<moving_plan_.bulk_counts.size();++rank){
     if(!moving_plan_.policy_offsets.empty())for(int policy=3;policy<=5;++policy)source_tiles[policy-3]+=moving_plan_.policy_offsets[rank*7+policy+1]-moving_plan_.policy_offsets[rank*7+policy];
     else source_tiles[2]+=moving_plan_.rank_offsets[rank+1]-moving_plan_.rank_offsets[rank]-moving_plan_.bulk_counts[rank];
    }
    result["moving_source_policy_tiles"]=nb::make_tuple(source_tiles[0],source_tiles[1],source_tiles[2]);
    result["moving_source_lowering"]="ordered_coincident_runtime";
    result["moving_geometry_lowering"]="canonical_geometry";
    nb::list source_counts;for(int c=0;c<6;++c)source_counts.append(base_inputs_[source_offset_+3*(3+c)].dims[0]);
    result["moving_source_counts"]=source_counts;
''' + text[b:]
    a = text.index("    const int32_t*plane_starts[9]")
    b = text.index("    constexpr int axes[6]", a)
    text = text[:a] + text[b:]
    text = text.replace("if(coeff.dims[0]<=1||!((attributes_.coincident_source_group_mask>>g)&1))throw std::invalid_argument(\"moving initially admits only multi-source coincident arithmetic\");",
        "if(coeff.dims[0]>1&&!((attributes_.coincident_source_group_mask>>g)&1))throw std::invalid_argument(\"moving requires coincident starts within each multi-source component\");")
    text = text.replace("const bool compact_enabled=(moving_workspace_.data_flags&beamz::cuda::moving::kBytePackedE)&&(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH);",
        "const bool compact_enabled=bool(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH);")
    text = text.replace("    const bool source_selector=BeamzResearchSourcePendingEnabled;append_bytes(&source_selector,sizeof(bool));\n", "")
    text = text.replace("    const bool plane_selector=BeamzResearchSourcePlanePairEnabled;append_bytes(&plane_selector,sizeof(bool));\n", "")
    a = text.index('  module.attr("moving_source_plane_pair")')
    b = text.index('  module.attr("moving_noio_uniform_1024")', a)
    text = text[:a] + text[b:]
    text = text.replace('NB_MODULE(_cuda_native, module) {', '''NB_MODULE(_cuda_native, module) {
  module.attr("unified_temporal")=true;
  module.attr("moving_unified_sources")=true;
  module.attr("moving_block_uniform_e")=true;''')
    p.write_text(text)
    for file in (src / "temporal_native.cuh", src / "native_moving.h", p):
        for forbidden in ("FixedNoIOCube", "FixedCubeCached", "SourceMask20", "Source54", "BeamzResearchSource", "presence==20", "presence==54"):
            if forbidden in file.read_text():
                raise ValueError(f"Benchmark selector survived in {file.name}: {forbidden}")
    ninja = Path(json.loads((Path(proof["parent"]) / "configure-command.json").read_text())[0]).with_name("ninja")
    targets = {"CMakeFiles/beamz_cuda_moving.dir/src/native_moving_kernel.cu.o": "native_moving_kernel.cu.o",
               "CMakeFiles/_cuda_native.dir/src/native_moving.cc.o": "native_moving.cc.o"}
    commands = []
    for target, obj in targets.items():
        command = shlex.split(subprocess.check_output([str(ninja), "-C", str(previous), "-t", "commands", target], text=True).splitlines()[-1])
        command = [v.replace(str(previous_source), str(source)) for v in command]
        commands.append([str(output / obj) + v[len(target):] if v.startswith(target) else v for v in command])
    commands.append([v.replace(str(parent / "source"), str(source)).replace(str(parent / "native_extension.cc.o"), str(output / "native_extension.cc.o")) for v in proof["compile_command"]])
    module = output / Path(proof["module"]).name
    replacements = {str(previous / target): str(output / obj) for target, obj in targets.items()}
    replacements.update({str(parent / "native_extension.cc.o"): str(output / "native_extension.cc.o"), proof["module"]: str(module)})
    link = [replacements.get(v, v) for v in proof["link_command"]]
    manifest = dict(parent=str(parent), parent_module_sha256=hashlib.sha256(Path(proof["module"]).read_bytes()).hexdigest(),
        status="building", commands=commands, link_command=link, source_lowering="ordered_coincident_runtime",
        geometry_lowering="canonical_geometry", per_benchmark_selectors=False,
        source_hashes={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()})
    (output / "proof.json").write_text(json.dumps(manifest, indent=2) + "\n")
    reused = set()
    if args.reuse_kernels:
        reuse = args.reuse_kernels.resolve()
        prior = json.loads((reuse / "proof.json").read_text())
        if prior["status"] != "complete":
            raise ValueError("Kernel reuse requires a completed immutable build")
        excluded = {"src/native_extension.cc", "src/native_zero_reset.inc"}
        old_hashes = {k: v for k, v in prior["source_hashes"].items() if k not in excluded}
        new_hashes = {k: v for k, v in manifest["source_hashes"].items() if k not in excluded}
        if old_hashes != new_hashes:
            raise ValueError("Kernel reuse source comparison failed")
        for command, name in zip(commands[:2], ("native_moving_kernel.cu.o", "native_moving.cc.o")):
            normalized = [v.replace(str(output), str(reuse)) for v in command]
            expected = prior["commands"][len(reused)]
            if normalized != expected:
                raise ValueError("Kernel reuse compile options differ")
            shutil.copyfile(reuse / name, output / name)
            reused.add(name)
        manifest["reused_kernel_objects_from"] = str(reuse)
    with (output / "build.log").open("w") as log:
        for command in commands[len(reused):] + [link]:
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest.update(status="complete", module=str(module), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest())
    (output / "proof.json").write_text(json.dumps(manifest, indent=2) + "\n")
    module.chmod(0o444)
    print(json.dumps({k: manifest[k] for k in ("status", "module", "module_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
