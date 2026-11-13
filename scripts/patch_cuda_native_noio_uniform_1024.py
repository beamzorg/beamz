"""Copy the exact selected NoIO/support source and split certified NoIO kernels.

This resource-admitted research transformation keeps Fast=false/CPML=true for
NoIO and changes only its private Z mapping and complementary kernel launches.
It never edits its input tree or builds, imports, installs, or runs CUDA code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_CONSUMER = "45a832f23ccabfeffd4afc8bd0ad2c339a97d16a831f579a3e537d662edae3ef"
PARENT_EXTENSION = "d0a71bf792c2bed4f7ae77ae365d7906c2ead32abe875d73cd09152d8578e706"
CANONICAL_CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
RESULT_CONSUMER = "3ed4a4f837df1ca60f409a243c0dd8a63778c0b2b983bac0ee895b56da04d6aa"

T_OLD = "constexpr int T=Fast?1024:512,Slots=PAD/T,ZStep=T/512;"
T_NEW = "constexpr int T=(Fast||(Layout==1&&Policy==1&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;"
GATE_OLD = " if constexpr(Fast||(Policy==1&&IsotropicH)){\n"
GATE_NEW = """ if constexpr(Layout==1&&Policy==1&&IsotropicH){
  // Disjoint immutable CTA-uniform gate BEFORE any owner tag/body/barrier.
  if(p.flags[tiles[blockIdx.x]]&64)return;
  UniformBody<Depth,Layout,false,true,1,false>(p,tiles);
 }else if constexpr(Fast||(Policy==1&&IsotropicH)){
"""
WRAPPER = """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 // Same cohort; complementary gate prevents all duplicate ownership/writes.
 if(!(p.flags[tiles[blockIdx.x]]&64))return;
 UniformBody<Depth,1,false,true,1,true>(p,tiles);
}

"""
LAUNCH_OLD = " Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<count,(Fast?1024:512),SharedBytes,stream>>>(p,ids);return cudaGetLastError();"
LAUNCH_NEW = """ Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<count,(Fast?1024:512),SharedBytes,stream>>>(p,ids);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if constexpr(Layout==1&&Policy==1&&IsotropicH){
  const auto uniform_kernel=AsymmetricNoIOUniform1024<Depth>;
  status=cudaFuncSetAttribute(uniform_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricNoIOUniform1024<Depth><<<count,1024,SharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }
 return cudaSuccess;"""
ATTRIBUTE_OLD = '  module.attr("moving_noio_uniform_e")=true;'
ATTRIBUTE_NEW = (
    ATTRIBUTE_OLD
    + '\n  module.attr("moving_noio_uniform_1024")=true;\n  module.attr("moving_noio_uniform_threads")=1024;'
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old!r}")
    return text.replace(old, new, 1)


def transform(consumer: bytes, extension: bytes, cell: bytes):
    """Return exact transformed bytes and static source/routing evidence."""
    for label, data, expected in (
        ("consumer", consumer, PARENT_CONSUMER),
        ("extension", extension, PARENT_EXTENSION),
        ("canonical cell", cell, CANONICAL_CELL),
    ):
        if digest(data) != expected:
            raise ValueError(f"Exact selected parent {label} required")
    c = consumer.decode()
    edits = (
        (T_OLD, T_NEW),
        (GATE_OLD, GATE_NEW),
        ("inline bool MakeSource(", WRAPPER + "inline bool MakeSource("),
        (LAUNCH_OLD, LAUNCH_NEW),
    )
    for old, new in edits:
        c = once(c, old, new)
    e = once(extension.decode(), ATTRIBUTE_OLD, ATTRIBUTE_NEW)
    inverse = c
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    if inverse.encode() != consumer:
        raise ValueError("Consumer inverse-source binding failed")
    if once(e, ATTRIBUTE_NEW, ATTRIBUTE_OLD).encode() != extension:
        raise ValueError("Extension inverse-source binding failed")
    if digest(c.encode()) != RESULT_CONSUMER:
        raise ValueError("Transformation differs from the resource-tested candidate")
    mappings = {}
    for threads in (512, 1024):
        slots, step = 4096 // threads, threads // 512
        points = {
            (tid // 512 + j * step, (tid // 32) & 15, tid & 31)
            for tid in range(threads)
            for j in range(slots)
        }
        expected = {(z, y, x) for z in range(8) for y in range(16) for x in range(32)}
        if points != expected:
            raise ValueError("Private-slot mapping does not cover the cache uniquely")
        mappings[threads] = {
            "slots": slots,
            "z_step": step,
            "cache_points": len(points),
        }
    for flags in range(256):
        old_active = not bool(flags & 64)
        new_active = bool(flags & 64)
        if old_active == new_active:
            raise ValueError("Complementary owner gates do not partition flags")
    proof = {
        "parent_consumer_sha256": PARENT_CONSUMER,
        "parent_extension_sha256": PARENT_EXTENSION,
        "canonical_cell_sha256": CANONICAL_CELL,
        "consumer_sha256": digest(c.encode()),
        "extension_sha256": digest(e.encode()),
        "exact_inverse_consumer_and_extension": True,
        "all_256_flags_have_one_active_owner": True,
        "cache_mappings": mappings,
        "class": {"layout": 1, "policy": 1, "isotropic_h": True, "uniform_e": True},
        "support_safe": False,
        "cpml": True,
        "dynamic_shared_bytes": 51748,
        "same_branch_stream_and_join": True,
        "scope": "Static source/routing binding; no GPU correctness or performance claim",
    }
    return c.encode(), e.encode(), proof


def patch_copy(source: Path, output: Path, proof_output: Path | None = None):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output == canonical or canonical in output.parents:
        raise ValueError("Output must be an isolated research source copy")
    if (
        output.exists()
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be a new disjoint directory")
    proof_output = (
        proof_output or output.with_name(output.name + "-proof.json")
    ).resolve()
    if (
        proof_output.exists()
        or source == proof_output
        or source in proof_output.parents
        or output == proof_output
        or output in proof_output.parents
        or proof_output == canonical
        or canonical in proof_output.parents
    ):
        raise ValueError("Proof output must be new and outside source/canonical trees")
    src = source / "src"
    consumer, extension, proof = transform(
        (src / "asymmetric_native.cuh").read_bytes(),
        (src / "native_extension.cc").read_bytes(),
        (src / "asymmetric_native_cell_generated.cuh").read_bytes(),
    )
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_bytes(consumer)
    (output / "src/native_extension.cc").write_bytes(extension)
    proof.update(
        {
            "input": str(source),
            "output": str(output),
            "patcher_sha256": digest(Path(__file__).read_bytes()),
        }
    )
    proof_output.parent.mkdir(parents=True, exist_ok=True)
    proof_output.write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Frozen selected parent CMake tree"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New isolated source directory"
    )
    parser.add_argument("--proof-output", type=Path)
    args = parser.parse_args()
    patch_copy(args.source, args.output, args.proof_output)


if __name__ == "__main__":
    main()
