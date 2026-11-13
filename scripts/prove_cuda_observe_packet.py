"""Prove policy2 raw-store retiming and compile its actual packet against an oracle.

Tokens stand for arbitrary post-source FP32 bits, so the schedule proof does
not assume zero fields or particular arithmetic results. The C++ check executes
actual emitted metadata/store code with finite, signed-zero and NaN payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "observe_packet_patcher", ROOT / "scripts/patch_cuda_native_observe_packet.py"
)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)
OWNED = (7, 15, 31)
SUPPORT_HELPER = "__device__ __forceinline__ unsigned NativeSupportMask(const Parameters&p,int z,int y,int x){\n const int nz=p.phase[0].own[0].z,ny=p.phase[0].own[0].y,nx=p.phase[0].own[1].x;\n if(z<0||y<0||x<0||z>nz||y>ny||x>nx)return 0;\n const unsigned iz=z<nz,iy=y<ny,ix=x<nx;\n return (iz&&iy)|((iz&&ix)<<1)|((iy&&ix)<<2)|(ix<<3)|(iy<<4)|(iz<<5);\n}\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_writes(depth, wrong_version=False, stale_h=False, high_only=False):
    points = [(1, 2, 3), (7, 15, 31), (30, 32, 32), (61, 45, 62)]
    # Every raw ID is unique by field+point; duplicates in detector records
    # share this same ID. Include an interior point that is not a high surface.
    raw = {
        (field, point): i
        for i, (field, point) in enumerate((f, p) for p in points for f in range(6))
    }
    eager, packet = {}, {}
    for n in range(1, depth + 1):
        for point in points:
            owner = tuple((point[a] + n) // OWNED[a] for a in range(3))
            relative = tuple(point[a] + n - owner[a] * OWNED[a] for a in range(3))
            before_h = [(f, n - 1, point, "post-source-f32") for f in range(3)]
            h = [(f, n, point, "post-source-f32") for f in range(3)]
            for f in range(3):
                eager[n, raw[f, point]] = h[f]
            # E writes only E and its separate psi. The H register values and
            # their observation tokens remain unchanged through this phase.
            e = [(f, n, point, "post-source-f32") for f in range(3, 6)]
            for f in range(3, 6):
                eager[n, raw[f, point]] = e[f - 3]
            if high_only and not any(relative[a] == OWNED[a] - 1 for a in range(3)):
                continue
            values = (before_h if stale_h else h) + e
            for f, value in enumerate(values):
                key = (n + int(wrong_version), raw[f, point])
                if key in packet:
                    raise AssertionError("duplicate raw writer")
                packet[key] = value
    return eager == packet, len(eager)


CPP = (
    r"""
#include "native_moving.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#define __device__
#define __forceinline__ inline
#define BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION 1
template<class T>T __ldg(const T*p){return *p;}
struct Field {int z,y,x;};struct Phase {Field own[3];};
struct Parameters {
 beamz::cuda::deep::Observation observation;Phase phase[2];
 int observation_ny,observation_nx;const int32_t*observation_tiles;const int32_t*observation_leaf;
};
@SUPPORT@
@PACKET@
uint32_t bits(float f){uint32_t b;std::memcpy(&b,&f,4);return b;}
float fp(uint32_t b){float f;std::memcpy(&f,&b,4);return f;}
uint32_t value(int f,int n,int z,int y,int x){
 const uint32_t k=uint32_t(f+7*n+11*z+13*y+17*x);
 switch(k%9){case0:return 0;case1:return 0x80000000u;case2:return 0x7fc12345u;case3:return 0xffc12345u;default:return 0x3f000000u^(k*2654435761u);}
}
int main(){
 size_t stores=0,calls=0;const uint32_t sentinel=0xdeadbeefu;
 for(const auto dims: {std::array<int,3>{9,11,13},std::array<int,3>{17,19,35},std::array<int,3>{33,37,41}}){
  const int nz=dims[0],ny=dims[1],nx=dims[2],mz=nz+1,my=ny+1,mx=nx+1;
  Parameters p{};p.phase[0].own[0]={nz,ny,nx+1};p.phase[0].own[1]={nz,ny+1,nx};p.phase[0].own[2]={nz+1,ny,nx};
  const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
  std::map<std::tuple<int,int,int,int>,int> raw;
  for(int z=0;z<mz;++z)for(int y=0;y<my;++y)for(int x=0;x<mx;++x)for(int f=0;f<6;++f)
   if(z<nz+add[f][0]&&y<ny+add[f][1]&&x<nx+add[f][2]&&((z*17+y*13+x*11+f)%47==0||(z==1&&y==2&&x==3)))raw[{f,z,y,x}]=0;
  int key_count=0;for(auto&k:raw)k.second=key_count++;
  BeamzBuffer fields[6]{};int64_t bases[6]{};int64_t total=0;
  for(int f=0;f<6;++f){bases[f]=total;fields[f].rank=3;fields[f].element_type=kBeamzF32;
   fields[f].dims[0]=nz+add[f][0];fields[f].dims[1]=ny+add[f][1];fields[f].dims[2]=nx+add[f][2];
   total+=fields[f].dims[0]*fields[f].dims[1]*fields[f].dims[2];
  }
  std::vector<int64_t>keys;
  for(const auto&it:raw){int f,z,y,x;std::tie(f,z,y,x)=it.first;keys.push_back(bases[f]+(z*fields[f].dims[1]+y)*fields[f].dims[2]+x);}
  const auto mapped=BeamzBuildNativeMovingSparseObservation(fields,keys);
  p.observation_ny=mapped.grid[1];p.observation_nx=mapped.grid[2];
  const auto&coarse=mapped.coarse;const auto&leaf=mapped.leaf;
  p.observation_tiles=coarse.data();p.observation_leaf=leaf.data();p.observation.key_count=key_count;
  for(bool clipped:{false,true}){
   p.observation.mask_nz=clipped?nz:mz;p.observation.mask_ny=my;p.observation.mask_nx=mx;
   std::vector<float>actual(size_t(17)*key_count+32,fp(sentinel));p.observation.values=actual.data()+16;
   std::vector<uint32_t>expected(size_t(17)*key_count,sentinel);
   for(int n=1;n<=17;++n){
    for(int z=-1;z<=mz;++z)for(int y=-1;y<=my;++y)for(int x=-1;x<=mx;++x){
     ObservePacket(p,n,z,y,x,fp(value(0,n,z,y,x)),fp(value(1,n,z,y,x)),fp(value(2,n,z,y,x)),fp(value(3,n,z,y,x)),fp(value(4,n,z,y,x)),fp(value(5,n,z,y,x)));++calls;
    }
    // Independent old per-component write oracle: original valid key+mask.
    for(const auto&it:raw){int f,z,y,x;std::tie(f,z,y,x)=it.first;if(z>=p.observation.mask_nz)continue;
     expected[size_t(n-1)*key_count+it.second]=value(f,n,z,y,x);++stores;
    }
   }
   for(size_t i=0;i<expected.size();++i)assert(bits(actual[i+16])==expected[i]);
   for(int i=0;i<16;++i)assert(bits(actual[i])==sentinel&&bits(actual[actual.size()-1-i])==sentinel);
  }
 }
 std::cout<<"{\"actual_packet_calls\":"<<calls<<",\"matched_raw_stores\":"<<stores<<",\"guard_words_checked\":192,\"pass\":true}\n";
}
""".replace("case0:", "case 0:")
    .replace("case1:", "case 1:")
    .replace("case2:", "case 2:")
    .replace("case3:", "case 3:")
)


def prove(source, output):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    consumer = (source / "src/asymmetric_native.cuh").read_text()
    extension = (source / "src/native_extension.cc").read_text()
    inverse = PATCHER.once(
        consumer, PATCHER.PUBLISH_REPLACEMENT, PATCHER.PUBLISH_ANCHOR
    )
    inverse = PATCHER.once(
        inverse,
        PATCHER.PACKET + "\nstruct FieldPacket {float hx,hy,hz,ex,ey,ez;};",
        "struct FieldPacket {float hx,hy,hz,ex,ey,ez;};",
    )
    inverse = PATCHER.once(inverse, "HasObserve&&Policy!=2", "HasObserve")
    if (
        hashlib.sha256(inverse.encode()).hexdigest()
        != PATCHER.PARENT_HASHES["asymmetric_native.cuh"]
    ):
        raise ValueError(
            "Actual H/E body, environment or schedule changed beyond the admitted patch"
        )
    if (
        digest(source / "src/asymmetric_native_cell_generated.cuh")
        != PATCHER.PARENT_HASHES["asymmetric_native_cell_generated.cuh"]
    ):
        raise ValueError("Canonical cell/write-set binding changed")
    inverse_extension = PATCHER.once(
        extension, '  module.attr("moving_observe_packet_policy2")=true;\n', ""
    )
    if (
        hashlib.sha256(inverse_extension.encode()).hexdigest()
        != PATCHER.PARENT_HASHES["native_extension.cc"]
    ):
        raise ValueError("Native consumer/DFT ordering or raw exposure changed")
    if (
        consumer.count(PATCHER.PACKET) != 1
        or consumer.count(PATCHER.PUBLISH_REPLACEMENT) != 1
    ):
        raise ValueError("Actual packet/publisher source differs from admitted patch")
    if "HasObserve&&Policy!=2" not in consumer:
        raise ValueError("Policy2 eager stores remain active")
    if "size_t(version-1)*o.key_count" not in PATCHER.PACKET:
        raise ValueError("Raw rows must retain size_t addressing")
    # Native host scheduling is serialized by this exact caller sequence.
    enqueue = extension.index("int error=BeamzEnqueueNativeAsymmetric(")
    dft = extension.index("error=EnqueueMovingDft(", enqueue)
    swap = extension.index("std::swap(current_,other_)", dft)
    assert enqueue < dft < swap
    if (
        "cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(w.policy_done_events[policy]),0)"
        not in consumer
    ):
        raise ValueError("Final rank joins missing")
    cases, writes = 0, 0
    for depth in (4, 16, 33):
        ok, count = token_writes(depth)
        assert ok
        cases += 1
        writes += count
    negatives = {}
    for name, args in (
        ("phase_step_n_plus_1", {"wrong_version": True}),
        ("stale_H_from_previous_version", {"stale_h": True}),
        ("publish_only_high_surface", {"high_only": True}),
    ):
        ok, _ = token_writes(16, **args)
        assert not ok
        negatives[name] = "rejected"
    support = consumer.split(
        "__device__ __forceinline__ unsigned NativeSupportMask", 1
    )[1].split(PATCHER.PACKET, 1)[0]
    support = "__device__ __forceinline__ unsigned NativeSupportMask" + support
    cpp = CPP.replace("@SUPPORT@", support).replace("@PACKET@", PATCHER.PACKET)
    cpp = "#include <array>\n" + cpp
    program = output / "packet-oracle.cc"
    program.write_text(cpp)
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-I",
        str(source / "src"),
        str(program),
        str(source / "src/native_moving.cc"),
        "-o",
        str(output / "packet-oracle"),
    ]
    subprocess.run(command, check=True)
    result = subprocess.run(
        [str(output / "packet-oracle")], check=True, capture_output=True, text=True
    )
    actual = json.loads(result.stdout)
    proof = {
        "consumer_sha256": digest(source / "src/asymmetric_native.cuh"),
        "extension_sha256": digest(source / "src/native_extension.cc"),
        "canonical_cell_sha256": digest(
            source / "src/asymmetric_native_cell_generated.cuh"
        ),
        "proof_source_sha256": digest(Path(__file__)),
        "actual_helper_cpp_sha256": digest(program),
        "schedule_cases": cases,
        "matched_field_coordinate_version_rawID_value_tokens": writes,
        "negative_controls": negatives,
        "actual_emitted_cpp_metadata_value_oracle": actual,
        "raw_consumer_order": "all macro rank joins -> DFT -> next macro; no external raw consumer in private native session",
        "H_lifetime": "E writes only E/psi; private H survives unchanged until all-owned publisher",
        "scope": "Policy2; canonical FP32 no-source/noCPML operator; static+CPU proof, no GPU claim",
    }
    (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prove(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
