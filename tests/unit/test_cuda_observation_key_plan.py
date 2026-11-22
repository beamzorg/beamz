"""Actual sparse key plan/remap matches the established dense-mask planner."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.cuda_source_layout import CUDA_INCLUDE_FLAGS


@pytest.fixture(scope="module")
def key_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    folder = tmp_path_factory.mktemp("observation-keys")
    root = Path(__file__).resolve().parents[2]
    source = folder / "keys.cc"
    source.write_text(r"""
#include "native_deep.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include "native_observation_rows.inc"
#include "native_observation_keys.inc"
int main(int argc,char**argv){
 std::ifstream in(argv[1],std::ios::binary);int32_t dims[18],shape[4];in.read((char*)dims,72);in.read((char*)shape,16);
 BeamzBuffer fields[6]{},indices{};for(int c=0;c<6;++c){fields[c].rank=3;for(int a=0;a<3;++a)fields[c].dims[a]=dims[3*c+a];}
 size_t count=1;indices.rank=4;indices.element_type=kBeamzS32;for(int a=0;a<4;++a){indices.dims[a]=shape[a];count*=shape[a];}
 std::vector<int32_t>input(count);in.read((char*)input.data(),count*4);if(!in)throw std::runtime_error("truncated probe");
 if(BeamzMovingIndexWidth(8,65535)!=2||BeamzMovingIndexWidth(8,65536)!=3||
    BeamzMovingIndexWidth(8,16777215)!=3||BeamzMovingIndexWidth(8,16777216)!=4||
    BeamzMovingIndexWidth(1,1)!=4)return 12;
 const auto old=BeamzBuildNativeDeepPlan(fields,indices,input.data());
 auto next=BeamzBuildNativeMovingKeyPlan(fields,indices,input.data());
 if(next.remapped_indices.capacity())return 3;
 const auto rows=BeamzBuildNativeMovingRowObservation(fields,next.keys);
 BeamzRemapNativeMovingRows(fields,indices,input.data(),next,rows);
 if(old.keys!=next.keys||old.field_base!=next.field_base||old.mask_shape!=next.mask_shape||old.remapped_indices!=next.remapped_indices||!next.publication_words.empty())return 1;
 // Counts change only the private remap layout. Valid padded/masked keys stay
 // in the canonical full key set, including when every monitor has NP=0.
 for(int pattern=0;pattern<4;++pattern){
  std::vector<int32_t>offsets(size_t(shape[0])+1,0),expected;
  const size_t stride=size_t(shape[2])*shape[3];
  for(int m=0;m<shape[0];++m){
   const int np=pattern==0?0:pattern==1?shape[2]:pattern==2?shape[2]/2:(m%3)*shape[2]/2;
   const size_t width=size_t(np)*shape[3];offsets[m+1]=offsets[m]+6*width;
   for(int c=0;c<6;++c){const auto begin=old.remapped_indices.begin()+(6*m+c)*stride;expected.insert(expected.end(),begin,begin+width);}
  }
  auto compact=BeamzBuildNativeMovingKeyPlan(fields,indices,input.data());
  BeamzRemapNativeMovingRows(fields,indices,input.data(),compact,rows,offsets);
  if(compact.keys!=old.keys||compact.field_base!=old.field_base||compact.mask_shape!=old.mask_shape||compact.remapped_indices!=expected)return 4;
  if(compact.remapped_indices.capacity()!=expected.size())return 5;
  if(shape[3]==8)for(size_t width:{size_t(2),size_t(3)}){
   if(width<BeamzMovingIndexWidth(8,compact.keys.size()))continue;
   BeamzRemapNativeMovingRows(fields,indices,input.data(),compact,rows,offsets,nullptr,0,width);
   if(compact.remapped_indices.size()*4!=expected.size()*width)return 10;
   const auto* data=reinterpret_cast<const unsigned char*>(compact.remapped_indices.data());
   for(size_t i=0;i<expected.size();++i){
    const auto* point=data+(i/8)*8*width;
    const int lane=int(i%8);
    unsigned decoded=point[2*lane]|(unsigned(point[2*lane+1])<<8);
    if(width==3)decoded|=unsigned(point[16+lane])<<16;
    const int result=decoded==((1u<<(width*8))-1)?-1:int(decoded);
    if(result!=expected[i])return 11;
   }
  }
 }
 std::vector<int32_t>invalid(size_t(shape[0])+1,0);invalid[0]=1;
 const auto rejects=[&](const std::vector<int32_t>&offsets){
  try{BeamzRemapNativeMovingRows(fields,indices,input.data(),next,rows,offsets);return false;}catch(const std::invalid_argument&){return true;}
 };
 if(!rejects(invalid)||!rejects(std::vector<int32_t>(size_t(shape[0])+2,0)))return 6;
 if(shape[0])for(int end:{-6,1,int(6*(size_t(shape[2])*shape[3]+1))}){
  std::fill(invalid.begin(),invalid.end(),end);invalid[0]=0;if(!rejects(invalid))return 7;
 }
 if(shape[0]&&shape[2]&&shape[3]>1){std::fill(invalid.begin(),invalid.end(),6);invalid[0]=0;if(!rejects(invalid))return 8;}
 // The remapper must detect a corrupted descriptor rather than silently publish
 // a sample to the wrong raw field slot.
 if(!next.keys.empty()){
  auto broken=rows;for(auto&row:broken.leaf)row=0;
  try{BeamzRemapNativeMovingRows(fields,indices,input.data(),next,broken);return 2;}catch(const std::logic_error&){}
  std::vector<int32_t>offsets(size_t(shape[0])+1,0);for(int m=0;m<shape[0];++m)offsets[m+1]=offsets[m]+6*shape[2]*shape[3];
  try{BeamzRemapNativeMovingRows(fields,indices,input.data(),next,broken,offsets);return 9;}catch(const std::logic_error&){}
 }
 return 0;
}
""")
    binary = folder / "keys"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++17",
            *CUDA_INCLUDE_FLAGS,
            "-I",
            str(root / "scripts"),
            str(source),
            str(root / "cuda/src/runtime/native_deep.cc"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


@pytest.mark.parametrize("monitor_count", [0, 1, 3])
@pytest.mark.parametrize("mode", ["empty", "duplicates", "ragged", "runs"])
def test_keys_and_constant_time_remap_preserve_all_monitor_semantics(
    key_probe, tmp_path, monitor_count, mode
):
    additions = np.asarray(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    )
    dimensions = np.asarray([17, 19, 33]) + additions
    shape = np.asarray([monitor_count, 6, 17, 4], np.int32)
    indices = np.full(shape, -1, np.int32)
    rng = np.random.default_rng(1501)
    for field in range(6):
        if mode == "duplicates":
            indices[:, field] = np.asarray([0, 5, 5, 23], np.int32)
        elif mode == "ragged":
            size = int(np.prod(dimensions[(field + 3) % 6]))
            indices[:, field] = rng.integers(
                -5, size + 5, size=(monitor_count, 17, 4), dtype=np.int32
            )
        elif mode == "runs":
            indices[:, field] = 5
            indices[:, field, ::4] = -1
            indices[:, field, 8:10] = 0
    path = tmp_path / "packet.bin"
    path.write_bytes(
        dimensions.astype(np.int32).tobytes() + shape.tobytes() + indices.tobytes()
    )
    subprocess.run([str(key_probe), str(path)], check=True)


@pytest.mark.parametrize(
    "monitor_count,max_np,neighbors",
    [
        (0, 0, 0),
        (0, 17, 0),
        (0, 0, 4),
        (1, 0, 0),
        (3, 17, 0),
        (3, 0, 4),
        (1, 1, 1),
        (3, 7, 2),
        (3, 17, 8),
        (1, 1800, 8),
    ],
)
def test_compact_remap_empty_supports_and_neighbor_widths(
    key_probe, tmp_path, monitor_count, max_np, neighbors
):
    # K=0 is rejected by the production MovingMonitorRows admission contract;
    # these host planners also handle its empty storage without division by zero.
    dimensions = np.tile([17, 19, 33], (6, 1)).astype(np.int32)
    shape = np.asarray([monitor_count, 6, max_np, neighbors], np.int32)
    indices = np.zeros(shape, np.int32)
    if max_np == 1800:
        dimensions[:, :] = [20, 25, 32]
        indices[:] = np.arange(max_np * neighbors, dtype=np.int32).reshape(
            max_np, neighbors
        )
        indices[..., ::73, 7] = -1
    path = tmp_path / "empty-support.bin"
    path.write_bytes(dimensions.tobytes() + shape.tobytes() + indices.tobytes())
    subprocess.run([str(key_probe), str(path)], check=True)
