"""Exercise the actual compact host partition independently of CUDA execution."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "compact_noio_patcher", ROOT / "scripts/patch_cuda_native_compact_noio.py"
)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


def test_compiled_compact_partition_preserves_queues_slots_and_empty_ranges(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    # Compile the real planner and emitted partition helper, using a private
    # header copy. The original canonical source remains untouched.
    header = (ROOT / "cuda/src/native_moving.h").read_text()
    header = PATCHER.once(
        header,
        "  std::vector<int32_t> bulk_counts,policy_offsets;",
        "  std::vector<int32_t> bulk_counts,policy_offsets,uniform_noio_counts;",
    )
    header += (
        "\nvoid BeamzCompactNativeMovingUniformNoIO(BeamzNativeMovingPlan&,bool);\n"
    )
    (tmp_path / "native_moving.h").write_text(header)
    host = tmp_path / "host.cc"
    host.write_text(
        (ROOT / "cuda/src/native_moving.cc").read_text() + PATCHER.HOST_FUNCTION
    )
    proof = tmp_path / "proof.cc"
    proof.write_text(r"""
#include "native_moving.h"
#include <cassert>
#include <set>
int main(){
 using namespace beamz::cuda::moving;
 int empty_uniform=0,empty_fallback=0,mixed=0;
 for(int n:{33,64})for(int depth:{4,16})for(bool enabled:{false,true})for(int pattern=0;pattern<3;++pattern){
  BeamzBuffer fields[6]{};const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
  for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=n+add[c][a];}
  auto p=BeamzBuildNativeMovingPlan(fields,depth,{{n/2,n/2,n/2}},nullptr,nullptr,12,true);
  for(size_t tile=0;tile<p.tile_flags.size();++tile)if(Policy(p.tile_flags[tile])==1&&(pattern==1||(pattern==2&&tile%2)))p.tile_flags[tile]|=64;
  const auto old=p;BeamzCompactNativeMovingUniformNoIO(p,enabled);
  assert(p.rank_offsets==old.rank_offsets&&p.policy_offsets==old.policy_offsets&&p.bulk_counts==old.bulk_counts&&p.tile_flags==old.tile_flags);
  assert(p.trail_floats==old.trail_floats&&p.metadata_bytes==old.metadata_bytes);
  std::set<int>seen;
  for(size_t rank=0;rank<p.bulk_counts.size();++rank){
   const int rb=p.rank_offsets[rank],re=p.rank_offsets[rank+1],a=p.policy_offsets[rank*7+1],b=p.policy_offsets[rank*7+2];
   std::vector<int>expected;int count=0;
   if(enabled){for(int i=a;i<b;++i)if(old.tile_flags[old.tile_ids[i]]&64){expected.push_back(old.tile_ids[i]);++count;}for(int i=a;i<b;++i)if(!(old.tile_flags[old.tile_ids[i]]&64))expected.push_back(old.tile_ids[i]);}
   else expected.assign(old.tile_ids.begin()+a,old.tile_ids.begin()+b);
   assert(std::equal(expected.begin(),expected.end(),p.tile_ids.begin()+a));assert(p.uniform_noio_counts[rank]==count);
   assert(std::equal(old.tile_ids.begin()+rb,old.tile_ids.begin()+a,p.tile_ids.begin()+rb));assert(std::equal(old.tile_ids.begin()+b,old.tile_ids.begin()+re,p.tile_ids.begin()+b));
   empty_uniform+=count==0;empty_fallback+=count==b-a;mixed+=count>0&&count<b-a;
   for(int i=rb;i<re;++i){const int tile=p.tile_ids[i];assert(seen.insert(tile).second&&p.slots[tile]==i-rb);
    const int z=tile/(p.counts[1]*p.counts[2]),y=(tile/p.counts[2])%p.counts[1],x=tile%p.counts[2];assert(z+y+x==int(rank));
    for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
     const int producer=tile-dz*p.counts[1]*p.counts[2]-dy*p.counts[2]-dx,pr=rank-dz-dy-dx,slot=p.slots[producer];
     assert(p.tile_ids[p.rank_offsets[pr]+slot]==producer);assert((pr&3)!=(rank&3));
    }
   }
  }
  assert(seen.size()==p.tile_ids.size());const auto once=p;BeamzCompactNativeMovingUniformNoIO(p,enabled);assert(once.tile_ids==p.tile_ids&&once.slots==p.slots&&once.uniform_noio_counts==p.uniform_noio_counts);
 }
 assert(empty_uniform&&empty_fallback&&mixed);
}
""")
    binary = tmp_path / "proof"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "cuda/src"),
            str(proof),
            str(host),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_compact_patcher_rejects_unbound_parent_and_duplicate_anchor():
    with pytest.raises(ValueError, match="Exact 4941 parent"):
        PATCHER.transformed(dict.fromkeys(PATCHER.PARENT_HASHES, b"changed"))
    with pytest.raises(ValueError, match="one exact"):
        PATCHER.once("anchor anchor", "anchor", "replacement")
