"""Execute the actual three-parent builder and captured host scheduler on CPU."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


def validate(candidate):
    candidate=Path(candidate).resolve();source=candidate/'source/src'
    proof=json.loads((candidate/'frontier-cohort-compiler-proof.json').read_text())
    consumer=(source/'asymmetric_native.cuh').read_text();header=(source/'native_cohort_dependencies.h').read_text()
    if hashlib.sha256(consumer.encode()).hexdigest()!=proof['consumer_sha256'] or hashlib.sha256(header.encode()).hexdigest()!=proof['dependency_builder_sha256']:
        raise ValueError('Actual scheduler source changed')
    directory=candidate/'host-frontier-cohort-proof';directory.mkdir(exist_ok=False)
    code='''#include "native_cohort_dependencies.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
int main(){using namespace beamz::cuda::moving;size_t cases=0,checks=0,edges=0;
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(auto dims:std::vector<std::array<int,3>>{{25,25,25},{33,47,61},{64,96,128},{136,136,136},{512,512,512},{568,568,568},{6000,25,25}})
 for(int depth:{4,16,32})for(int pattern:{0,1}){
  BeamzBuffer f[6]{};for(int c=0;c<6;++c){f[c].rank=3;f[c].element_type=kBeamzF32;for(int a=0;a<3;++a)f[c].dims[a]=dims[a]+add[c][a];}
  auto p=BeamzBuildNativeMovingPlan(f,depth,{{0,0,0},{dims[0]/2,dims[1]/2,dims[2]/2}},nullptr,nullptr,12,true);
  if(pattern){const uint8_t flags[6]={0,2,1,16,17,3};for(size_t t=0;t<p.tile_flags.size();++t)p.tile_flags[t]=flags[(t*1103515245u+12345u)%6];p.policy_offsets.clear();
   for(size_t r=0;r<p.bulk_counts.size();++r){auto a=p.tile_ids.begin()+p.rank_offsets[r],b=p.tile_ids.begin()+p.rank_offsets[r+1];std::stable_sort(a,b,[&](int a,int b){return Policy(p.tile_flags[a])<Policy(p.tile_flags[b]);});auto it=a;
    for(int k=0;k<6;++k){p.policy_offsets.push_back(it-p.tile_ids.begin());while(it!=b&&Policy(p.tile_flags[*it])==k)++it;}p.policy_offsets.push_back(b-p.tile_ids.begin());for(auto it=a;it!=b;++it)p.slots[*it]=it-a;
   }
  }
  const auto g=BeamzBuildNativeMovingCohorts(p);if(g.reuse_links)return 5;
  const size_t nodes=g.offsets.size()-1,words=(nodes+63)/64;
  std::vector<std::vector<uint64_t>>anc(nodes,std::vector<uint64_t>(words));int previous[6]={-1,-1,-1,-1,-1,-1};
  for(size_t c=0;c<nodes;++c){const int r=c/6,k=c%6;if(p.policy_offsets[r*7+k]==p.policy_offsets[r*7+k+1])continue;
   auto include=[&](int d){if(d<0)return;if(d>=int(c))throw 1;for(size_t w=0;w<words;++w)anc[c][w]|=anc[d][w];anc[c][d/64]|=uint64_t(1)<<(d%64);};
   include(previous[k]);for(int i=g.offsets[c];i<g.offsets[c+1];++i)include(g.dependencies[i]);previous[k]=c;
  }
  const int ny=p.counts[1],nx=p.counts[2];
  for(size_t tile=0;tile<p.tile_ids.size();++tile){const int z=tile/(ny*nx),y=(tile/nx)%ny,x=tile%nx,c=g.tile_cohort[tile];
   for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
    const int d=g.tile_cohort[tile-dz*ny*nx-dy*nx-dx];if(!(anc[c][d/64]&(uint64_t(1)<<(d%64))))return 4;++checks;
   }
  }
  ++cases;edges+=g.dependencies.size();
 }
 std::cout<<"{\\"cases\\":"<<cases<<",\\"all_seven_neighbor_reachability_checks\\":"<<checks<<",\\"edges\\":"<<edges<<",\\"passed\\":true}\\n";
}
'''
    rows=[]
    for name in ('actual','missing-x-parent'):
        folder=directory/name;folder.mkdir()
        trial=header if name=='actual' else header.replace('for(int bits:{1,2,4})','for(int bits:{1,2})')
        (folder/'native_cohort_dependencies.h').write_text(trial)
        src=folder/'proof.cc';binary=folder/'proof';src.write_text(code)
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=undefined','-I'+str(folder),'-I'+str(source),str(src),str(source/'native_moving.cc'),'-o',str(binary)],capture_output=True,text=True,check=True)
        r=subprocess.run([str(binary)],capture_output=True,text=True)
        if (r.returncode!=0 if name=='actual' else r.returncode==0):raise RuntimeError(name+r.stdout+r.stderr)
        rows.append(dict(name=name,returncode=r.returncode,stdout=r.stdout))
    # Reuse the capture-frontier mock, but execute the actual current scheduler
    # text and its current source-specialized dispatch signatures.
    schedule='template<int Depth,int Layout,bool IsotropicH>\n'+block(consumer,'inline cudaError_t LaunchCohortRanks(')[2]
    (directory/'actual-schedule.inc').write_text(schedule)
    launch=Path('scripts/prove_cuda_native_cohort_launch.cc').read_text()
    launch=launch.replace('#include "cuda_native_cohort_dependencies.h"','#include "native_cohort_dependencies.h"')
    stub='''template<int D,int L,bool Iso,int Policy>static int LaunchSourcePending(cudaStream_t s,const Parameters&,const int32_t*i,int n,unsigned,bool){return launch(s,i,n,Policy);}
'''
    launch=launch.replace('#include "cuda_native_cohort_schedule.inc"',stub+'#include "actual-schedule.inc"')
    src=directory/'capture.cc';binary=directory/'capture';src.write_text(launch)
    subprocess.run(['g++','-std=c++17','-O2','-fsanitize=undefined','-I'+str(directory),'-I'+str(source),str(src),str(source/'native_moving.cc'),'-o',str(binary)],capture_output=True,text=True,check=True)
    r=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],builder_sha256=proof['dependency_builder_sha256'],rows=rows,capture=json.loads(r.stdout),
                scope='Actual three-parent builder proves all seven producer paths over42prepared/ragged/synthetic-policy cases; missing-X negative; actual class-specialized host scheduler captures/rejoins three macros. Physical-value WAR legality uses the separate token proof. GPU checks separate.')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
