"""Execute the actual affine slab planner and check exact numerical bindings."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


def validate(candidate,parent):
    candidate,parent=Path(candidate).resolve(),Path(parent).resolve();src=candidate/'source/src'
    proof=json.loads((candidate/'macro-pipeline-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();original=(parent/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(c.encode()).hexdigest()!=proof['consumer_sha256'] or hashlib.sha256(original.encode()).hexdigest()!=proof['parent_consumer_sha256']:
        raise ValueError('Explicit numerical ancestry changed')
    for filename in ('asymmetric_native_cell_generated.cuh','yee_cell_generated.cuh'):
        if (src/filename).read_bytes()!=(parent/'src'/filename).read_bytes():raise ValueError('Canonical arithmetic header changed')
    finish=block(c,'float Finish(')[2];old=block(original,'float Finish(')[2]
    if finish.replace('+step_offset','')!=old:raise ValueError('Integer macro offset is not the only Finish change')
    native=proof.get('native_banked',False)
    for signature in ('bool Valid(','int Packed(','bool LocalActive(','int PsiOffset(','float LoadLocal(','void StoreLocal(','float Coefficient(','float Difference(','void Observe(',*(() if native else ('FieldPacket ImportPacket(',))):
        if block(c,signature)[2]!=block(original,signature)[2]:raise ValueError('Unrelated helper changed:'+signature)
    if 'p.flags[tiles[blockIdx.x]]' in c or 'p.flags[ids[blockIdx.x]]' in c:raise ValueError('Undecoded wrapper flags remain')
    if c.count('env.step_offset=MacroStepOffset<Depth>(p,tiles[blockIdx.x]);')!=3:raise ValueError('Missing actual macro time binding')
    if native:
        for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
            body=block(c,signature)[2]
            body=body.replace('env.step_offset=MacroStepOffset<Depth>(p,tiles[blockIdx.x]);','')
            body=body.replace('const int encoded=tiles[blockIdx.x],macro_index=MacroStepOffset<1>(p,encoded),tile=PhysicalTileID(p,encoded),','const int tile=tiles[blockIdx.x],')
            body=body.replace('rank=bz+by+bx+macro_index*p.macro_level_stride;','rank=bz+by+bx;')
            body=body.replace('p.slots[encoded]','p.slots[tile]').replace('uint32_t(encoded)','uint32_t(tile)')
            body=body.replace('p.slots[producer+macro_index*p.nz*p.ny*p.nx]','p.slots[producer]').replace('uint32_t(producer+macro_index*p.nz*p.ny*p.nx)','uint32_t(producer)')
            import re
            body=re.sub(r'Initial\(p,([^,]+),z,y,x,macro_index\)',r'Initial(p,\1,z,y,x)',body)
            body=re.sub(r'MacroFinalField\(p,([^,]+),macro_index\)',r'FinalField(p,\1)',body)
            body=body.replace('if(Layout!=1||n<Depth)p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=','p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=')
            body=body.replace('if(surface>=0){','if(surface>=0&&(Layout!=1||n<Depth)){')
            if body!=block(original,signature)[2]:raise ValueError('Native exact loop inverse failed:'+signature)
    planner=(src/'native_macro_slab.h').read_text()
    if hashlib.sha256(planner.encode()).hexdigest()!=proof['planner_sha256']:raise ValueError('Actual planner changed')
    directory=candidate/'host-macro-pipeline-proof';directory.mkdir(exist_ok=False)
    code='''#include "native_macro_slab.h"
#include <set>
#include <iostream>
int main(){using namespace beamz::cuda::moving;size_t cases=0,checks=0,certificates=0;
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(auto dims:std::vector<std::array<int,3>>{{25,27,29},{33,47,61},{64,96,128},{136,136,136},{512,512,512},{6000,25,25}})
 for(int depth:{4,16,32})for(int macros:{2,4,8}){
  BeamzBuffer fields[6]{};for(int f=0;f<6;++f){fields[f].rank=3;fields[f].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[f].dims[a]=dims[a]+add[f][a];}
  auto p=BeamzBuildNativeMovingPlan(fields,depth,{{0,0,0},{dims[0]/2,dims[1]/2,dims[2]/2}},nullptr,nullptr,12,true);
  // Deliberately heterogeneous immutable certificates. Compact using actual
  // native helpers before compiling merged multi-macro prefixes.
  for(size_t t=0;t<p.tile_flags.size();++t)if(t%3)p.tile_flags[t]|=64;
  BeamzCompactNativeMovingUniformNoIO(p,true);BeamzCompactNativeMovingUniformObservers(p,true);
  const auto slab=BeamzBuildNativeMovingMacroSlab(p,macros);const int tiles=p.tile_ids.size(),ny=p.counts[1],nx=p.counts[2];
  std::vector<int>where(size_t(tiles)*macros,-1);std::set<int>physical_certificates;
  for(int rank=0;rank<int(p.bulk_counts.size());++rank)for(int policy:{1,2}){
   const int count=policy==1?p.uniform_noio_counts[rank]:p.uniform_observer_counts[rank];for(int i=0;i<count;++i)physical_certificates.insert(p.tile_ids[p.policy_offsets[rank*7+policy]+i]);
  }
  int observed_frontier=0;
  for(int level=0;level<slab.level_count;++level){
   const int begin_level=slab.policy_offsets[level*7],end_level=slab.policy_offsets[level*7+6];observed_frontier=std::max(observed_frontier,end_level-begin_level);
   for(int i=begin_level;i<end_level;++i)if(slab.slots[slab.tile_ids[i]]!=i-begin_level)return 9;
  }
  if(observed_frontier!=slab.max_frontier)return 10;
  for(int level=0;level<slab.level_count;++level)for(int policy=0;policy<6;++policy){
   const int begin=slab.policy_offsets[level*7+policy],end=slab.policy_offsets[level*7+policy+1];
   const int certified=policy==1?slab.uniform_noio_counts[level]:policy==2?slab.uniform_observer_counts[level]:0;
   for(int i=begin;i<end;++i){const int encoded=slab.tile_ids[i],m=encoded/tiles,t=encoded%tiles;
    if(encoded<0||encoded>=tiles*macros||where[encoded]!=-1||Policy(p.tile_flags[t])!=policy)return 3;
    if(level!=t/(ny*nx)+(t/nx)%ny+t%nx+m*slab.macro_level_stride)return 4;
    if(policy==1||policy==2){if(physical_certificates.count(t)!=size_t(i<begin+certified))return 5;++certificates;}
    where[encoded]=level;
   }
  }
  for(int m=0;m<macros;++m)for(int t=0;t<tiles;++t){const int q[3]={t/(ny*nx),(t/nx)%ny,t%nx},level=where[m*tiles+t];if(level<0)return 6;
   for(int bits=1;bits<8;++bits){const int d[3]={bits&1,(bits>>1)&1,(bits>>2)&1};if(q[0]<d[0]||q[1]<d[1]||q[2]<d[2])continue;
    const int parent=t-d[0]*ny*nx-d[1]*nx-d[2];if(where[m*tiles+parent]>=level)return 7;++checks;
   }
   if(m){int parent[3];for(int a=0;a<3;++a)parent[a]=std::min(q[a]+(depth+p.owned[a]-1)/p.owned[a],p.counts[a]-1);
    if(where[(m-1)*tiles+(parent[0]*ny+parent[1])*nx+parent[2]]>=level)return 8;++checks;
   }
  }
  ++cases;
 }
 std::cout<<"{\\"cases\\":"<<cases<<",\\"dependency_checks\\":"<<checks<<",\\"certificate_checks\\":"<<certificates<<",\\"passed\\":true}\\n";
}
'''
    rows=[]
    for name in ('actual','missing-strict-level','wrong-macro-encoding'):
        folder=directory/name;folder.mkdir();header=planner
        if name=='missing-strict-level':header=header.replace('int stride=1;','int stride=0;')
        if name=='wrong-macro-encoding':header=header.replace('size_t(m)*tiles+tile','size_t(m)*(tiles-1)+tile')
        (folder/'native_macro_slab.h').write_text(header);cc=folder/'proof.cc';binary=folder/'proof';cc.write_text(code)
        subprocess.run(['g++','-std=c++17','-O2','-fsanitize=address,undefined','-I'+str(folder),'-I'+str(src),str(cc),str(src/'native_moving.cc'),'-o',str(binary)],capture_output=True,text=True,check=True)
        r=subprocess.run([str(binary)],capture_output=True,text=True)
        if (r.returncode!=0 if name=='actual' else r.returncode==0):raise RuntimeError(name+r.stdout+r.stderr)
        rows.append(dict(name=name,returncode=r.returncode,stdout=r.stdout))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],planner_sha256=proof['planner_sha256'],rows=rows,
                unchanged_helpers=True,canonical_headers_byte_identical=True,exact_Finish_inverse=True,
                scope='Actual merged affine task IDs, all7neighbor and previous-macro completion levels, native certificate prefixes and negative controls; ASan/UBSan. Exact integer-only source/raw time binding; whole-field version legality separately proved. GPU semantics and performance separate.')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate,a.parent)))
