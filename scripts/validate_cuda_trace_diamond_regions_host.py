"""Check actual native diamond region/certificate arithmetic under ASan/UBSan."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_diamond_regions import physical_bounds,has_update


def validate(candidate):
    candidate=Path(candidate).resolve();header=candidate/'source/src/native_moving.h'
    text=header.read_text();start=text.index('inline int64_t BeamzDiamondFloorHalf')
    end=text.index('namespace beamz::cuda::moving',start);helpers=text[start:end]
    directory=candidate/'host-region-proof';directory.mkdir(exist_ok=False)
    proof=json.loads((candidate/'diamond-compiler-proof.json').read_text());orientation=proof.get('orientation','xy')
    rows=[];owned=tuple(c-1 for c in proof['cache'])
    for bias,depth,tile,r,shape in itertools.product((26,27),(1,4,16),
            itertools.product(range(4),range(4),range(5)),(0,1),((26,27,28),(37,40,43))):
        origin=tuple(a*b for a,b in zip(tile,owned));lo,hi=physical_bounds(origin,owned,depth,bias,r,orientation=orientation)
        hit=has_update(origin,owned,depth,bias,shape,r,orientation=orientation)
        rows.append('{'+','.join(map(str,(*origin,depth,bias,r,*shape,*lo,*hi,int(hit))))+'}')
    source='''#include <cstdint>
#include <iostream>
'''+helpers+'''
int main(){
 const int b[3]={'''+','.join(map(str,owned))+'''};
 const int cases[][16]={
'''+',\n'.join(rows)+'''};
 for(auto&row:cases){
  const int o[3]={row[0],row[1],row[2]},shape[3]={row[6],row[7],row[8]};int64_t lo[3],hi[3];
  BeamzNativeDiamondBounds(o,b,row[3],row[4],lo,hi,row[5]);
  for(int a=0;a<3;++a)if(lo[a]!=row[9+a]||hi[a]!=row[12+a])return 2;
  if(BeamzNativeDiamondHasUpdate(o,b,row[3],row[4],shape,row[5])!=bool(row[15]))return 3;
 }
 std::cout<<"{\\\"passed\\\":true,\\\"region_cases\\\":"<<sizeof(cases)/sizeof(cases[0])<<"}\\n";
}
'''
    path=directory/'regions.cc';binary=directory/'regions';path.write_text(source)
    command=['g++','-std=c++17','-O2','-fsanitize=address,undefined',str(path),'-o',str(binary)]
    subprocess.run(command,capture_output=True,text=True,check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
    proof=dict(passed=True,ASan_UBSan=True,actual_header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
               cases=len(rows),stdout=result.stdout,command=command,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               scope='Actual physical region bounds and integer nonempty certificates; lattice enumeration is independently tested in unit suite')
    (directory/'result.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args();print(json.dumps(validate(a.candidate)))
