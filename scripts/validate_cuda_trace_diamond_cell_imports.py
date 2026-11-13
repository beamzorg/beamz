"""Execute generated grouped imports against physical-cell ownership oracles."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


def validate(candidate,proof_name='host-cell-import-proof'):
    candidate=Path(candidate).resolve();proof=json.loads((candidate/'diamond-private-proof.json').read_text())
    if not proof.get('factorized_cell'):raise ValueError('Actual factorized cell candidate required')
    general=(candidate/'source/src/trace_tile_generated.cuh').read_text()
    private=(candidate/'source/src/trace_diamond_private_generated.cuh').read_text()
    consumer=(candidate/'source/src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(private.encode()).hexdigest()!=proof['source_sha256']:raise ValueError('Factorized source identity changed')
    positions=general.split('template<int Depth,int Layout,int Field>')[0]
    effects=private.split('template<int Depth,int Layout,class Env>')[0]
    surface=block(consumer,'int Surface(')[2]
    support=block(consumer,'unsigned NativeSupportMask(')[2]
    source='''#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <vector>
#define __host__
#define __device__
#define __forceinline__ inline
constexpr int CZ=16,CY=16,CX=32,PAD=CZ*CY*CX,BZ=CZ-1,BY=CY-1,BX=CX-1;
constexpr int SurfacePoints=BZ*BY*BX-(BZ-1)*(BY-1)*(BX-1);
'''+positions+'''
struct Field {float*data=nullptr;int z=0,y=0,x=0;size_t row=0,plane=0;};
struct Phase {Field own[3],output[3];};
struct Parameters {Phase phase[2];float*trails;unsigned*bad;};
const Field&FinalField(const Parameters&p,int field){return p.phase[field/3].output[field%3];}
float __ldcg(const float*p){return *p;}
float __int_as_float(int bits){return std::bit_cast<float>(bits);}
void atomicOr(unsigned*p,unsigned bits){*p|=bits;}
'''+surface+'\n'+support+'\n'+effects+'''
int floor_div(int a,int b){return a/b-((a<0)&&(a%b));}
float oracle(const Parameters&p,const size_t*bases,int field,int version,int depth,int layout,const DiamondSite&s,const int*origin){
 const auto&f=p.phase[field/3].own[field%3];
 if(s.z<0||s.y<0||s.x<0||s.z>=f.z||s.y>=f.y||s.x>=f.x)return 0.f;
 if(layout==1&&version==depth){
  if(field>=3)return 0.f;
  return float((field+1)*3000000+(s.z*f.y+s.y)*f.x+s.x);
 }
 const int r[6]={0,1,0,1,0,1},owned[3]={BZ,BY,BX};
 const int v[3]={s.z+s.y,s.z-s.y+p.phase[0].own[0].y+r[field],s.x};
 int relative[3],bits=0;
 for(int axis=0;axis<3;++axis){
  const int owner=floor_div(v[axis]+version,owned[axis]),current=origin[axis]/owned[axis],difference=current-owner;
  if(difference<0||difference>1)return -101.f;
  bits|=difference<<axis;relative[axis]=v[axis]+version-owner*owned[axis];
 }
 if(!bits)return -102.f;
 int surface=-1;
 if(relative[0]==BZ-1)surface=relative[1]*BX+relative[2];
 else if(relative[1]==BY-1)surface=BY*BX+relative[0]*BX+relative[2];
 else if(relative[2]==BX-1)surface=BY*BX+(BZ-1)*BX+relative[0]*(BY-1)+relative[1];
 if(surface<0||surface>=SurfacePoints)return -103.f;
 return float(bases[bits]+((version-1)*6+field)*SurfacePoints+surface+1);
}
size_t checks=0;
template<int Depth,int Layout,int Mask,int R>bool check(const Parameters&p,const size_t*bases,int n,int q,const int*origin,int bias){
 const auto cell=DiamondCellPosition(q,n,origin[0],origin[1],origin[2],bias);
 const auto s=DiamondCellSite<R>(cell),expected=DiamondPosition<R?1:0>(q,n,origin[0],origin[1],origin[2],bias);
 if(s.z!=expected.z||s.y!=expected.y||s.x!=expected.x||s.dz!=expected.dz||s.dp!=expected.dp||s.dm!=expected.dm||s.owned!=expected.owned)return false;
 if(s.owned)return true;
 const auto packet=DiamondImportCell<Depth,Layout,Mask>(p,bases,n,s);
 const float actual[6]={packet.v0,packet.v1,packet.v2,packet.v3,packet.v4,packet.v5};
 for(int f=0;f<6;++f){
  const float value=(Mask&(1<<f))?oracle(p,bases,f,n,Depth,Layout,s,origin):0.f;
  if(std::bit_cast<uint32_t>(value)!=std::bit_cast<uint32_t>(actual[f]))return false;
  ++checks;
 }
 return !*p.bad;
}
template<int Depth,int Layout>bool run(Parameters&p){
 std::vector<float>trails(8*Depth*6*SurfacePoints);for(size_t i=0;i<trails.size();++i)trails[i]=float(i+1);p.trails=trails.data();
 size_t bases[8];for(int i=0;i<8;++i)bases[i]=i*Depth*6*SurfacePoints;
 const int origins[][3]={{0,90,0},{30,90,31},{90,135,62},{210,135,124},{240,195,155}};
 for(const auto&origin:origins)for(int n=1;n<=Depth;++n)for(int q=0;q<DiamondPoints;++q){
  if(!check<Depth,Layout,21,0>(p,bases,n,q,origin,p.phase[0].own[0].y)||
     !check<Depth,Layout,42,1>(p,bases,n,q,origin,p.phase[0].own[0].y)||
     !check<Depth,Layout,5,0>(p,bases,n,q,origin,p.phase[0].own[0].y)||
     !check<Depth,Layout,2,1>(p,bases,n,q,origin,p.phase[0].own[0].y)||
     !check<Depth,Layout,16,0>(p,bases,n,q,origin,p.phase[0].own[0].y)||
     !check<Depth,Layout,40,1>(p,bases,n,q,origin,p.phase[0].own[0].y))return false;
 }
 return true;
}
int main(){
 const int deltas[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(int bias:{136,137}){Parameters p{};unsigned bad=0;p.bad=&bad;std::vector<float>output[3];
  for(int f=0;f<6;++f){Field value;value.z=136+deltas[f][0];value.y=bias+deltas[f][1];value.x=139+deltas[f][2];value.row=value.x;value.plane=value.row*value.y;
   if(f<3){output[f].resize(size_t(value.z)*value.plane);for(size_t i=0;i<output[f].size();++i)output[f][i]=float((f+1)*3000000+i);value.data=output[f].data();}
   p.phase[f/3].own[f%3]=p.phase[f/3].output[f%3]=value;
  }
  if(!run<4,0>(p)||!run<4,1>(p)||!run<16,0>(p)||!run<16,1>(p))return 3;
 }
 std::cout<<"{\\\"passed\\\":true,\\\"checked_import_fields\\\":"<<checks<<"}\\n";
}
'''
    if proof.get('support_safe_admission'):
        source=source.replace('template<int Depth,int Layout,int Mask,int R>bool check(',
                              'template<int Depth,int Layout,int Mask,int R,bool Safe=false>bool check(')
        source=source.replace('DiamondImportCell<Depth,Layout,Mask>(p,bases,n,s)',
                              'DiamondImportCell<Depth,Layout,Mask,Safe>(p,bases,n,s)')
        marker='  if(!check<Depth,Layout,21,0>'
        safe='''  if(origin[0]==90&&origin[1]==135&&origin[2]==62){
   if(!check<Depth,Layout,21,0,true>(p,bases,n,q,origin,p.phase[0].own[0].y)||
      !check<Depth,Layout,42,1,true>(p,bases,n,q,origin,p.phase[0].own[0].y)||
      !check<Depth,Layout,5,0,true>(p,bases,n,q,origin,p.phase[0].own[0].y)||
      !check<Depth,Layout,2,1,true>(p,bases,n,q,origin,p.phase[0].own[0].y)||
      !check<Depth,Layout,16,0,true>(p,bases,n,q,origin,p.phase[0].own[0].y)||
      !check<Depth,Layout,40,1,true>(p,bases,n,q,origin,p.phase[0].own[0].y))return false;
  }
'''
        if source.count(marker)!=1:raise ValueError('Safe-import oracle binding changed')
        source=source.replace(marker,safe+marker)
    directory=candidate/proof_name;directory.mkdir(exist_ok=False);rows=[]
    for name in ('actual','wrong-version-stride','wrong-retiming-wrap'):
        modified=source.replace('address=p.trails+base+(version-1)*6*SurfacePoints','address=p.trails+base+(version-1)*5*SurfacePoints',1) if name=='wrong-version-stride' else source.replace('cell.y+(wrap?CY/2:0)','cell.y-(wrap?CY/2:0)',1) if name=='wrong-retiming-wrap' else source
        if name!='actual' and modified==source:raise ValueError('Missing grouped-import negative witness')
        path=directory/(name+'.cc');binary=directory/name;path.write_text(modified)
        command=['g++','-std=c++20','-O2',str(path),'-o',str(binary)]
        subprocess.run(command,capture_output=True,text=True,check=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True)
        if result.returncode!=(0 if name=='actual' else 3):raise RuntimeError(name+result.stdout+result.stderr+str(result.returncode))
        rows.append(dict(name=name,returncode=result.returncode,stdout=result.stdout,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(passed=True,generated_sha256=proof['source_sha256'],rows=rows,
                scope='Actual grouped native import helper and retiming-site factorization; independent physical-owner/surface oracle, both layouts/depths, clipped supports, finalH and deadE, two negative controls; no CUDA visibility claim')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--proof-name',default='host-cell-import-proof')
    a=p.parse_args();print(json.dumps(validate(a.candidate,a.proof_name)))
