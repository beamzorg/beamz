"""Actual budgeted point accessors use canonical dense slabs and exact copies."""

import shutil
import subprocess

import pytest

from scripts.patch_cuda_native_budgeted_epsi import point_helpers


def test_actual_emitted_global_and_private_accessors(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++20 compiler required")
    helper = point_helpers()
    source = r'''
#include <array>
#include <vector>
#include <cassert>
#include <cstdint>
#include <bit>
#include <cstdio>
#define __device__
#define __forceinline__ inline
namespace beamz::cuda::yee{constexpr int CpmlAxis(int t){constexpr int a[6]={1,0,0,2,2,1};return a[t];}}
struct Psi{const float*input;float*output;};
struct Parameters{struct Phase{Psi psi[6];}phase[2];int physical_n[3]={32,35,37};};
struct Base{const Parameters&p;
 int Packed(int slot,int z,int y,int x)const{int point[]={z,y,x},axis=beamz::cuda::yee::CpmlAxis(slot%6);int c=point[axis],n=p.physical_n[axis]+slot/6;return c<12?c:24+c-n;}
};
'''+helper+r'''
template<int Mode>int check(){
 Parameters p;Base base{p};std::vector<float>memory[12];int shapes[12][3];
 const int delta[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};int checks=0;
 for(int s=0;s<12;++s){int f=3*(s/6)+(s%6)/2,axis=beamz::cuda::yee::CpmlAxis(s%6),count=1;
  for(int a=0;a<3;++a){shapes[s][a]=a==axis?24:p.physical_n[a]+delta[f][a];count*=shapes[s][a];}
  memory[s].resize(count);for(int i=0;i<count;++i)memory[s][i]=float(100000*s+i);
  p.phase[s/6].psi[s%6]={memory[s].data(),memory[s].data()};
 }
 for(int s=0;s<12;++s){int f=3*(s/6)+(s%6)/2,axis=beamz::cuda::yee::CpmlAxis(s%6),extent=p.physical_n[axis]+s/6;
  for(int coordinate: {0,11,extent-12,extent-1})for(int edge:{0,1}){
   int point[3];for(int a=0;a<3;++a)point[a]=edge?p.physical_n[a]+delta[f][a]-1:0;point[axis]=coordinate;
   int packed[3]={point[0],point[1],point[2]};packed[axis]=coordinate<12?coordinate:coordinate-extent+24;
   int offset=(packed[0]*shapes[s][1]+packed[1])*shapes[s][2]+packed[2];const float old=memory[s][offset];
   float a=-123.f,b=-456.f;auto make=[&](bool first,bool flush){if constexpr(Mode==1)return BudgetedPointA<Base>(base,a,first,flush);else return BudgetedPointBoth<Base>(base,a,b,first,flush);};
   auto first=make(true,false);assert(first.LoadLocal(s,point[0],point[1],point[2])==old);++checks;
   const bool cached=s>=6&&(Mode==2||s%6==0||s%6==1||s%6==3);
   first.StoreLocal(s,point[0],point[1],point[2],-0.f);
   if(cached)assert(memory[s][offset]==old);else assert(std::bit_cast<uint32_t>(memory[s][offset])==0x80000000u);++checks;
   auto warm=make(false,false);assert(std::bit_cast<uint32_t>(warm.LoadLocal(s,point[0],point[1],point[2]))==0x80000000u);++checks;
   auto final=make(false,true);final.StoreLocal(s,point[0],point[1],point[2],123.25f);assert(memory[s][offset]==123.25f);++checks;
  }
 }
 return checks;
}
int main(){std::printf("checks=%d errors=0\n",check<1>()+check<2>());}
'''
    cpp, binary = tmp_path / "state.cc", tmp_path / "state"
    cpp.write_text(source)
    subprocess.run([compiler, "-std=c++20", "-O2", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                    "-ffp-contract=off", str(cpp), "-o", str(binary)], check=True, capture_output=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "checks=768 errors=0" in result.stdout
