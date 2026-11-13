"""Compile the actual compact adapter, including lazy guarded address creation."""

import json
import shutil
import subprocess

import pytest

from scripts.patch_cuda_native_compact_x_epsi import POINT


def test_actual_adapter_has_no_address_before_active_guard_and_preserves_float_words(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ compiler required")
    cpp = r'''#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
struct Base {uint32_t*global;int*loads,*stores,*packed_calls;
 float LoadLocal(int s,int,int,int)const{++*loads;float v;std::memcpy(&v,global+s,4);return v;}
 void StoreLocal(int s,int,int,int,float v)const{++*stores;std::memcpy(global+s,&v,4);}
 int Packed(int s,int,int,int x)const{++*packed_calls;if((s!=9&&s!=10)||x<0||x>64||(x>=12&&x<53))std::abort();return x<12?x:23+x-64;}
};
''' + POINT.replace("__device__", "").replace("__forceinline__", "") + r'''
uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
float value(uint32_t v){float f;std::memcpy(&f,&v,4);return f;}
int main(){int errors=0,checks=0,constructs=0;
 const uint32_t patterns[]={0,0x80000000u,1,0x80000001u,0x3f800001u,0x7f800000u,0xff800000u,0x7fc12345u};
 const int xs[]={-1,0,1,11,12,52,53,64,65};
 for(unsigned p:patterns)for(int slot=0;slot<12;++slot)for(int x:xs)for(int first=0;first<2;++first)for(int flush=0;flush<2;++flush)for(int enabled=0;enabled<2;++enabled){
  uint32_t g[12];for(int i=0;i<12;++i)g[i]=0x3f800000u+i;int loads=0,stores=0,packed=0;
  Base base{g,&loads,&stores,&packed};float guarded[3074];for(float&v:guarded)v=value(0x40000001u);
  volatile float*bank=guarded+1;const int zs=7,ys=15;
  SharedEPsiPoint<Base> env(base,bank,zs,ys,first,flush,enabled);
  ++constructs;errors+=packed!=0; // Invalid/interior constructors must be inert.
  const bool active=(slot==9||slot==10)&&x>=0&&x<=64&&(x<12||x>=53);
  if(enabled&&slot>=6&&!active)continue; // CanonicalLocalActive forbids allcalls.
  const bool cached=enabled&&slot>=6;
  int index=-1;if(cached){const int color=slot==10;const int k=x<12?x:x-53;index=((color*8+zs)*16+ys)*12+k;}
  const uint32_t expected=cached&&!first?0x40000001u:g[slot];
  errors+=bits(env.LoadLocal(slot,0,0,x))!=expected;
  errors+=loads!=int(!cached||first);errors+=packed!=int(cached&&!first);
  env.StoreLocal(slot,0,0,x,value(p));
  errors+=stores!=int(!cached||flush);errors+=packed!=int(cached)*(first?1:2);
  errors+=(!cached||flush)&&g[slot]!=p;
  for(int i=-1;i<=3072;++i)errors+=bits(bank[i])!=((cached&&i==index)?p:0x40000001u);
  ++checks;
 }
 printf("{\"constructs\":%d,\"access_cases\":%d,\"errors\":%d}\n",constructs,checks,errors);return errors!=0;
}
'''
    source, binary = tmp_path / "adapter.cc", tmp_path / "adapter"
    source.write_text(cpp)
    subprocess.run(["c++", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    run = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
    result = json.loads(run.stdout)
    assert result["constructs"] == 6912 and result["access_cases"] > 4000
    assert result["errors"] == 0
