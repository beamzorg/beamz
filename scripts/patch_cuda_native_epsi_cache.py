"""Copied resource candidate: E-only owned CPML state on NoIO UniformE tiles."""

import argparse
import hashlib
import json
from pathlib import Path

PARENT = "232aa3005b91c94504b4d915c22ffaa2bc19d1e3e0216983c4a3de4d8afd9fcb"
POINT = r"""
// Only a proved single-axis E recurrence uses these two guard-exclusive colors.
template<bool Enabled,int Slots> struct EPsiStorage {};
template<int Slots> struct EPsiStorage<true,Slots> {float a[Slots],b[Slots];};
template<class Base> struct PrivateEPsiPoint : Base {
 float &a,&b;bool first,flush;
 __device__ __forceinline__ PrivateEPsiPoint(Base&base,float&pa,float&pb,bool fresh,bool publish)
 :Base(base),a(pa),b(pb),first(fresh),flush(publish){}
 __device__ __forceinline__ float& State(int slot)const{const int term=slot%6;return term==2||term==4||term==5?b:a;}
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if(slot<6)return Base::LoadLocal(slot,z,y,x);
  return first?Base::LoadLocal(slot,z,y,x):State(slot);
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  if(slot<6){Base::StoreLocal(slot,z,y,x,value);return;}
  State(slot)=value;if(flush)Base::StoreLocal(slot,z,y,x,value);
 }
};
"""


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one E-psi anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source, validation_counter=False):
    src = source / "src"
    p = src / "asymmetric_native.cuh"
    text = p.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != PARENT:
        raise ValueError("Exact NoIO UniformE parent required")
    point_source = POINT
    if validation_counter:
        point_source = once(
            point_source,
            "return first?Base::LoadLocal(slot,z,y,x):State(slot);",
            "if(first)return Base::LoadLocal(slot,z,y,x);\n  atomicOr(this->p.bad,8); // Validation-only actual non-first E cache load.\n  return State(slot);",
        )
    text = once(
        text,
        "template<int Depth,bool Safe,int T>\n__device__",
        point_source + "\ntemplate<int Depth,bool Safe,int T>\n__device__",
    )
    text = once(
        text,
        "int Policy,bool UniformE>\n__device__ __forceinline__ void UniformBody",
        "int Policy,bool UniformE,bool CacheE=false>\n__device__ __forceinline__ void UniformBody",
    )
    text = once(
        text,
        " float h[3][Slots],e[3][Slots];",
        ' static_assert(!CacheE||(!Fast&&Policy==1&&IsotropicH&&UniformE&&Layout==1),"Epsi class requires byte/NoIO/isotropic/UniformE");\n EPsiStorage<CacheE,Slots> epsi;\n float h[3][Slots],e[3][Slots];',
    )
    marker = """    if(zs!=ehz&&sy!=ehy&&sx!=ehx){
     const auto read="""
    point = """    if(zs!=ehz&&sy!=ehy&&sx!=ehx){
     auto point_env=[&] __device__(){
      if constexpr(CacheE){const int dz=(zs-ehz)&7,dy=(sy-ehy)&15,dx=(sx-ehx)&31;
       return PrivateEPsiPoint<decltype(env)>(env,epsi.a[j],epsi.b[j],n==window[0]||dz==1||dy==1||dx==1,n==Depth||dz==BZ||dy==BY||dx==BX);
      }else return env;
     }();
     const auto read="""
    text = once(text, marker, point)
    for component in range(3):
        text = once(
            text,
            f"AdvanceNativeCell<1,{component},false>(env,read,n,z,y,x)",
            f"AdvanceNativeCell<1,{component},false>(point_env,read,n,z,y,x)",
        )
    old = "  if(p.flags[tiles[blockIdx.x]]&64)UniformBody<Depth,Layout,Fast,IsotropicH,Policy,true>(p,tiles);"
    new = """  if(p.flags[tiles[blockIdx.x]]&64){
   if constexpr(!Fast&&Policy==1&&IsotropicH&&Layout==1){
    if(p.flags[tiles[blockIdx.x]]&128)UniformBody<Depth,Layout,Fast,IsotropicH,Policy,true,true>(p,tiles);
    else UniformBody<Depth,Layout,Fast,IsotropicH,Policy,true,false>(p,tiles);
   }else UniformBody<Depth,Layout,Fast,IsotropicH,Policy,true,false>(p,tiles);
  }"""
    text = once(text, old, new)
    p.write_text(text)

    p = src / "native_extension.cc"
    text = p.read_text()
    anchor = "    moving_uniform_bulk_=certified_bulk;"
    annotate = """    size_t cached_epsi=0;
    // Existing axes[] is the canonical differentiated-axis table proved above.
    if(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH){
      for(size_t tile=0;tile<moving_plan_.tile_flags.size();++tile){const uint8_t flags=moving_plan_.tile_flags[tile];const unsigned mask=flags&14u;
        if(!(flags&64)||beamz::cuda::moving::Policy(flags)!=1||(mask!=2&&mask!=4&&mask!=8))continue;
        const int axis=mask==2?0:mask==4?1:2;
        const int q[3]={int(tile/(moving_plan_.counts[1]*moving_plan_.counts[2])),int((tile/moving_plan_.counts[2])%moving_plan_.counts[1]),int(tile%moving_plan_.counts[2])};
        bool active=false;
        for(int c=0;c<3;++c){if(axes[2*c]!=axis&&axes[2*c+1]!=axis)continue;const auto&f=base_inputs_[3+c];
          for(int side=0;side<2;++side){int64_t first=1,last=depth;
            for(int a=0;a<3;++a){const int64_t origin=int64_t(q[a])*moving_plan_.owned[a];int64_t lo=0,hi=f.dims[a];
              if(a==axis){if(side)lo=f.dims[a]-12;else hi=12;}
              first=std::max<int64_t>(first,origin-hi+1);last=std::min<int64_t>(last,origin+moving_plan_.owned[a]-lo-1);
            }
            active|=first<=last;
          }
        }
        if(active){moving_plan_.tile_flags[tile]|=128;++cached_epsi;
          if(moving_private_epsi_witness_[0]<0){
            for(int c=0;c<3&&moving_private_epsi_witness_[0]<0;++c){if(axes[2*c]!=axis&&axes[2*c+1]!=axis)continue;const auto&f=base_inputs_[3+c];
              for(int side=0;side<2&&moving_private_epsi_witness_[0]<0;++side){int64_t first=1,last=depth-1;int64_t lower[3]{},upper[3]{},origin[3]{};
                for(int a=0;a<3;++a){origin[a]=int64_t(q[a])*moving_plan_.owned[a];upper[a]=f.dims[a];if(a==axis){if(side)lower[a]=f.dims[a]-12;else upper[a]=12;}
                  first=std::max<int64_t>(first,origin[a]-upper[a]+1);last=std::min<int64_t>(last,origin[a]+moving_plan_.owned[a]-lower[a]-2);
                }
                if(first<=last){const int term=axes[2*c]==axis?2*c:2*c+1;
                  moving_private_epsi_witness_={int(tile),int(first+1),6+term,int(std::max(lower[0],origin[0]-first)),int(std::max(lower[1],origin[1]-first)),int(std::max(lower[2],origin[2]-first)),axis,int(first)};
                }
              }
            }
          }
        }
      }
    }
    moving_private_epsi_=cached_epsi;
"""
    text = once(text, anchor, annotate + anchor)
    text = once(
        text,
        "size_t moving_uniform_bulk_=0,moving_uniform_noio_=0,",
        "size_t moving_private_epsi_=0;\n  std::array<int,8> moving_private_epsi_witness_{{-1,-1,-1,-1,-1,-1,-1,-1}};\n  size_t moving_uniform_bulk_=0,moving_uniform_noio_=0,",
    )
    text = once(
        text,
        'result["moving_uniform_noio_tiles"]=',
        'result["moving_private_epsi_tiles"]=moving_private_epsi_;result["moving_private_epsi_scalars_per_thread"]=16;result["moving_private_epsi_witness"]=nb::make_tuple(moving_private_epsi_witness_[0],moving_private_epsi_witness_[1],moving_private_epsi_witness_[2],moving_private_epsi_witness_[3],moving_private_epsi_witness_[4],moving_private_epsi_witness_[5],moving_private_epsi_witness_[6],moving_private_epsi_witness_[7]);result["moving_uniform_noio_tiles"]=',
    )
    text = once(
        text,
        "NB_MODULE(_cuda_native, module) {",
        'NB_MODULE(_cuda_native, module) {\n  module.attr("moving_private_epsi")=true;\n  module.attr("moving_private_epsi_families")="E-only";',
    )
    if validation_counter:
        text = once(
            text,
            "  size_t moving_private_epsi_=0;",
            "  bool moving_private_epsi_debug_hit_=false;int32_t moving_private_epsi_debug_raw_=0;\n  size_t moving_private_epsi_=0;",
        )
        text = once(
            text,
            'Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");',
            'moving_private_epsi_debug_hit_=false;moving_private_epsi_debug_raw_=0;\n      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");',
        )
        text = once(
            text,
            'if(bad)throw std::runtime_error("native moving producer/staging diagnostic failed: "+std::to_string(bad));',
            'moving_private_epsi_debug_raw_=bad;moving_private_epsi_debug_hit_=(bad&8)!=0;\n      if(bad&~8)throw std::runtime_error("native moving producer/staging diagnostic failed: "+std::to_string(bad));',
        )
        text = once(
            text,
            'result["moving_private_epsi_tiles"]=',
            'result["moving_private_epsi_debug_instrumented"]=true;result["moving_private_epsi_debug_hit"]=moving_private_epsi_debug_hit_;result["moving_private_epsi_debug_raw"]=moving_private_epsi_debug_raw_;result["moving_private_epsi_debug_real_errors"]=moving_private_epsi_debug_raw_&~8;result["moving_private_epsi_tiles"]=',
        )
        text = once(
            text,
            'module.attr("moving_private_epsi")=true;',
            'module.attr("moving_private_epsi")=true;\n  module.attr("moving_private_epsi_validation_counter")=true;',
        )
    p.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation-counter", action="store_true")
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("Private copied source required")
    patch(args.source, args.validation_counter)
    print(
        json.dumps(
            {
                "consumer_sha256": hashlib.sha256(
                    (args.source / "src/asymmetric_native.cuh").read_bytes()
                ).hexdigest(),
                "private_psi": "E-only two arrays[Slots8]=16 scalars; H psi unchanged global",
                "admission": "Policy1&&IsotropicH&&UniformE&&Layout1&&single-axis&&active E recurrence",
                "resource_gate": "zero stack/spill required before GPU",
                "validation_counter": args.validation_counter,
            }
        )
    )


if __name__ == "__main__":
    main()
