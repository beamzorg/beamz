"""Copy-only immutable material certification for moving policy-zero CTAs."""

import argparse
import hashlib
import json
from pathlib import Path


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one uniform-E anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source):
    src = source / "src"
    p = src / "native_moving.h"
    text = p.read_text()
    text = once(text, "inline constexpr int Policy(uint8_t flags){", "inline constexpr int Policy(uint8_t flags){\n  flags&=63;")
    text = once(text, "float h_decay[3]{},h_source[3]{};", "float common_e[3]{};\n  float h_decay[3]{},h_source[3]{};")
    p.write_text(text)

    p = src / "native_extension.cc"
    text = p.read_text()
    text = once(text, "(i>=6&&i<12)||", "(i>=6&&i<12)||(i>=37&&i<43)||")
    anchor = "    BeamzNativeMovingSparseObservation sparse;"
    insert = """    // Coefficients and codes were privately captured with the initial GPU upload.
    // Certification never reads caller-owned bytes after construction.
    const void*table_data[3]{},*code_data[3]{};
    for(int c=0;c<3;++c){table_data[c]=host_[37+c].data;code_data[c]=host_[40+c].data;}
    const auto certificate=BeamzCertifyNativeMovingUniformE(base_inputs_.data(),base_inputs_.data()+37,base_inputs_.data()+40,table_data,code_data,moving_plan_);
    size_t certified_bulk=0;
    if(certificate.supported){
      std::copy(certificate.values.begin(),certificate.values.end(),moving_workspace_.common_e);
      for(size_t tile=0;tile<moving_plan_.tile_flags.size();++tile)
        if(moving_plan_.tile_flags[tile]==0&&certificate.admitted[tile]){moving_plan_.tile_flags[tile]|=64;++certified_bulk;}
    }
    moving_uniform_bulk_=certified_bulk;moving_uniform_prefix_bytes_=certificate.peak_scratch_bytes;
"""
    text = once(text, anchor, insert + anchor)
    text = once(text, "append_bytes(moving_workspace_.h_decay,sizeof(moving_workspace_.h_decay));", "append_bytes(moving_workspace_.common_e,sizeof(moving_workspace_.common_e));append_bytes(moving_workspace_.h_decay,sizeof(moving_workspace_.h_decay));")
    text = once(text, "  bool moving_ready_=false;", "  size_t moving_uniform_bulk_=0,moving_uniform_prefix_bytes_=0;\n  bool moving_ready_=false;")
    text = once(text, '    result["moving_depth"]=moving_workspace_.depth;', '    result["moving_uniform_bulk_tiles"]=moving_uniform_bulk_;result["moving_uniform_certificate_prefix_bytes"]=moving_uniform_prefix_bytes_;\n    result["moving_depth"]=moving_workspace_.depth;')
    p.write_text(text)

    p = src / "asymmetric_native.cuh"
    text = p.read_text()
    text = once(text, "int nz,ny,nx,frontier,macro_step;float h_source;", "int nz,ny,nx,frontier,macro_step;float h_source;float common_e[3];")
    text = once(text, "int Policy=(Fast?0:5)> struct Environment", "int Policy=(Fast?0:5),bool UniformE=false> struct Environment")
    text = once(text, "  const int c=index-3;const auto&v=", "  if constexpr(UniformE)return p.common_e[index-3];\n  const int c=index-3;const auto&v=")
    old = """template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>
__global__ __launch_bounds__((Fast?1024:512),1) void Asymmetric(
 const __grid_constant__ Parameters p,const int32_t*tiles){"""
    new = """template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE>
__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){"""
    text = once(text, old, new)
    text = once(text, "Environment<Layout,Fast,IsotropicH,Policy>env{p,flags};", "Environment<Layout,Fast,IsotropicH,Policy,UniformE>env{p,flags};")
    anchor = "inline bool MakeSource("
    wrapper = """template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>
__global__ __launch_bounds__((Fast?1024:512),1) void Asymmetric(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 if constexpr(Fast){
  // CTA-uniform immutable flag; barriers remain converged in either complete body.
  if(p.flags[tiles[blockIdx.x]]&64)UniformBody<Depth,Layout,Fast,IsotropicH,Policy,true>(p,tiles);
  else UniformBody<Depth,Layout,Fast,IsotropicH,Policy,false>(p,tiles);
 }else UniformBody<Depth,Layout,Fast,IsotropicH,Policy,false>(p,tiles);
}

"""
    text = once(text, anchor, wrapper + anchor)
    text = once(text, "out->macro_step=macro;out->h_source=w.h_source[0];", "out->macro_step=macro;out->h_source=w.h_source[0];std::copy(w.common_e,w.common_e+3,out->common_e);")
    p.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.source.resolve() == root / "cuda":
        parser.error("Only a private copied source may be patched")
    patch(args.source)
    print(json.dumps({"source": str(args.source), "consumer_sha256": hashlib.sha256((args.source / "src/asymmetric_native.cuh").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
