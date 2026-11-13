"""Lower captured rectangular support/pitch facts into the certified NoIO body.

This generalizes the existing cube partial evaluation. Only exact matching
canonical shapes enter the generated body; coefficients and CPML profiles
remain independent runtime values. Field handoffs retain their actual Layout.
"""
from pathlib import Path
from build_cuda_block_uniform import once


def patch(source: Path, shape):
    if len(shape) != 3 or any(n <= 24 for n in shape):
        raise ValueError("Three positive CPML12 interiors required")
    path = source / "src/asymmetric_native.cuh"
    text = path.read_text()
    begin = text.index("// Only geometry is specialized;")
    end = text.index("template<int Depth,int N>\n__global__ __launch_bounds__(1024,1) void AsymmetricNoIOFixedCube", begin)
    region = text[begin:end]
    region = once(region,
        "template<int N,int ActiveAxes=-1>struct FixedCubeCachedEnvironment:Environment<1,",
        "template<int NZ,int NY,int NX,int Layout,int ActiveAxes=-1>struct FixedRectCachedEnvironment:Environment<Layout,")
    region = region.replace("using Base=Environment<1,", "using Base=Environment<Layout,")
    region = once(region, 'static_assert(N==512||N==564,"Common-shape AOT512/564 only");',
                  'static_assert(NZ>24&&NY>24&&NX>24,"Canonical CPML12 interiors");')
    region = region.replace("FixedCubeCachedEnvironment(", "FixedRectCachedEnvironment(")
    region = once(region, "constexpr int nz=N,ny=N,nx=N;", "constexpr int nz=NZ,ny=NY,nx=NX;")
    if region.count("const int size=N+family;") != 2:
        raise ValueError("Expected distinct packed-index and CPML-activity facts")
    region = region.replace("const int size=N+family;", "const int size=(axis==0?NZ:axis==1?NY:NX)+family;")
    region = once(region, "const int ny=axis==1?24:N+", "const int ny=axis==1?24:NY+")
    region = once(region, "const int nx=axis==2?24:N+", "const int nx=axis==2?24:NX+")
    region = once(region, "coordinate==N)", "coordinate==(axis==0?NZ:axis==1?NY:NX))")
    region = once(region, "template<int Depth,int N,int ActiveAxes=-1>\n__device__ __forceinline__ void FixedCubeCachedBody",
                  "template<int Depth,int NZ,int NY,int NX,int Layout,int ActiveAxes=-1>\n__device__ __forceinline__ void FixedRectCachedBody")
    region = once(region, "constexpr int Layout=1,Policy=1;", "constexpr int Policy=1;")
    region = once(region, "FixedCubeCachedEnvironment<N,ActiveAxes>env", "FixedRectCachedEnvironment<NZ,NY,NX,Layout,ActiveAxes>env")
    nz, ny, nx = shape
    kernels = f"""template<int Depth,int Layout>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIORectCached(
 const __grid_constant__ Parameters p,const int32_t*tiles){{
 switch(p.flags[PhysicalTileID(p,tiles[blockIdx.x])]&14){{
  case 2:FixedRectCachedBody<Depth,{nz},{ny},{nx},Layout,1>(p,tiles);return;
  case 4:FixedRectCachedBody<Depth,{nz},{ny},{nx},Layout,2>(p,tiles);return;
  case 8:FixedRectCachedBody<Depth,{nz},{ny},{nx},Layout,4>(p,tiles);return;
  default:FixedRectCachedBody<Depth,{nz},{ny},{nx},Layout,-1>(p,tiles);return;
 }}
}}

"""
    # Keep the existing cube entries and generic fallback byte-for-byte.
    text = text[:end] + region + kernels + text[end:]
    anchor = "  // Common-shape AOT only; all original capture/admission certificates still apply."
    dispatch = f"""  // Captured physical supports; Prepare has proved canonical FP32 pitches.
  if(p.physical_n[0]=={nz}&&p.physical_n[1]=={ny}&&p.physical_n[2]=={nx}){{
   const auto kernel=AsymmetricNoIORectCached<Depth,Layout>;
   status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ProfileCachedSharedBytes);
   if(status!=cudaSuccess)return status;
   AsymmetricNoIORectCached<Depth,Layout><<<uniform_count,1024,ProfileCachedSharedBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }}
"""
    text = once(text, anchor, dispatch + anchor)
    path.write_text(text)
    path = source / "src/native_extension.cc"
    path.write_text(once(path.read_text(), '  module.attr("moving_block_uniform_e")=true;',
        f'  module.attr("moving_rect_shape_zyx")=nb::make_tuple({nz},{ny},{nx});\n  module.attr("moving_block_uniform_e")=true;'))
