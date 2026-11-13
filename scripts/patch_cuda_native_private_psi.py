"""Copied-header RESOURCE-ONLY private CPML state specialization.

No launch dispatch is changed. Explicit template instantiations expose fixed
axis/NoIO kernels to ptxas, but the native runner continues its original path.
An eventual dispatcher must admit NoIO/single-axis facts before selecting them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_predicated_state_storage import allocate

POINT_ENV = r"""
// Four scalar references, never a pointer/reference to an entire private array.
// Valid/LocalActive guards in the traced functor precede each state access.
template<int Axis,class Base> struct PrivatePsiPoint : Base {
 float &ha,&hb,&ea,&eb;bool first,flush;
 __device__ __forceinline__ PrivatePsiPoint(Base&base,float&h0,float&h1,float&e0,float&e1,bool fresh,bool publish)
  :Base(base),ha(h0),hb(h1),ea(e0),eb(e1),first(fresh),flush(publish){}
 __device__ __forceinline__ float& State(int slot)const{
  constexpr int first_term=Axis==0?1:Axis==1?0:3;
  return slot<6?(slot%6==first_term?ha:hb):(slot%6==first_term?ea:eb);
 }
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if constexpr(Axis<0)return Base::LoadLocal(slot,z,y,x);
  if(beamz::cuda::yee::CpmlAxis(slot%6)!=Axis)return Base::LoadLocal(slot,z,y,x);
  return first?Base::LoadLocal(slot,z,y,x):State(slot);
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  if constexpr(Axis<0){Base::StoreLocal(slot,z,y,x,value);return;}
  if(beamz::cuda::yee::CpmlAxis(slot%6)!=Axis){Base::StoreLocal(slot,z,y,x,value);return;}
  State(slot)=value;
  // Unique GPR state has no other reader in this CTA. Later owner kernels wait
  // for this complete rank, so high/final publication needs no shared barrier.
  if(flush)Base::StoreLocal(slot,z,y,x,value);
 }
};
"""


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Private psi patch expected one anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source, *, strong_axis=False, cta_switch=False, colored=False):
    if colored and not cta_switch:
        raise ValueError("Colored state storage requires the immutable single-axis CTA guard")
    kernel_template = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__"
    launch_template = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\ninline cudaError_t Launch("
    source = once(source, kernel_template,
                  "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),int CacheAxis=-1>\n__global__")
    source = once(source, launch_template,
                  "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),int CacheAxis=-1>\ninline cudaError_t Launch(")
    source = once(source, "const auto kernel=Asymmetric<Depth,Layout,Fast,IsotropicH,Policy>;",
                  "const auto kernel=Asymmetric<Depth,Layout,Fast,IsotropicH,Policy,CacheAxis>;")
    source = once(source, "Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<",
                  "Asymmetric<Depth,Layout,Fast,IsotropicH,Policy,CacheAxis><<<")
    point_environment = POINT_ENV
    if colored:
        allocation = allocate()
        expected = [0, 0, 1, 0, 1, 1, 2, 2, 3, 2, 3, 3]
        if list(allocation["colors"].values()) != expected:
            raise ValueError("Current backend does not match the proved state coloring")
        point_environment = point_environment.replace("if constexpr(Axis<0)", "if constexpr(Axis==-1)")
        point_environment = point_environment.replace("if(beamz::cuda::yee::CpmlAxis(slot%6)!=Axis)",
                                                       "if(Axis>=0&&beamz::cuda::yee::CpmlAxis(slot%6)!=Axis)")
        point_environment = once(point_environment, "  constexpr int first_term=Axis==0?1:Axis==1?0:3;",
                                 "  if constexpr(Axis==-2){const int term=slot%6;const bool second=term==2||term==4||term==5;\n"
                                 "   return slot<6?(second?hb:ha):(second?eb:ea);}\n"
                                 "  constexpr int first_term=Axis==0?1:Axis==1?0:3;")
    if strong_axis:
        point_environment = once(point_environment, " float &ha,&hb,&ea,&eb;bool first,flush;",
                                 " float &ha,&hb,&ea,&eb;bool first,flush;\n"
                                 " __device__ __forceinline__ bool LocalActive(int slot,int z,int y,int x)const{\n"
                                 "  if constexpr(Axis>=0){if(beamz::cuda::yee::CpmlAxis(slot%6)!=Axis)return false;}\n"
                                 "  return Base::LocalActive(slot,z,y,x);\n }")
    source = once(source, "template<int Depth,bool Fast>\n__device__ __forceinline__ FieldPacket ImportPacket(",
                  point_environment + "\ntemplate<int Depth,bool Fast>\n__device__ __forceinline__ FieldPacket ImportPacket(")
    condition = "CacheAxis==-1||(!Fast&&Policy==1&&(CacheAxis==-2||(CacheAxis>=0&&CacheAxis<3)))" if colored else "CacheAxis<0||(!Fast&&Policy==1&&CacheAxis<3)"
    source = once(source, " const int sx=threadIdx.x&31,sy=threadIdx.x/32;float hy[CZ],ey[CZ];",
                  f" static_assert({condition},\"private psi resource class requires NoIO\");\n"
                  " float psi_ha[CZ],psi_hb[CZ],psi_ea[CZ],psi_eb[CZ];\n"
                  " const int sx=threadIdx.x&31,sy=threadIdx.x/32;float hy[CZ],ey[CZ];")
    point = "if(j!=head&&sy!=yh&&sx!=xh){const int z=oz+low+WrapZ(j-head);"
    if source.count(point) != 2:
        raise ValueError("Expected one unrolled H and E point loop")
    source = source.replace(point, point + "\n"
                            "     const int dz=WrapZ(j-head),dy=(sy-yh)&15,dx=(sx-xh)&31;\n"
                            "     PrivatePsiPoint<CacheAxis,decltype(env)> point_env(env,psi_ha[j],psi_hb[j],psi_ea[j],psi_eb[j],\n"
                            "      n==1||dz==1||dy==1||dx==1,n==Depth||dz==BZ||dy==BY||dx==BX);")
    calls = "(env,read,n,z,y,x)"
    if source.count(calls) != 6:
        raise ValueError("Expected six fixed component functor calls")
    source = source.replace(calls, "(point_env,read,n,z,y,x)")
    # Force resource emission without adding a runtime/native binding. The
    # original LaunchRanks still selects CacheAxis=-1 in every branch.
    marker = "template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks("
    instantiations = "// RESOURCE-ONLY: no native dispatch selects these axis specializations.\n"
    for depth, axis in ((4, 0), (16, 0), (16, 1), (16, 2)):
        instantiations += f"template cudaError_t Launch<{depth},1,false,true,1,{axis}>(cudaStream_t,const Parameters&,const int32_t*,int);\n"
    if colored:
        for depth in (4, 16):
            instantiations += f"template cudaError_t Launch<{depth},1,false,true,1,-2>(cudaStream_t,const Parameters&,const int32_t*,int);\n"
    source = once(source, marker, instantiations + "\n" + marker)
    if cta_switch:
        old = """template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),int CacheAxis=-1>
__global__ __launch_bounds__(Threads,(Fast?2:1)) void Asymmetric(
 const __grid_constant__ Parameters p,const int32_t*tiles){"""
        source = once(source, old,
                      "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),int CacheAxis=-1>\n"
                      "__device__ __forceinline__ void AsymmetricBody(const Parameters&p,const int32_t*tiles){")
        wrapper = r"""
// blockIdx and immutable tile facts make this switch uniform for the whole CTA.
// Every body contains CTA barriers, so no lane-dependent dispatch is admitted.
template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),int CacheAxis=-1>
__global__ __launch_bounds__(Threads,(Fast?2:1)) void Asymmetric(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 if constexpr(CacheAxis>=0){AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,CacheAxis>(p,tiles);}
 else if constexpr(Policy==1){
  const unsigned axes=__ldg(p.flags+tiles[blockIdx.x])&14u;
  switch(axes){
   case 2:AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,0>(p,tiles);break;
   case 4:AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,1>(p,tiles);break;
   case 8:AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,2>(p,tiles);break;
   default:AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,-1>(p,tiles);break;
  }
 }else{AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,-1>(p,tiles);}
}
"""
        if colored:
            wrapper = wrapper.replace("if constexpr(CacheAxis>=0)", "if constexpr(CacheAxis!=-1)")
            begin = wrapper.index("  switch(axes){")
            end = wrapper.index("\n }else", begin)
            wrapper = wrapper[:begin] + "  if(axes==2||axes==4||axes==8)AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,-2>(p,tiles);\n" + \
                "  else AsymmetricBody<Depth,Layout,Fast,IsotropicH,Policy,-1>(p,tiles);" + wrapper[end:]
        source = once(source, "inline bool MakeSource(", wrapper + "\ninline bool MakeSource(")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--strong-axis", action="store_true", help="Require a proved single-axis class; erase other-axis active branches")
    parser.add_argument("--cta-switch", action="store_true", help="Emit uniform NoIO CTA switch; all GPU parity/sanitizer gates still required")
    parser.add_argument("--colored", action="store_true", help="Use proved guard-exclusive four-color storage in one single-axis body")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input is frozen; output must be a separate file")
    result = patch(args.input.read_text(), strong_axis=args.strong_axis, cta_switch=args.cta_switch, colored=args.colored)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    manifest = {"input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "output": str(args.output), "output_sha256": hashlib.sha256(result.encode()).hexdigest(),
                "GPU_candidate": False, "runtime_private_dispatch": args.cta_switch,
                "strong_single_axis_fact": args.strong_axis,
                "guard_coloring": allocate() if args.colored else None,
                "reason": "Resource screening only; runtime CTA dispatch, if emitted, still needs all numerical/sanitizer gates",
                "private_state": "four fixed arrays[CZ], 24 scalars/thread; per-point scalar references only",
                "semantics": "selected axis first-owned global load/high-or-final store; other axes unchanged global fallback"}
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
