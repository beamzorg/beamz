// Deterministic output of scripts/generate_cuda_native_cell.py; research opt-in.
#pragma once
#include <cuda_bf16.h>
namespace beamz { namespace cuda { namespace trace_native {
template<int Phase,int Component,bool CheckSupport=true,class Environment,class ReadField>
__device__ __forceinline__ float AdvanceNativeCell(Environment&env,ReadField read,int version,int z,int y,int x){
static_assert(Phase==0||Phase==1,"Yee phase must be H/E");
static_assert(Component>=0&&Component<3,"Yee component must be x/y/z");
if constexpr(Phase==0&&Component==0){
if constexpr(CheckSupport){if(!env.Valid(0,z,y,x))return 0.f;}
float result=0.f;
const float derivative_0=env.Difference(0,5,1,z,y,x,(read(5,(version-1),0,0,0)),(read(5,(version-1),0,1,0)));
float corrected_0=__fmul_rn(1.0f,derivative_0);
if(env.LocalActive(0,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,0,z+(0),y+(0),x+(0)),env.LoadLocal(0,z,y,x),__fmul_rn(env.Coefficient(2,0,z+(0),y+(0),x+(0)),derivative_0));
corrected_0=__fmul_rn(1.0f,__fmaf_rn(derivative_0,env.Coefficient(4,0,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(0,z,y,x,next_psi);
}
const float derivative_1=env.Difference(0,4,0,z,y,x,(read(4,(version-1),0,0,0)),(read(4,(version-1),1,0,0)));
float corrected_1=__fmul_rn(-1.0f,derivative_1);
if(env.LocalActive(1,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,1,z+(0),y+(0),x+(0)),env.LoadLocal(1,z,y,x),__fmul_rn(env.Coefficient(2,1,z+(0),y+(0),x+(0)),derivative_1));
corrected_1=__fmul_rn(-1.0f,__fmaf_rn(derivative_1,env.Coefficient(4,1,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(1,z,y,x,next_psi);
}
const float native_Hx_0 = env.Coefficient(1,0,z+(0),y+(0),x+(0));
const float native_Hx_1 = (-native_Hx_0);
const float native_Hx_2 = corrected_0;
const float native_Hx_3 = corrected_1;
const float native_Hx_4 = __fadd_rn(native_Hx_2, native_Hx_3);
const float native_Hx_5 = env.Coefficient(0,0,z+(0),y+(0),x+(0));
const float native_Hx_6 = read(0,(version-1),0,0,0);
const float native_Hx_7 = __fmul_rn(native_Hx_5, native_Hx_6);
const float native_Hx_8 = __fmaf_rn(native_Hx_1, native_Hx_4, native_Hx_7);
result = native_Hx_8;
return env.Finish(0,version,z,y,x,result);
}
if constexpr(Phase==0&&Component==1){
if constexpr(CheckSupport){if(!env.Valid(1,z,y,x))return 0.f;}
float result=0.f;
const float derivative_0=env.Difference(0,3,0,z,y,x,(read(3,(version-1),0,0,0)),(read(3,(version-1),1,0,0)));
float corrected_0=__fmul_rn(1.0f,derivative_0);
if(env.LocalActive(2,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,2,z+(0),y+(0),x+(0)),env.LoadLocal(2,z,y,x),__fmul_rn(env.Coefficient(2,2,z+(0),y+(0),x+(0)),derivative_0));
corrected_0=__fmul_rn(1.0f,__fmaf_rn(derivative_0,env.Coefficient(4,2,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(2,z,y,x,next_psi);
}
const float derivative_1=env.Difference(0,5,2,z,y,x,(read(5,(version-1),0,0,0)),(read(5,(version-1),0,0,1)));
float corrected_1=__fmul_rn(-1.0f,derivative_1);
if(env.LocalActive(3,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,3,z+(0),y+(0),x+(0)),env.LoadLocal(3,z,y,x),__fmul_rn(env.Coefficient(2,3,z+(0),y+(0),x+(0)),derivative_1));
corrected_1=__fmul_rn(-1.0f,__fmaf_rn(derivative_1,env.Coefficient(4,3,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(3,z,y,x,next_psi);
}
const float native_Hy_0 = env.Coefficient(1,1,z+(0),y+(0),x+(0));
const float native_Hy_1 = (-native_Hy_0);
const float native_Hy_2 = corrected_0;
const float native_Hy_3 = corrected_1;
const float native_Hy_4 = __fadd_rn(native_Hy_2, native_Hy_3);
const float native_Hy_5 = env.Coefficient(0,1,z+(0),y+(0),x+(0));
const float native_Hy_6 = read(1,(version-1),0,0,0);
const float native_Hy_7 = __fmul_rn(native_Hy_5, native_Hy_6);
const float native_Hy_8 = __fmaf_rn(native_Hy_1, native_Hy_4, native_Hy_7);
result = native_Hy_8;
return env.Finish(1,version,z,y,x,result);
}
if constexpr(Phase==0&&Component==2){
if constexpr(CheckSupport){if(!env.Valid(2,z,y,x))return 0.f;}
float result=0.f;
const float derivative_0=env.Difference(0,4,2,z,y,x,(read(4,(version-1),0,0,0)),(read(4,(version-1),0,0,1)));
float corrected_0=__fmul_rn(1.0f,derivative_0);
if(env.LocalActive(4,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,4,z+(0),y+(0),x+(0)),env.LoadLocal(4,z,y,x),__fmul_rn(env.Coefficient(2,4,z+(0),y+(0),x+(0)),derivative_0));
corrected_0=__fmul_rn(1.0f,__fmaf_rn(derivative_0,env.Coefficient(4,4,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(4,z,y,x,next_psi);
}
const float derivative_1=env.Difference(0,3,1,z,y,x,(read(3,(version-1),0,0,0)),(read(3,(version-1),0,1,0)));
float corrected_1=__fmul_rn(-1.0f,derivative_1);
if(env.LocalActive(5,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,5,z+(0),y+(0),x+(0)),env.LoadLocal(5,z,y,x),__fmul_rn(env.Coefficient(2,5,z+(0),y+(0),x+(0)),derivative_1));
corrected_1=__fmul_rn(-1.0f,__fmaf_rn(derivative_1,env.Coefficient(4,5,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(5,z,y,x,next_psi);
}
const float native_Hz_0 = env.Coefficient(1,2,z+(0),y+(0),x+(0));
const float native_Hz_1 = (-native_Hz_0);
const float native_Hz_2 = corrected_0;
const float native_Hz_3 = corrected_1;
const float native_Hz_4 = __fadd_rn(native_Hz_2, native_Hz_3);
const float native_Hz_5 = env.Coefficient(0,2,z+(0),y+(0),x+(0));
const float native_Hz_6 = read(2,(version-1),0,0,0);
const float native_Hz_7 = __fmul_rn(native_Hz_5, native_Hz_6);
const float native_Hz_8 = __fmaf_rn(native_Hz_1, native_Hz_4, native_Hz_7);
result = native_Hz_8;
return env.Finish(2,version,z,y,x,result);
}
if constexpr(Phase==1&&Component==0){
if constexpr(CheckSupport){if(!env.Valid(3,z,y,x))return 0.f;}
float result=0.f;
const float derivative_0=env.Difference(1,2,1,z,y,x,(read(2,version,0,0,0)),(read(2,version,0,-1,0)));
float corrected_0=__fmul_rn(1.0f,derivative_0);
if(env.LocalActive(6,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,6,z+(0),y+(0),x+(0)),env.LoadLocal(6,z,y,x),__fmul_rn(env.Coefficient(2,6,z+(0),y+(0),x+(0)),derivative_0));
corrected_0=__fmul_rn(1.0f,__fmaf_rn(derivative_0,env.Coefficient(4,6,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(6,z,y,x,next_psi);
}
const float derivative_1=env.Difference(1,1,0,z,y,x,(read(1,version,0,0,0)),(read(1,version,-1,0,0)));
float corrected_1=__fmul_rn(-1.0f,derivative_1);
if(env.LocalActive(7,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,7,z+(0),y+(0),x+(0)),env.LoadLocal(7,z,y,x),__fmul_rn(env.Coefficient(2,7,z+(0),y+(0),x+(0)),derivative_1));
corrected_1=__fmul_rn(-1.0f,__fmaf_rn(derivative_1,env.Coefficient(4,7,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(7,z,y,x,next_psi);
}
const float native_Ex_0 = env.Coefficient(1,3,z+(0),y+(0),x+(0));
const float native_Ex_1 = corrected_0;
const float native_Ex_2 = corrected_1;
const float native_Ex_3 = __fadd_rn(native_Ex_1, native_Ex_2);
const float native_Ex_4 = env.Coefficient(0,3,z+(0),y+(0),x+(0));
const float native_Ex_5 = read(3,(version-1),0,0,0);
const float native_Ex_6 = __fmul_rn(native_Ex_4, native_Ex_5);
const float native_Ex_7 = __fmaf_rn(native_Ex_0, native_Ex_3, native_Ex_6);
result = native_Ex_7;
return env.Finish(3,version,z,y,x,result);
}
if constexpr(Phase==1&&Component==1){
if constexpr(CheckSupport){if(!env.Valid(4,z,y,x))return 0.f;}
float result=0.f;
const float derivative_0=env.Difference(1,0,0,z,y,x,(read(0,version,0,0,0)),(read(0,version,-1,0,0)));
float corrected_0=__fmul_rn(1.0f,derivative_0);
if(env.LocalActive(8,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,8,z+(0),y+(0),x+(0)),env.LoadLocal(8,z,y,x),__fmul_rn(env.Coefficient(2,8,z+(0),y+(0),x+(0)),derivative_0));
corrected_0=__fmul_rn(1.0f,__fmaf_rn(derivative_0,env.Coefficient(4,8,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(8,z,y,x,next_psi);
}
const float derivative_1=env.Difference(1,2,2,z,y,x,(read(2,version,0,0,0)),(read(2,version,0,0,-1)));
float corrected_1=__fmul_rn(-1.0f,derivative_1);
if(env.LocalActive(9,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,9,z+(0),y+(0),x+(0)),env.LoadLocal(9,z,y,x),__fmul_rn(env.Coefficient(2,9,z+(0),y+(0),x+(0)),derivative_1));
corrected_1=__fmul_rn(-1.0f,__fmaf_rn(derivative_1,env.Coefficient(4,9,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(9,z,y,x,next_psi);
}
const float native_Ey_0 = env.Coefficient(1,4,z+(0),y+(0),x+(0));
const float native_Ey_1 = corrected_0;
const float native_Ey_2 = corrected_1;
const float native_Ey_3 = __fadd_rn(native_Ey_1, native_Ey_2);
const float native_Ey_4 = env.Coefficient(0,4,z+(0),y+(0),x+(0));
const float native_Ey_5 = read(4,(version-1),0,0,0);
const float native_Ey_6 = __fmul_rn(native_Ey_4, native_Ey_5);
const float native_Ey_7 = __fmaf_rn(native_Ey_0, native_Ey_3, native_Ey_6);
result = native_Ey_7;
return env.Finish(4,version,z,y,x,result);
}
if constexpr(Phase==1&&Component==2){
if constexpr(CheckSupport){if(!env.Valid(5,z,y,x))return 0.f;}
float result=0.f;
const float derivative_0=env.Difference(1,1,2,z,y,x,(read(1,version,0,0,0)),(read(1,version,0,0,-1)));
float corrected_0=__fmul_rn(1.0f,derivative_0);
if(env.LocalActive(10,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,10,z+(0),y+(0),x+(0)),env.LoadLocal(10,z,y,x),__fmul_rn(env.Coefficient(2,10,z+(0),y+(0),x+(0)),derivative_0));
corrected_0=__fmul_rn(1.0f,__fmaf_rn(derivative_0,env.Coefficient(4,10,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(10,z,y,x,next_psi);
}
const float derivative_1=env.Difference(1,0,1,z,y,x,(read(0,version,0,0,0)),(read(0,version,0,-1,0)));
float corrected_1=__fmul_rn(-1.0f,derivative_1);
if(env.LocalActive(11,z,y,x)) {
const float next_psi=__fmaf_rn(env.Coefficient(3,11,z+(0),y+(0),x+(0)),env.LoadLocal(11,z,y,x),__fmul_rn(env.Coefficient(2,11,z+(0),y+(0),x+(0)),derivative_1));
corrected_1=__fmul_rn(-1.0f,__fmaf_rn(derivative_1,env.Coefficient(4,11,z+(0),y+(0),x+(0)),next_psi));
env.StoreLocal(11,z,y,x,next_psi);
}
const float native_Ez_0 = env.Coefficient(1,5,z+(0),y+(0),x+(0));
const float native_Ez_1 = corrected_0;
const float native_Ez_2 = corrected_1;
const float native_Ez_3 = __fadd_rn(native_Ez_1, native_Ez_2);
const float native_Ez_4 = env.Coefficient(0,5,z+(0),y+(0),x+(0));
const float native_Ez_5 = read(5,(version-1),0,0,0);
const float native_Ez_6 = __fmul_rn(native_Ez_4, native_Ez_5);
const float native_Ez_7 = __fmaf_rn(native_Ez_0, native_Ez_3, native_Ez_6);
result = native_Ez_7;
return env.Finish(5,version,z,y,x,result);
}
return 0.f;
}
} } } // namespace beamz::cuda::trace_native
