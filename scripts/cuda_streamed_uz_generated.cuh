// Deterministic output of scripts/cuda_space_time_codegen.py.
// Callback contract is recorded in the companion manifest; research only.
#pragma once
#include <cuda_bf16.h>
namespace beamz { namespace cuda { namespace streamed_ir {
inline constexpr char SemanticSHA256[]="4e27e7fbca4c7b6134f7977a27727a20d74dd05be652a759b317f718e4a0fd2c";
inline constexpr int Depth=4, Threads=512, SharedBytes=8192;
template<class Environment> __device__ __forceinline__
void RunStreamedTile(Environment &env,float *exchange,int oz,int oy,int ox) {
const int q=threadIdx.x,qx=q&31,qy=q>>5;
float history_Ex_1=0.f;
float history_Ey_1=0.f;
float history_Ez_1=0.f;
float history_Hx_1=0.f;
float history_Hy_1=0.f;
float history_Hz_1=0.f;
float history_Ex_2=0.f;
float history_Ey_2=0.f;
float history_Ez_2=0.f;
float history_Hx_2=0.f;
float history_Hy_2=0.f;
float history_Hz_2=0.f;
float history_Ex_3=0.f;
float history_Ey_3=0.f;
float history_Ez_3=0.f;
float history_Hx_3=0.f;
float history_Hy_3=0.f;
float history_Hz_3=0.f;
float history_Hx_4=0.f;
float history_Hy_4=0.f;
for(int iz=0;iz<64;++iz) {
float old_h_x=0.f,old_h_y=0.f,old_h_z=0.f;
float old_e_x=0.f,old_e_y=0.f,old_e_z=0.f;
float current_h_x=0.f,current_h_y=0.f,current_h_z=0.f;
float current_e_x=0.f,current_e_y=0.f,current_e_z=0.f;
{
const bool owned_xy=qx>=3&&qx<32&&qy>=3&&qy<16;
__syncthreads(); // A: publish previous-E exchange; retire previous E's H readers.
{
const int z=oz+iz-1;
const int y=oy-4+qy,x=ox-4+qx;
{
const float captured_old=history_Hx_1;
float next_h_x=0.f;
if(owned_xy&&env.Valid(0,z,y,x)) {
const float s1_H_x_0 = env.Coefficient(1,0,z,y,x);
const float s1_H_x_1 = (-s1_H_x_0);
const float s1_H_x_2 = env.Read(5,0,z+(0),y+(1),x+(0));
const float s1_H_x_3 = env.Read(5,0,z+(0),y+(0),x+(0));
const float s1_H_x_4 = __fsub_rn(s1_H_x_2, s1_H_x_3);
const float s1_H_x_5 = env.inv_spacing;
const float s1_H_x_6 = __fmul_rn(s1_H_x_4, s1_H_x_5);
const float s1_H_x_7 = env.Read(4,0,z+(1),y+(0),x+(0));
const float s1_H_x_8 = env.Read(4,0,z+(0),y+(0),x+(0));
const float s1_H_x_9 = __fsub_rn(s1_H_x_7, s1_H_x_8);
const float s1_H_x_10 = __fmul_rn(s1_H_x_9, s1_H_x_5);
const float s1_H_x_11 = __fsub_rn(s1_H_x_6, s1_H_x_10);
const float s1_H_x_12 = env.Coefficient(0,0,z,y,x);
const float s1_H_x_13 = env.Read(0,0,z+(0),y+(0),x+(0));
const float s1_H_x_14 = __fmul_rn(s1_H_x_12, s1_H_x_13);
const float s1_H_x_15 = __fmaf_rn(s1_H_x_1, s1_H_x_11, s1_H_x_14);
next_h_x = s1_H_x_15;
next_h_x=env.Finish(0,1,z,y,x,next_h_x);
}
old_h_x=captured_old;
current_h_x=next_h_x;
history_Hx_1=next_h_x;
}
{
const float captured_old=history_Hy_1;
float next_h_y=0.f;
if(owned_xy&&env.Valid(1,z,y,x)) {
const float s1_H_y_0 = env.Coefficient(1,1,z,y,x);
const float s1_H_y_1 = (-s1_H_y_0);
const float s1_H_y_2 = env.Read(3,0,z+(1),y+(0),x+(0));
const float s1_H_y_3 = env.Read(3,0,z+(0),y+(0),x+(0));
const float s1_H_y_4 = __fsub_rn(s1_H_y_2, s1_H_y_3);
const float s1_H_y_5 = env.inv_spacing;
const float s1_H_y_6 = __fmul_rn(s1_H_y_4, s1_H_y_5);
const float s1_H_y_7 = env.Read(5,0,z+(0),y+(0),x+(1));
const float s1_H_y_8 = env.Read(5,0,z+(0),y+(0),x+(0));
const float s1_H_y_9 = __fsub_rn(s1_H_y_7, s1_H_y_8);
const float s1_H_y_10 = __fmul_rn(s1_H_y_9, s1_H_y_5);
const float s1_H_y_11 = __fsub_rn(s1_H_y_6, s1_H_y_10);
const float s1_H_y_12 = env.Coefficient(0,1,z,y,x);
const float s1_H_y_13 = env.Read(1,0,z+(0),y+(0),x+(0));
const float s1_H_y_14 = __fmul_rn(s1_H_y_12, s1_H_y_13);
const float s1_H_y_15 = __fmaf_rn(s1_H_y_1, s1_H_y_11, s1_H_y_14);
next_h_y = s1_H_y_15;
next_h_y=env.Finish(1,1,z,y,x,next_h_y);
}
old_h_y=captured_old;
current_h_y=next_h_y;
history_Hy_1=next_h_y;
}
{
const float captured_old=history_Hz_1;
float next_h_z=0.f;
if(owned_xy&&env.Valid(2,z,y,x)) {
const float s1_H_z_0 = env.Coefficient(1,2,z,y,x);
const float s1_H_z_1 = (-s1_H_z_0);
const float s1_H_z_2 = env.Read(4,0,z+(0),y+(0),x+(1));
const float s1_H_z_3 = env.Read(4,0,z+(0),y+(0),x+(0));
const float s1_H_z_4 = __fsub_rn(s1_H_z_2, s1_H_z_3);
const float s1_H_z_5 = env.inv_spacing;
const float s1_H_z_6 = __fmul_rn(s1_H_z_4, s1_H_z_5);
const float s1_H_z_7 = env.Read(3,0,z+(0),y+(1),x+(0));
const float s1_H_z_8 = env.Read(3,0,z+(0),y+(0),x+(0));
const float s1_H_z_9 = __fsub_rn(s1_H_z_7, s1_H_z_8);
const float s1_H_z_10 = __fmul_rn(s1_H_z_9, s1_H_z_5);
const float s1_H_z_11 = __fsub_rn(s1_H_z_6, s1_H_z_10);
const float s1_H_z_12 = env.Coefficient(0,2,z,y,x);
const float s1_H_z_13 = env.Read(2,0,z+(0),y+(0),x+(0));
const float s1_H_z_14 = __fmul_rn(s1_H_z_12, s1_H_z_13);
const float s1_H_z_15 = __fmaf_rn(s1_H_z_1, s1_H_z_11, s1_H_z_14);
next_h_z = s1_H_z_15;
next_h_z=env.Finish(2,1,z,y,x,next_h_z);
}
old_h_z=captured_old;
current_h_z=next_h_z;
history_Hz_1=next_h_z;
}
}
exchange[0+q]=current_h_x;
exchange[512+q]=current_h_z;
__syncthreads(); // B: publish H exchange; retire H's previous-E readers.
{
const int z=oz+iz-1;
const int y=oy-4+qy,x=ox-4+qx;
const float shuffle_1_1_0=__shfl_sync(0xffffffffu,current_h_z,(qx+(-1))&31);
const float shuffle_1_1_1=__shfl_sync(0xffffffffu,current_h_y,(qx+(-1))&31);
{
const float captured_old=history_Ex_1;
float next_e_x=0.f;
if(owned_xy&&env.Valid(3,z,y,x)) {
const float s1_E_x_0 = env.Coefficient(1,3,z,y,x);
const float s1_E_x_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(current_h_z):(env.Read(2,1,z+(0),y+(0),x+(0))));
const float s1_E_x_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-4)>=0&&qy+(-4)<13&&qx+(-3)>=0&&qx+(-3)<29)?(exchange[512+q+(-1)*32+(0)]):(env.Read(2,1,z+(0),y+(-1),x+(0))));
const float s1_E_x_3 = __fsub_rn(s1_E_x_1, s1_E_x_2);
const float s1_E_x_4 = env.inv_spacing;
const float s1_E_x_5 = __fmul_rn(s1_E_x_3, s1_E_x_4);
const float s1_E_x_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(current_h_y):(env.Read(1,1,z+(0),y+(0),x+(0))));
const float s1_E_x_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_h_y):(env.Read(1,1,z+(-1),y+(0),x+(0))));
const float s1_E_x_8 = __fsub_rn(s1_E_x_6, s1_E_x_7);
const float s1_E_x_9 = __fmul_rn(s1_E_x_8, s1_E_x_4);
const float s1_E_x_10 = __fsub_rn(s1_E_x_5, s1_E_x_9);
const float s1_E_x_11 = env.Coefficient(0,3,z,y,x);
const float s1_E_x_12 = env.Read(3,0,z+(0),y+(0),x+(0));
const float s1_E_x_13 = __fmul_rn(s1_E_x_11, s1_E_x_12);
const float s1_E_x_14 = __fmaf_rn(s1_E_x_0, s1_E_x_10, s1_E_x_13);
next_e_x = s1_E_x_14;
next_e_x=env.Finish(3,1,z,y,x,next_e_x);
}
old_e_x=captured_old;
current_e_x=next_e_x;
history_Ex_1=next_e_x;
}
{
const float captured_old=history_Ey_1;
float next_e_y=0.f;
if(owned_xy&&env.Valid(4,z,y,x)) {
const float s1_E_y_0 = env.Coefficient(1,4,z,y,x);
const float s1_E_y_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(current_h_x):(env.Read(0,1,z+(0),y+(0),x+(0))));
const float s1_E_y_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_h_x):(env.Read(0,1,z+(-1),y+(0),x+(0))));
const float s1_E_y_3 = __fsub_rn(s1_E_y_1, s1_E_y_2);
const float s1_E_y_4 = env.inv_spacing;
const float s1_E_y_5 = __fmul_rn(s1_E_y_3, s1_E_y_4);
const float s1_E_y_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(current_h_z):(env.Read(2,1,z+(0),y+(0),x+(0))));
const float s1_E_y_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-4)>=0&&qx+(-4)<29)?(shuffle_1_1_0):(env.Read(2,1,z+(0),y+(0),x+(-1))));
const float s1_E_y_8 = __fsub_rn(s1_E_y_6, s1_E_y_7);
const float s1_E_y_9 = __fmul_rn(s1_E_y_8, s1_E_y_4);
const float s1_E_y_10 = __fsub_rn(s1_E_y_5, s1_E_y_9);
const float s1_E_y_11 = env.Coefficient(0,4,z,y,x);
const float s1_E_y_12 = env.Read(4,0,z+(0),y+(0),x+(0));
const float s1_E_y_13 = __fmul_rn(s1_E_y_11, s1_E_y_12);
const float s1_E_y_14 = __fmaf_rn(s1_E_y_0, s1_E_y_10, s1_E_y_13);
next_e_y = s1_E_y_14;
next_e_y=env.Finish(4,1,z,y,x,next_e_y);
}
old_e_y=captured_old;
current_e_y=next_e_y;
history_Ey_1=next_e_y;
}
{
const float captured_old=history_Ez_1;
float next_e_z=0.f;
if(owned_xy&&env.Valid(5,z,y,x)) {
const float s1_E_z_0 = env.Coefficient(1,5,z,y,x);
const float s1_E_z_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(current_h_y):(env.Read(1,1,z+(0),y+(0),x+(0))));
const float s1_E_z_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-4)>=0&&qx+(-4)<29)?(shuffle_1_1_1):(env.Read(1,1,z+(0),y+(0),x+(-1))));
const float s1_E_z_3 = __fsub_rn(s1_E_z_1, s1_E_z_2);
const float s1_E_z_4 = env.inv_spacing;
const float s1_E_z_5 = __fmul_rn(s1_E_z_3, s1_E_z_4);
const float s1_E_z_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(current_h_x):(env.Read(0,1,z+(0),y+(0),x+(0))));
const float s1_E_z_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-4)>=0&&qy+(-4)<13&&qx+(-3)>=0&&qx+(-3)<29)?(exchange[0+q+(-1)*32+(0)]):(env.Read(0,1,z+(0),y+(-1),x+(0))));
const float s1_E_z_8 = __fsub_rn(s1_E_z_6, s1_E_z_7);
const float s1_E_z_9 = __fmul_rn(s1_E_z_8, s1_E_z_4);
const float s1_E_z_10 = __fsub_rn(s1_E_z_5, s1_E_z_9);
const float s1_E_z_11 = env.Coefficient(0,5,z,y,x);
const float s1_E_z_12 = env.Read(5,0,z+(0),y+(0),x+(0));
const float s1_E_z_13 = __fmul_rn(s1_E_z_11, s1_E_z_12);
const float s1_E_z_14 = __fmaf_rn(s1_E_z_0, s1_E_z_10, s1_E_z_13);
next_e_z = s1_E_z_14;
next_e_z=env.Finish(5,1,z,y,x,next_e_z);
}
old_e_z=captured_old;
current_e_z=next_e_z;
history_Ez_1=next_e_z;
}
if(owned_xy) {
env.Publish(0,1,z,y,x,current_h_x);
env.Publish(1,1,z,y,x,current_h_y);
env.Publish(2,1,z,y,x,current_h_z);
env.Publish(3,1,z,y,x,current_e_x);
env.Publish(4,1,z,y,x,current_e_y);
env.Publish(5,1,z,y,x,current_e_z);
}
}
}
{
const bool owned_xy=qx>=2&&qx<31&&qy>=2&&qy<15;
exchange[1024+q]=old_e_x;
exchange[1536+q]=old_e_z;
__syncthreads(); // A: publish previous-E exchange; retire previous E's H readers.
{
const int z=oz+iz-2;
const int y=oy-4+qy,x=ox-4+qx;
const float shuffle_2_0_0=__shfl_sync(0xffffffffu,old_e_z,(qx+(1))&31);
const float shuffle_2_0_1=__shfl_sync(0xffffffffu,old_e_y,(qx+(1))&31);
{
const float captured_old=history_Hx_2;
float next_h_x=0.f;
if(owned_xy&&env.Valid(0,z,y,x)) {
const float s2_H_x_0 = env.Coefficient(1,0,z,y,x);
const float s2_H_x_1 = (-s2_H_x_0);
const float s2_H_x_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-3)>=0&&qx+(-3)<29)?(exchange[1536+q+(1)*32+(0)]):(env.Read(5,1,z+(0),y+(1),x+(0))));
const float s2_H_x_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_z):(env.Read(5,1,z+(0),y+(0),x+(0))));
const float s2_H_x_4 = __fsub_rn(s2_H_x_2, s2_H_x_3);
const float s2_H_x_5 = env.inv_spacing;
const float s2_H_x_6 = __fmul_rn(s2_H_x_4, s2_H_x_5);
const float s2_H_x_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(current_e_y):(env.Read(4,1,z+(1),y+(0),x+(0))));
const float s2_H_x_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_y):(env.Read(4,1,z+(0),y+(0),x+(0))));
const float s2_H_x_9 = __fsub_rn(s2_H_x_7, s2_H_x_8);
const float s2_H_x_10 = __fmul_rn(s2_H_x_9, s2_H_x_5);
const float s2_H_x_11 = __fsub_rn(s2_H_x_6, s2_H_x_10);
const float s2_H_x_12 = env.Coefficient(0,0,z,y,x);
const float s2_H_x_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_h_x):(env.Read(0,1,z+(0),y+(0),x+(0))));
const float s2_H_x_14 = __fmul_rn(s2_H_x_12, s2_H_x_13);
const float s2_H_x_15 = __fmaf_rn(s2_H_x_1, s2_H_x_11, s2_H_x_14);
next_h_x = s2_H_x_15;
next_h_x=env.Finish(0,2,z,y,x,next_h_x);
}
old_h_x=captured_old;
current_h_x=next_h_x;
history_Hx_2=next_h_x;
}
{
const float captured_old=history_Hy_2;
float next_h_y=0.f;
if(owned_xy&&env.Valid(1,z,y,x)) {
const float s2_H_y_0 = env.Coefficient(1,1,z,y,x);
const float s2_H_y_1 = (-s2_H_y_0);
const float s2_H_y_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(current_e_x):(env.Read(3,1,z+(1),y+(0),x+(0))));
const float s2_H_y_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_x):(env.Read(3,1,z+(0),y+(0),x+(0))));
const float s2_H_y_4 = __fsub_rn(s2_H_y_2, s2_H_y_3);
const float s2_H_y_5 = env.inv_spacing;
const float s2_H_y_6 = __fmul_rn(s2_H_y_4, s2_H_y_5);
const float s2_H_y_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-2)>=0&&qx+(-2)<29)?(shuffle_2_0_0):(env.Read(5,1,z+(0),y+(0),x+(1))));
const float s2_H_y_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_z):(env.Read(5,1,z+(0),y+(0),x+(0))));
const float s2_H_y_9 = __fsub_rn(s2_H_y_7, s2_H_y_8);
const float s2_H_y_10 = __fmul_rn(s2_H_y_9, s2_H_y_5);
const float s2_H_y_11 = __fsub_rn(s2_H_y_6, s2_H_y_10);
const float s2_H_y_12 = env.Coefficient(0,1,z,y,x);
const float s2_H_y_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_h_y):(env.Read(1,1,z+(0),y+(0),x+(0))));
const float s2_H_y_14 = __fmul_rn(s2_H_y_12, s2_H_y_13);
const float s2_H_y_15 = __fmaf_rn(s2_H_y_1, s2_H_y_11, s2_H_y_14);
next_h_y = s2_H_y_15;
next_h_y=env.Finish(1,2,z,y,x,next_h_y);
}
old_h_y=captured_old;
current_h_y=next_h_y;
history_Hy_2=next_h_y;
}
{
const float captured_old=history_Hz_2;
float next_h_z=0.f;
if(owned_xy&&env.Valid(2,z,y,x)) {
const float s2_H_z_0 = env.Coefficient(1,2,z,y,x);
const float s2_H_z_1 = (-s2_H_z_0);
const float s2_H_z_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-2)>=0&&qx+(-2)<29)?(shuffle_2_0_1):(env.Read(4,1,z+(0),y+(0),x+(1))));
const float s2_H_z_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_y):(env.Read(4,1,z+(0),y+(0),x+(0))));
const float s2_H_z_4 = __fsub_rn(s2_H_z_2, s2_H_z_3);
const float s2_H_z_5 = env.inv_spacing;
const float s2_H_z_6 = __fmul_rn(s2_H_z_4, s2_H_z_5);
const float s2_H_z_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-3)>=0&&qx+(-3)<29)?(exchange[1024+q+(1)*32+(0)]):(env.Read(3,1,z+(0),y+(1),x+(0))));
const float s2_H_z_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_x):(env.Read(3,1,z+(0),y+(0),x+(0))));
const float s2_H_z_9 = __fsub_rn(s2_H_z_7, s2_H_z_8);
const float s2_H_z_10 = __fmul_rn(s2_H_z_9, s2_H_z_5);
const float s2_H_z_11 = __fsub_rn(s2_H_z_6, s2_H_z_10);
const float s2_H_z_12 = env.Coefficient(0,2,z,y,x);
const float s2_H_z_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_h_z):(env.Read(2,1,z+(0),y+(0),x+(0))));
const float s2_H_z_14 = __fmul_rn(s2_H_z_12, s2_H_z_13);
const float s2_H_z_15 = __fmaf_rn(s2_H_z_1, s2_H_z_11, s2_H_z_14);
next_h_z = s2_H_z_15;
next_h_z=env.Finish(2,2,z,y,x,next_h_z);
}
old_h_z=captured_old;
current_h_z=next_h_z;
history_Hz_2=next_h_z;
}
}
exchange[0+q]=current_h_x;
exchange[512+q]=current_h_z;
__syncthreads(); // B: publish H exchange; retire H's previous-E readers.
{
const int z=oz+iz-2;
const int y=oy-4+qy,x=ox-4+qx;
const float shuffle_2_1_0=__shfl_sync(0xffffffffu,current_h_z,(qx+(-1))&31);
const float shuffle_2_1_1=__shfl_sync(0xffffffffu,current_h_y,(qx+(-1))&31);
{
const float captured_old=history_Ex_2;
float next_e_x=0.f;
if(owned_xy&&env.Valid(3,z,y,x)) {
const float s2_E_x_0 = env.Coefficient(1,3,z,y,x);
const float s2_E_x_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(current_h_z):(env.Read(2,2,z+(0),y+(0),x+(0))));
const float s2_E_x_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-2)>=0&&qx+(-2)<29)?(exchange[512+q+(-1)*32+(0)]):(env.Read(2,2,z+(0),y+(-1),x+(0))));
const float s2_E_x_3 = __fsub_rn(s2_E_x_1, s2_E_x_2);
const float s2_E_x_4 = env.inv_spacing;
const float s2_E_x_5 = __fmul_rn(s2_E_x_3, s2_E_x_4);
const float s2_E_x_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(current_h_y):(env.Read(1,2,z+(0),y+(0),x+(0))));
const float s2_E_x_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_h_y):(env.Read(1,2,z+(-1),y+(0),x+(0))));
const float s2_E_x_8 = __fsub_rn(s2_E_x_6, s2_E_x_7);
const float s2_E_x_9 = __fmul_rn(s2_E_x_8, s2_E_x_4);
const float s2_E_x_10 = __fsub_rn(s2_E_x_5, s2_E_x_9);
const float s2_E_x_11 = env.Coefficient(0,3,z,y,x);
const float s2_E_x_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_x):(env.Read(3,1,z+(0),y+(0),x+(0))));
const float s2_E_x_13 = __fmul_rn(s2_E_x_11, s2_E_x_12);
const float s2_E_x_14 = __fmaf_rn(s2_E_x_0, s2_E_x_10, s2_E_x_13);
next_e_x = s2_E_x_14;
next_e_x=env.Finish(3,2,z,y,x,next_e_x);
}
old_e_x=captured_old;
current_e_x=next_e_x;
history_Ex_2=next_e_x;
}
{
const float captured_old=history_Ey_2;
float next_e_y=0.f;
if(owned_xy&&env.Valid(4,z,y,x)) {
const float s2_E_y_0 = env.Coefficient(1,4,z,y,x);
const float s2_E_y_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(current_h_x):(env.Read(0,2,z+(0),y+(0),x+(0))));
const float s2_E_y_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_h_x):(env.Read(0,2,z+(-1),y+(0),x+(0))));
const float s2_E_y_3 = __fsub_rn(s2_E_y_1, s2_E_y_2);
const float s2_E_y_4 = env.inv_spacing;
const float s2_E_y_5 = __fmul_rn(s2_E_y_3, s2_E_y_4);
const float s2_E_y_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(current_h_z):(env.Read(2,2,z+(0),y+(0),x+(0))));
const float s2_E_y_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-3)>=0&&qx+(-3)<29)?(shuffle_2_1_0):(env.Read(2,2,z+(0),y+(0),x+(-1))));
const float s2_E_y_8 = __fsub_rn(s2_E_y_6, s2_E_y_7);
const float s2_E_y_9 = __fmul_rn(s2_E_y_8, s2_E_y_4);
const float s2_E_y_10 = __fsub_rn(s2_E_y_5, s2_E_y_9);
const float s2_E_y_11 = env.Coefficient(0,4,z,y,x);
const float s2_E_y_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_y):(env.Read(4,1,z+(0),y+(0),x+(0))));
const float s2_E_y_13 = __fmul_rn(s2_E_y_11, s2_E_y_12);
const float s2_E_y_14 = __fmaf_rn(s2_E_y_0, s2_E_y_10, s2_E_y_13);
next_e_y = s2_E_y_14;
next_e_y=env.Finish(4,2,z,y,x,next_e_y);
}
old_e_y=captured_old;
current_e_y=next_e_y;
history_Ey_2=next_e_y;
}
{
const float captured_old=history_Ez_2;
float next_e_z=0.f;
if(owned_xy&&env.Valid(5,z,y,x)) {
const float s2_E_z_0 = env.Coefficient(1,5,z,y,x);
const float s2_E_z_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(current_h_y):(env.Read(1,2,z+(0),y+(0),x+(0))));
const float s2_E_z_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-3)>=0&&qx+(-3)<29)?(shuffle_2_1_1):(env.Read(1,2,z+(0),y+(0),x+(-1))));
const float s2_E_z_3 = __fsub_rn(s2_E_z_1, s2_E_z_2);
const float s2_E_z_4 = env.inv_spacing;
const float s2_E_z_5 = __fmul_rn(s2_E_z_3, s2_E_z_4);
const float s2_E_z_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(current_h_x):(env.Read(0,2,z+(0),y+(0),x+(0))));
const float s2_E_z_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-2)>=0&&qx+(-2)<29)?(exchange[0+q+(-1)*32+(0)]):(env.Read(0,2,z+(0),y+(-1),x+(0))));
const float s2_E_z_8 = __fsub_rn(s2_E_z_6, s2_E_z_7);
const float s2_E_z_9 = __fmul_rn(s2_E_z_8, s2_E_z_4);
const float s2_E_z_10 = __fsub_rn(s2_E_z_5, s2_E_z_9);
const float s2_E_z_11 = env.Coefficient(0,5,z,y,x);
const float s2_E_z_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-3)>=0&&qy+(-3)<13&&qx+(-3)>=0&&qx+(-3)<29)?(old_e_z):(env.Read(5,1,z+(0),y+(0),x+(0))));
const float s2_E_z_13 = __fmul_rn(s2_E_z_11, s2_E_z_12);
const float s2_E_z_14 = __fmaf_rn(s2_E_z_0, s2_E_z_10, s2_E_z_13);
next_e_z = s2_E_z_14;
next_e_z=env.Finish(5,2,z,y,x,next_e_z);
}
old_e_z=captured_old;
current_e_z=next_e_z;
history_Ez_2=next_e_z;
}
if(owned_xy) {
env.Publish(0,2,z,y,x,current_h_x);
env.Publish(1,2,z,y,x,current_h_y);
env.Publish(2,2,z,y,x,current_h_z);
env.Publish(3,2,z,y,x,current_e_x);
env.Publish(4,2,z,y,x,current_e_y);
env.Publish(5,2,z,y,x,current_e_z);
}
}
}
{
const bool owned_xy=qx>=1&&qx<30&&qy>=1&&qy<14;
exchange[1024+q]=old_e_x;
exchange[1536+q]=old_e_z;
__syncthreads(); // A: publish previous-E exchange; retire previous E's H readers.
{
const int z=oz+iz-3;
const int y=oy-4+qy,x=ox-4+qx;
const float shuffle_3_0_0=__shfl_sync(0xffffffffu,old_e_z,(qx+(1))&31);
const float shuffle_3_0_1=__shfl_sync(0xffffffffu,old_e_y,(qx+(1))&31);
{
const float captured_old=history_Hx_3;
float next_h_x=0.f;
if(owned_xy&&env.Valid(0,z,y,x)) {
const float s3_H_x_0 = env.Coefficient(1,0,z,y,x);
const float s3_H_x_1 = (-s3_H_x_0);
const float s3_H_x_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-2)>=0&&qx+(-2)<29)?(exchange[1536+q+(1)*32+(0)]):(env.Read(5,2,z+(0),y+(1),x+(0))));
const float s3_H_x_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_z):(env.Read(5,2,z+(0),y+(0),x+(0))));
const float s3_H_x_4 = __fsub_rn(s3_H_x_2, s3_H_x_3);
const float s3_H_x_5 = env.inv_spacing;
const float s3_H_x_6 = __fmul_rn(s3_H_x_4, s3_H_x_5);
const float s3_H_x_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(current_e_y):(env.Read(4,2,z+(1),y+(0),x+(0))));
const float s3_H_x_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_y):(env.Read(4,2,z+(0),y+(0),x+(0))));
const float s3_H_x_9 = __fsub_rn(s3_H_x_7, s3_H_x_8);
const float s3_H_x_10 = __fmul_rn(s3_H_x_9, s3_H_x_5);
const float s3_H_x_11 = __fsub_rn(s3_H_x_6, s3_H_x_10);
const float s3_H_x_12 = env.Coefficient(0,0,z,y,x);
const float s3_H_x_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_h_x):(env.Read(0,2,z+(0),y+(0),x+(0))));
const float s3_H_x_14 = __fmul_rn(s3_H_x_12, s3_H_x_13);
const float s3_H_x_15 = __fmaf_rn(s3_H_x_1, s3_H_x_11, s3_H_x_14);
next_h_x = s3_H_x_15;
next_h_x=env.Finish(0,3,z,y,x,next_h_x);
}
old_h_x=captured_old;
current_h_x=next_h_x;
history_Hx_3=next_h_x;
}
{
const float captured_old=history_Hy_3;
float next_h_y=0.f;
if(owned_xy&&env.Valid(1,z,y,x)) {
const float s3_H_y_0 = env.Coefficient(1,1,z,y,x);
const float s3_H_y_1 = (-s3_H_y_0);
const float s3_H_y_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(current_e_x):(env.Read(3,2,z+(1),y+(0),x+(0))));
const float s3_H_y_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_x):(env.Read(3,2,z+(0),y+(0),x+(0))));
const float s3_H_y_4 = __fsub_rn(s3_H_y_2, s3_H_y_3);
const float s3_H_y_5 = env.inv_spacing;
const float s3_H_y_6 = __fmul_rn(s3_H_y_4, s3_H_y_5);
const float s3_H_y_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-1)>=0&&qx+(-1)<29)?(shuffle_3_0_0):(env.Read(5,2,z+(0),y+(0),x+(1))));
const float s3_H_y_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_z):(env.Read(5,2,z+(0),y+(0),x+(0))));
const float s3_H_y_9 = __fsub_rn(s3_H_y_7, s3_H_y_8);
const float s3_H_y_10 = __fmul_rn(s3_H_y_9, s3_H_y_5);
const float s3_H_y_11 = __fsub_rn(s3_H_y_6, s3_H_y_10);
const float s3_H_y_12 = env.Coefficient(0,1,z,y,x);
const float s3_H_y_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_h_y):(env.Read(1,2,z+(0),y+(0),x+(0))));
const float s3_H_y_14 = __fmul_rn(s3_H_y_12, s3_H_y_13);
const float s3_H_y_15 = __fmaf_rn(s3_H_y_1, s3_H_y_11, s3_H_y_14);
next_h_y = s3_H_y_15;
next_h_y=env.Finish(1,3,z,y,x,next_h_y);
}
old_h_y=captured_old;
current_h_y=next_h_y;
history_Hy_3=next_h_y;
}
{
const float captured_old=history_Hz_3;
float next_h_z=0.f;
if(owned_xy&&env.Valid(2,z,y,x)) {
const float s3_H_z_0 = env.Coefficient(1,2,z,y,x);
const float s3_H_z_1 = (-s3_H_z_0);
const float s3_H_z_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-1)>=0&&qx+(-1)<29)?(shuffle_3_0_1):(env.Read(4,2,z+(0),y+(0),x+(1))));
const float s3_H_z_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_y):(env.Read(4,2,z+(0),y+(0),x+(0))));
const float s3_H_z_4 = __fsub_rn(s3_H_z_2, s3_H_z_3);
const float s3_H_z_5 = env.inv_spacing;
const float s3_H_z_6 = __fmul_rn(s3_H_z_4, s3_H_z_5);
const float s3_H_z_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-2)>=0&&qx+(-2)<29)?(exchange[1024+q+(1)*32+(0)]):(env.Read(3,2,z+(0),y+(1),x+(0))));
const float s3_H_z_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_x):(env.Read(3,2,z+(0),y+(0),x+(0))));
const float s3_H_z_9 = __fsub_rn(s3_H_z_7, s3_H_z_8);
const float s3_H_z_10 = __fmul_rn(s3_H_z_9, s3_H_z_5);
const float s3_H_z_11 = __fsub_rn(s3_H_z_6, s3_H_z_10);
const float s3_H_z_12 = env.Coefficient(0,2,z,y,x);
const float s3_H_z_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_h_z):(env.Read(2,2,z+(0),y+(0),x+(0))));
const float s3_H_z_14 = __fmul_rn(s3_H_z_12, s3_H_z_13);
const float s3_H_z_15 = __fmaf_rn(s3_H_z_1, s3_H_z_11, s3_H_z_14);
next_h_z = s3_H_z_15;
next_h_z=env.Finish(2,3,z,y,x,next_h_z);
}
old_h_z=captured_old;
current_h_z=next_h_z;
history_Hz_3=next_h_z;
}
}
exchange[0+q]=current_h_x;
exchange[512+q]=current_h_z;
__syncthreads(); // B: publish H exchange; retire H's previous-E readers.
{
const int z=oz+iz-3;
const int y=oy-4+qy,x=ox-4+qx;
const float shuffle_3_1_0=__shfl_sync(0xffffffffu,current_h_z,(qx+(-1))&31);
const float shuffle_3_1_1=__shfl_sync(0xffffffffu,current_h_y,(qx+(-1))&31);
{
const float captured_old=history_Ex_3;
float next_e_x=0.f;
if(owned_xy&&env.Valid(3,z,y,x)) {
const float s3_E_x_0 = env.Coefficient(1,3,z,y,x);
const float s3_E_x_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(current_h_z):(env.Read(2,3,z+(0),y+(0),x+(0))));
const float s3_E_x_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-1)>=0&&qx+(-1)<29)?(exchange[512+q+(-1)*32+(0)]):(env.Read(2,3,z+(0),y+(-1),x+(0))));
const float s3_E_x_3 = __fsub_rn(s3_E_x_1, s3_E_x_2);
const float s3_E_x_4 = env.inv_spacing;
const float s3_E_x_5 = __fmul_rn(s3_E_x_3, s3_E_x_4);
const float s3_E_x_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(current_h_y):(env.Read(1,3,z+(0),y+(0),x+(0))));
const float s3_E_x_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_h_y):(env.Read(1,3,z+(-1),y+(0),x+(0))));
const float s3_E_x_8 = __fsub_rn(s3_E_x_6, s3_E_x_7);
const float s3_E_x_9 = __fmul_rn(s3_E_x_8, s3_E_x_4);
const float s3_E_x_10 = __fsub_rn(s3_E_x_5, s3_E_x_9);
const float s3_E_x_11 = env.Coefficient(0,3,z,y,x);
const float s3_E_x_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_x):(env.Read(3,2,z+(0),y+(0),x+(0))));
const float s3_E_x_13 = __fmul_rn(s3_E_x_11, s3_E_x_12);
const float s3_E_x_14 = __fmaf_rn(s3_E_x_0, s3_E_x_10, s3_E_x_13);
next_e_x = s3_E_x_14;
next_e_x=env.Finish(3,3,z,y,x,next_e_x);
}
old_e_x=captured_old;
current_e_x=next_e_x;
history_Ex_3=next_e_x;
}
{
const float captured_old=history_Ey_3;
float next_e_y=0.f;
if(owned_xy&&env.Valid(4,z,y,x)) {
const float s3_E_y_0 = env.Coefficient(1,4,z,y,x);
const float s3_E_y_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(current_h_x):(env.Read(0,3,z+(0),y+(0),x+(0))));
const float s3_E_y_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_h_x):(env.Read(0,3,z+(-1),y+(0),x+(0))));
const float s3_E_y_3 = __fsub_rn(s3_E_y_1, s3_E_y_2);
const float s3_E_y_4 = env.inv_spacing;
const float s3_E_y_5 = __fmul_rn(s3_E_y_3, s3_E_y_4);
const float s3_E_y_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(current_h_z):(env.Read(2,3,z+(0),y+(0),x+(0))));
const float s3_E_y_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-2)>=0&&qx+(-2)<29)?(shuffle_3_1_0):(env.Read(2,3,z+(0),y+(0),x+(-1))));
const float s3_E_y_8 = __fsub_rn(s3_E_y_6, s3_E_y_7);
const float s3_E_y_9 = __fmul_rn(s3_E_y_8, s3_E_y_4);
const float s3_E_y_10 = __fsub_rn(s3_E_y_5, s3_E_y_9);
const float s3_E_y_11 = env.Coefficient(0,4,z,y,x);
const float s3_E_y_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_y):(env.Read(4,2,z+(0),y+(0),x+(0))));
const float s3_E_y_13 = __fmul_rn(s3_E_y_11, s3_E_y_12);
const float s3_E_y_14 = __fmaf_rn(s3_E_y_0, s3_E_y_10, s3_E_y_13);
next_e_y = s3_E_y_14;
next_e_y=env.Finish(4,3,z,y,x,next_e_y);
}
old_e_y=captured_old;
current_e_y=next_e_y;
history_Ey_3=next_e_y;
}
{
const float captured_old=history_Ez_3;
float next_e_z=0.f;
if(owned_xy&&env.Valid(5,z,y,x)) {
const float s3_E_z_0 = env.Coefficient(1,5,z,y,x);
const float s3_E_z_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(current_h_y):(env.Read(1,3,z+(0),y+(0),x+(0))));
const float s3_E_z_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-2)>=0&&qx+(-2)<29)?(shuffle_3_1_1):(env.Read(1,3,z+(0),y+(0),x+(-1))));
const float s3_E_z_3 = __fsub_rn(s3_E_z_1, s3_E_z_2);
const float s3_E_z_4 = env.inv_spacing;
const float s3_E_z_5 = __fmul_rn(s3_E_z_3, s3_E_z_4);
const float s3_E_z_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(current_h_x):(env.Read(0,3,z+(0),y+(0),x+(0))));
const float s3_E_z_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-1)>=0&&qx+(-1)<29)?(exchange[0+q+(-1)*32+(0)]):(env.Read(0,3,z+(0),y+(-1),x+(0))));
const float s3_E_z_8 = __fsub_rn(s3_E_z_6, s3_E_z_7);
const float s3_E_z_9 = __fmul_rn(s3_E_z_8, s3_E_z_4);
const float s3_E_z_10 = __fsub_rn(s3_E_z_5, s3_E_z_9);
const float s3_E_z_11 = env.Coefficient(0,5,z,y,x);
const float s3_E_z_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-2)>=0&&qy+(-2)<13&&qx+(-2)>=0&&qx+(-2)<29)?(old_e_z):(env.Read(5,2,z+(0),y+(0),x+(0))));
const float s3_E_z_13 = __fmul_rn(s3_E_z_11, s3_E_z_12);
const float s3_E_z_14 = __fmaf_rn(s3_E_z_0, s3_E_z_10, s3_E_z_13);
next_e_z = s3_E_z_14;
next_e_z=env.Finish(5,3,z,y,x,next_e_z);
}
old_e_z=captured_old;
current_e_z=next_e_z;
history_Ez_3=next_e_z;
}
if(owned_xy) {
env.Publish(0,3,z,y,x,current_h_x);
env.Publish(1,3,z,y,x,current_h_y);
env.Publish(2,3,z,y,x,current_h_z);
env.Publish(3,3,z,y,x,current_e_x);
env.Publish(4,3,z,y,x,current_e_y);
env.Publish(5,3,z,y,x,current_e_z);
}
}
}
{
const bool owned_xy=qx>=0&&qx<29&&qy>=0&&qy<13;
exchange[1024+q]=old_e_x;
exchange[1536+q]=old_e_z;
__syncthreads(); // A: publish previous-E exchange; retire previous E's H readers.
{
const int z=oz+iz-4;
const int y=oy-4+qy,x=ox-4+qx;
const float shuffle_4_0_0=__shfl_sync(0xffffffffu,old_e_z,(qx+(1))&31);
const float shuffle_4_0_1=__shfl_sync(0xffffffffu,old_e_y,(qx+(1))&31);
{
const float captured_old=history_Hx_4;
float next_h_x=0.f;
if(owned_xy&&env.Valid(0,z,y,x)) {
const float s4_H_x_0 = env.Coefficient(1,0,z,y,x);
const float s4_H_x_1 = (-s4_H_x_0);
const float s4_H_x_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(-1)>=0&&qx+(-1)<29)?(exchange[1536+q+(1)*32+(0)]):(env.Read(5,3,z+(0),y+(1),x+(0))));
const float s4_H_x_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_z):(env.Read(5,3,z+(0),y+(0),x+(0))));
const float s4_H_x_4 = __fsub_rn(s4_H_x_2, s4_H_x_3);
const float s4_H_x_5 = env.inv_spacing;
const float s4_H_x_6 = __fmul_rn(s4_H_x_4, s4_H_x_5);
const float s4_H_x_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(current_e_y):(env.Read(4,3,z+(1),y+(0),x+(0))));
const float s4_H_x_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_y):(env.Read(4,3,z+(0),y+(0),x+(0))));
const float s4_H_x_9 = __fsub_rn(s4_H_x_7, s4_H_x_8);
const float s4_H_x_10 = __fmul_rn(s4_H_x_9, s4_H_x_5);
const float s4_H_x_11 = __fsub_rn(s4_H_x_6, s4_H_x_10);
const float s4_H_x_12 = env.Coefficient(0,0,z,y,x);
const float s4_H_x_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_h_x):(env.Read(0,3,z+(0),y+(0),x+(0))));
const float s4_H_x_14 = __fmul_rn(s4_H_x_12, s4_H_x_13);
const float s4_H_x_15 = __fmaf_rn(s4_H_x_1, s4_H_x_11, s4_H_x_14);
next_h_x = s4_H_x_15;
next_h_x=env.Finish(0,4,z,y,x,next_h_x);
}
old_h_x=captured_old;
current_h_x=next_h_x;
history_Hx_4=next_h_x;
}
{
const float captured_old=history_Hy_4;
float next_h_y=0.f;
if(owned_xy&&env.Valid(1,z,y,x)) {
const float s4_H_y_0 = env.Coefficient(1,1,z,y,x);
const float s4_H_y_1 = (-s4_H_y_0);
const float s4_H_y_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(current_e_x):(env.Read(3,3,z+(1),y+(0),x+(0))));
const float s4_H_y_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_x):(env.Read(3,3,z+(0),y+(0),x+(0))));
const float s4_H_y_4 = __fsub_rn(s4_H_y_2, s4_H_y_3);
const float s4_H_y_5 = env.inv_spacing;
const float s4_H_y_6 = __fmul_rn(s4_H_y_4, s4_H_y_5);
const float s4_H_y_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(0)>=0&&qx+(0)<29)?(shuffle_4_0_0):(env.Read(5,3,z+(0),y+(0),x+(1))));
const float s4_H_y_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_z):(env.Read(5,3,z+(0),y+(0),x+(0))));
const float s4_H_y_9 = __fsub_rn(s4_H_y_7, s4_H_y_8);
const float s4_H_y_10 = __fmul_rn(s4_H_y_9, s4_H_y_5);
const float s4_H_y_11 = __fsub_rn(s4_H_y_6, s4_H_y_10);
const float s4_H_y_12 = env.Coefficient(0,1,z,y,x);
const float s4_H_y_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_h_y):(env.Read(1,3,z+(0),y+(0),x+(0))));
const float s4_H_y_14 = __fmul_rn(s4_H_y_12, s4_H_y_13);
const float s4_H_y_15 = __fmaf_rn(s4_H_y_1, s4_H_y_11, s4_H_y_14);
next_h_y = s4_H_y_15;
next_h_y=env.Finish(1,4,z,y,x,next_h_y);
}
old_h_y=captured_old;
current_h_y=next_h_y;
history_Hy_4=next_h_y;
}
{
const float captured_old=0.f;
float next_h_z=0.f;
if(owned_xy&&env.Valid(2,z,y,x)) {
const float s4_H_z_0 = env.Coefficient(1,2,z,y,x);
const float s4_H_z_1 = (-s4_H_z_0);
const float s4_H_z_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(0)>=0&&qx+(0)<29)?(shuffle_4_0_1):(env.Read(4,3,z+(0),y+(0),x+(1))));
const float s4_H_z_3 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_y):(env.Read(4,3,z+(0),y+(0),x+(0))));
const float s4_H_z_4 = __fsub_rn(s4_H_z_2, s4_H_z_3);
const float s4_H_z_5 = env.inv_spacing;
const float s4_H_z_6 = __fmul_rn(s4_H_z_4, s4_H_z_5);
const float s4_H_z_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(-1)>=0&&qx+(-1)<29)?(exchange[1024+q+(1)*32+(0)]):(env.Read(3,3,z+(0),y+(1),x+(0))));
const float s4_H_z_8 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_x):(env.Read(3,3,z+(0),y+(0),x+(0))));
const float s4_H_z_9 = __fsub_rn(s4_H_z_7, s4_H_z_8);
const float s4_H_z_10 = __fmul_rn(s4_H_z_9, s4_H_z_5);
const float s4_H_z_11 = __fsub_rn(s4_H_z_6, s4_H_z_10);
const float s4_H_z_12 = env.Coefficient(0,2,z,y,x);
const float s4_H_z_13 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_h_z):(env.Read(2,3,z+(0),y+(0),x+(0))));
const float s4_H_z_14 = __fmul_rn(s4_H_z_12, s4_H_z_13);
const float s4_H_z_15 = __fmaf_rn(s4_H_z_1, s4_H_z_11, s4_H_z_14);
next_h_z = s4_H_z_15;
next_h_z=env.Finish(2,4,z,y,x,next_h_z);
}
old_h_z=captured_old;
current_h_z=next_h_z;
}
}
exchange[0+q]=current_h_x;
exchange[512+q]=current_h_z;
__syncthreads(); // B: publish H exchange; retire H's previous-E readers.
{
const int z=oz+iz-4;
const int y=oy-4+qy,x=ox-4+qx;
const float shuffle_4_1_0=__shfl_sync(0xffffffffu,current_h_z,(qx+(-1))&31);
const float shuffle_4_1_1=__shfl_sync(0xffffffffu,current_h_y,(qx+(-1))&31);
{
const float captured_old=0.f;
float next_e_x=0.f;
if(owned_xy&&env.Valid(3,z,y,x)) {
const float s4_E_x_0 = env.Coefficient(1,3,z,y,x);
const float s4_E_x_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(0)>=0&&qx+(0)<29)?(current_h_z):(env.Read(2,4,z+(0),y+(0),x+(0))));
const float s4_E_x_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(0)>=0&&qx+(0)<29)?(exchange[512+q+(-1)*32+(0)]):(env.Read(2,4,z+(0),y+(-1),x+(0))));
const float s4_E_x_3 = __fsub_rn(s4_E_x_1, s4_E_x_2);
const float s4_E_x_4 = env.inv_spacing;
const float s4_E_x_5 = __fmul_rn(s4_E_x_3, s4_E_x_4);
const float s4_E_x_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(0)>=0&&qx+(0)<29)?(current_h_y):(env.Read(1,4,z+(0),y+(0),x+(0))));
const float s4_E_x_7 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(0)>=0&&qx+(0)<29)?(old_h_y):(env.Read(1,4,z+(-1),y+(0),x+(0))));
const float s4_E_x_8 = __fsub_rn(s4_E_x_6, s4_E_x_7);
const float s4_E_x_9 = __fmul_rn(s4_E_x_8, s4_E_x_4);
const float s4_E_x_10 = __fsub_rn(s4_E_x_5, s4_E_x_9);
const float s4_E_x_11 = env.Coefficient(0,3,z,y,x);
const float s4_E_x_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_x):(env.Read(3,3,z+(0),y+(0),x+(0))));
const float s4_E_x_13 = __fmul_rn(s4_E_x_11, s4_E_x_12);
const float s4_E_x_14 = __fmaf_rn(s4_E_x_0, s4_E_x_10, s4_E_x_13);
next_e_x = s4_E_x_14;
next_e_x=env.Finish(3,4,z,y,x,next_e_x);
}
old_e_x=captured_old;
current_e_x=next_e_x;
}
{
const float captured_old=0.f;
float next_e_y=0.f;
if(owned_xy&&env.Valid(4,z,y,x)) {
const float s4_E_y_0 = env.Coefficient(1,4,z,y,x);
const float s4_E_y_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(0)>=0&&qx+(0)<29)?(current_h_x):(env.Read(0,4,z+(0),y+(0),x+(0))));
const float s4_E_y_2 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(0)>=0&&qx+(0)<29)?(old_h_x):(env.Read(0,4,z+(-1),y+(0),x+(0))));
const float s4_E_y_3 = __fsub_rn(s4_E_y_1, s4_E_y_2);
const float s4_E_y_4 = env.inv_spacing;
const float s4_E_y_5 = __fmul_rn(s4_E_y_3, s4_E_y_4);
const float s4_E_y_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(0)>=0&&qx+(0)<29)?(current_h_z):(env.Read(2,4,z+(0),y+(0),x+(0))));
const float s4_E_y_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(-1)>=0&&qx+(-1)<29)?(shuffle_4_1_0):(env.Read(2,4,z+(0),y+(0),x+(-1))));
const float s4_E_y_8 = __fsub_rn(s4_E_y_6, s4_E_y_7);
const float s4_E_y_9 = __fmul_rn(s4_E_y_8, s4_E_y_4);
const float s4_E_y_10 = __fsub_rn(s4_E_y_5, s4_E_y_9);
const float s4_E_y_11 = env.Coefficient(0,4,z,y,x);
const float s4_E_y_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_y):(env.Read(4,3,z+(0),y+(0),x+(0))));
const float s4_E_y_13 = __fmul_rn(s4_E_y_11, s4_E_y_12);
const float s4_E_y_14 = __fmaf_rn(s4_E_y_0, s4_E_y_10, s4_E_y_13);
next_e_y = s4_E_y_14;
next_e_y=env.Finish(4,4,z,y,x,next_e_y);
}
old_e_y=captured_old;
current_e_y=next_e_y;
}
{
const float captured_old=0.f;
float next_e_z=0.f;
if(owned_xy&&env.Valid(5,z,y,x)) {
const float s4_E_z_0 = env.Coefficient(1,5,z,y,x);
const float s4_E_z_1 = ((iz+(0)>=0&&iz+(0)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(0)>=0&&qx+(0)<29)?(current_h_y):(env.Read(1,4,z+(0),y+(0),x+(0))));
const float s4_E_z_2 = ((iz+(0)>=0&&iz+(0)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(-1)>=0&&qx+(-1)<29)?(shuffle_4_1_1):(env.Read(1,4,z+(0),y+(0),x+(-1))));
const float s4_E_z_3 = __fsub_rn(s4_E_z_1, s4_E_z_2);
const float s4_E_z_4 = env.inv_spacing;
const float s4_E_z_5 = __fmul_rn(s4_E_z_3, s4_E_z_4);
const float s4_E_z_6 = ((iz+(0)>=0&&iz+(0)<64&&qy+(0)>=0&&qy+(0)<13&&qx+(0)>=0&&qx+(0)<29)?(current_h_x):(env.Read(0,4,z+(0),y+(0),x+(0))));
const float s4_E_z_7 = ((iz+(0)>=0&&iz+(0)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(0)>=0&&qx+(0)<29)?(exchange[0+q+(-1)*32+(0)]):(env.Read(0,4,z+(0),y+(-1),x+(0))));
const float s4_E_z_8 = __fsub_rn(s4_E_z_6, s4_E_z_7);
const float s4_E_z_9 = __fmul_rn(s4_E_z_8, s4_E_z_4);
const float s4_E_z_10 = __fsub_rn(s4_E_z_5, s4_E_z_9);
const float s4_E_z_11 = env.Coefficient(0,5,z,y,x);
const float s4_E_z_12 = ((iz+(-1)>=0&&iz+(-1)<64&&qy+(-1)>=0&&qy+(-1)<13&&qx+(-1)>=0&&qx+(-1)<29)?(old_e_z):(env.Read(5,3,z+(0),y+(0),x+(0))));
const float s4_E_z_13 = __fmul_rn(s4_E_z_11, s4_E_z_12);
const float s4_E_z_14 = __fmaf_rn(s4_E_z_0, s4_E_z_10, s4_E_z_13);
next_e_z = s4_E_z_14;
next_e_z=env.Finish(5,4,z,y,x,next_e_z);
}
old_e_z=captured_old;
current_e_z=next_e_z;
}
if(owned_xy) {
env.Publish(0,4,z,y,x,current_h_x);
env.Publish(1,4,z,y,x,current_h_y);
env.Publish(2,4,z,y,x,current_h_z);
env.Publish(3,4,z,y,x,current_e_x);
env.Publish(4,4,z,y,x,current_e_y);
env.Publish(5,4,z,y,x,current_e_z);
}
}
}
}
}
} } } // namespace beamz::cuda::streamed_ir
