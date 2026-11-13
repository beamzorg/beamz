// Shared selector for resource estimates and actual private NativeSession inputs.
#pragma once
#include "native_moving.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

// ACTUAL_BALANCED_PLAN_BUILDER

struct BeamzBalancedPlanChoice{
 BeamzNativeMovingPlan plan;
 BeamzNativeMovingUniformE certificate;
 bool balanced=false;
};
inline bool BeamzBalancedProgramFact(const BeamzBuffer*fields,const void*const*h_coeff,
 const BeamzSourceGroupLaunch*groups,const int32_t*const*starts){
 if(!fields||!h_coeff||!groups||!starts)return false;
 for(int c=0;c<3;++c)if(fields[40+c].rank!=1||fields[40+c].element_type!=kBeamzS32)return false;
 uint32_t source[3]{};
 for(int c=0;c<3;++c){float decay=0.f,value=0.f;
  for(int slot:{c,c+3}){const auto&f=fields[6+slot];if(f.rank!=0||f.element_type!=kBeamzF32||!h_coeff[slot])return false;}
  std::memcpy(&decay,h_coeff[c],4);std::memcpy(&value,h_coeff[c+3],4);std::memcpy(&source[c],h_coeff[c+3],4);
  if(decay!=1.f||!std::isfinite(value))return false;
 }
 if(source[0]!=source[1]||source[1]!=source[2])return false;
 const uint32_t presence=BeamzNativeMovingSourcePresenceMask(groups,9);if(!presence)return true;
 int minimum[3]={INT32_MAX,INT32_MAX,INT32_MAX};
 for(int c=0;c<6;++c)for(int a=0;a<3;++a){if(fields[c].dims[a]<1||fields[c].dims[a]>INT32_MAX)return false;minimum[a]=std::min(minimum[a],int(fields[c].dims[a]));}
 return BeamzNativeSourcePlanePairFact(groups,9,starts,minimum,presence);
}
inline BeamzBalancedPlanChoice BeamzChooseBalancedMovingPlan(const BeamzBuffer*fields,int depth,
 const std::vector<std::array<int,3>>&points,const BeamzSourceGroupLaunch*groups,const int32_t*const*starts,
 bool partition,const void*const*table_data,const void*const*code_data,const void*const*h_coeff){
 BeamzBalancedPlanChoice choice;
 if(partition&&BeamzBalancedProgramFact(fields,h_coeff,groups,starts)){
  choice.plan=BeamzBuildBalancedMovingPlan(fields,depth,points,groups,starts,12,partition);
  choice.certificate=BeamzCertifyNativeMovingUniformE(fields,fields+37,fields+40,table_data,code_data,choice.plan);
  bool accepted=choice.certificate.supported&&choice.certificate.admitted.size()==choice.plan.tile_flags.size();
  if(accepted)for(size_t t=0;t<choice.plan.tile_flags.size();++t){const int policy=beamz::cuda::moving::Policy(choice.plan.tile_flags[t]);if((policy==1||policy==5)&&!choice.certificate.admitted[t]){accepted=false;break;}}
  if(accepted){choice.balanced=true;return choice;}
 }
 choice.plan=BeamzBuildNativeMovingPlan(fields,depth,points,groups,starts,12,partition);
 choice.certificate=BeamzCertifyNativeMovingUniformE(fields,fields+37,fields+40,table_data,code_data,choice.plan);
 return choice;
}
