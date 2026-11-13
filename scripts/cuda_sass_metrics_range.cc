// Diagnostic software SASS counters. Never used for performance timings.
#include <cuda.h>
#include <cupti.h>
#include <cupti_profiler_target.h>
#include <cupti_sass_metrics.h>
#include <cupti_target.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace {
CUcontext context=nullptr;
int device=0;
std::string output,error;
std::map<uint64_t,std::string> names;
bool enabled=false,configured=false;
int check(CUptiResult status,const char*call){
 if(status==CUPTI_SUCCESS)return 0;
 const char*message=nullptr;cuptiGetResultString(status,&message);
 error=std::string(call)+": "+(message?message:"unknown CUPTI error");return int(status);
}
std::string quote(const char*value){
 std::string out="\"";for(const char*p=value;*p;++p){if(*p=='\\'||*p=='\"')out+='\\';out+=*p;}return out+'\"';
}
}

extern "C" const char*beamz_sass_error(){return error.c_str();}
extern "C" int beamz_sass_begin(const char*path){
 if(enabled||configured){error="Nested software metric range";return -1;}
 output=path;error.clear();names.clear();
 if(cuCtxGetCurrent(&context)!=CUDA_SUCCESS||!context||cuCtxGetDevice(&device)!=CUDA_SUCCESS){error="Current native CUDA context required";return -2;}
 CUpti_Profiler_Initialize_Params initialize{CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
 if(int r=check(cuptiProfilerInitialize(&initialize),"profilerInitialize"))return r;
 CUpti_Device_GetChipName_Params chip{CUpti_Device_GetChipName_Params_STRUCT_SIZE};chip.deviceIndex=device;
 if(int r=check(cuptiDeviceGetChipName(&chip),"deviceGetChipName"))return r;
 std::vector<CUpti_SassMetrics_Config> configs;
 const char*metrics[]={"smsp__sass_inst_executed","smsp__sass_thread_inst_executed_pred_on",
  "smsp__sass_sectors_mem_global","smsp__sass_sectors_mem_global_ideal",
  "smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared","smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal"};
 for(const char*name:metrics){
  CUpti_SassMetrics_GetProperties_Params properties{CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
  properties.pChipName=chip.pChipName;properties.pMetricName=name;
  if(int r=check(cuptiSassMetricsGetProperties(&properties),"sassGetProperties"))return r;
  names[properties.metric.metricId]=name;
  configs.push_back({properties.metric.metricId,CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU});
 }
 CUpti_SassMetricsSetConfig_Params config{CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE};
 config.deviceIndex=device;config.numOfMetricConfig=configs.size();config.pConfigs=configs.data();
 if(int r=check(cuptiSassMetricsSetConfig(&config),"sassSetConfig"))return r;configured=true;
 CUpti_SassMetricsEnable_Params enable{CUpti_SassMetricsEnable_Params_STRUCT_SIZE};enable.ctx=context;enable.enableLazyPatching=1;
 if(int r=check(cuptiSassMetricsEnable(&enable),"sassEnable"))return r;enabled=true;
 return 0;
}

extern "C" int beamz_sass_end(){
 if(!enabled){error="No enabled software metric range";return -3;}
 if(cuCtxSynchronize()!=CUDA_SUCCESS){error="Instrumented kernel synchronization failed";return -4;}
 CUpti_SassMetricsGetDataProperties_Params properties{CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE};properties.ctx=context;
 if(int r=check(cuptiSassMetricsGetDataProperties(&properties),"sassDataProperties"))return r;
 if(!properties.numOfPatchedInstructionRecords||!properties.numOfInstances){error="Empty software instruction range";return -5;}
 std::vector<CUpti_SassMetrics_Data> data(properties.numOfPatchedInstructionRecords);
 std::vector<std::vector<CUpti_SassMetrics_InstanceValue>> instances(data.size(),std::vector<CUpti_SassMetrics_InstanceValue>(properties.numOfInstances));
 for(size_t i=0;i<data.size();++i)data[i].pInstanceValues=instances[i].data();
 CUpti_SassMetricsFlushData_Params flush{CUpti_SassMetricsFlushData_Params_STRUCT_SIZE};
 flush.ctx=context;flush.numOfInstances=properties.numOfInstances;flush.numOfPatchedInstructionRecords=data.size();flush.pMetricsData=data.data();
 if(int r=check(cuptiSassMetricsFlushData(&flush),"sassFlushData"))return r;
 CUpti_SassMetricsDisable_Params disable{CUpti_SassMetricsDisable_Params_STRUCT_SIZE};disable.ctx=context;
 if(int r=check(cuptiSassMetricsDisable(&disable),"sassDisable"))return r;enabled=false;
 CUpti_SassMetricsUnsetConfig_Params unset{CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};unset.deviceIndex=device;
 if(int r=check(cuptiSassMetricsUnsetConfig(&unset),"sassUnsetConfig"))return r;configured=false;
 std::ofstream file(output);file<<std::setprecision(17)<<"{\"schema\":\"beamz.software-SASS-range.v1\",\"dropped_records\":"<<disable.numOfDroppedRecords
  <<",\"records\":"<<data.size()<<",\"instances\":"<<properties.numOfInstances<<",\"rows\":[\n";
 bool first=true;
 for(auto&row:data){
  if(!first)file<<",\n";first=false;
  file<<"{\"function\":"<<quote(row.functionName)<<",\"pc\":"<<row.pcOffset<<",\"cubin_crc\":"<<row.cubinCrc<<",\"metrics\":{";
  bool metric_first=true;
  for(size_t i=0;i<properties.numOfInstances;++i){const auto&v=row.pInstanceValues[i];auto found=names.find(v.metricId);if(found==names.end())continue;
   if(!metric_first)file<<',';metric_first=false;file<<quote(found->second.c_str())<<':'<<v.value;
  }
  file<<"}}";free(const_cast<char*>(row.functionName));
 }
 file<<"\n],\"scope\":\"Software-instrumented instruction/sector/wavefront counts; not timings or DRAM byte measurements\"}\n";
 if(!file){error="Failed to write software instruction data";return -6;}
 if(disable.numOfDroppedRecords){error="Dropped software instruction records";return -7;}
 return 0;
}
