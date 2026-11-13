// Optional CUPTI activity collector for research measurements, never linked into BeamZ.
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mutex>

namespace {
FILE* output = nullptr;
std::mutex mutex;
uint64_t dropped = 0;
const char* error_text = nullptr;
int Check(CUptiResult result) {
  if (result == CUPTI_SUCCESS) return 0;
  cuptiGetResultString(result, &error_text);
  return static_cast<int>(result);
}
void CUPTIAPI Request(uint8_t** buffer, size_t* size, size_t* records) {
  *size = 8 * 1024 * 1024;
  *buffer = static_cast<uint8_t*>(std::malloc(*size));
  *records = 0;
}
void CUPTIAPI Complete(CUcontext context, uint32_t stream, uint8_t* buffer,
                       size_t, size_t valid) {
  std::lock_guard<std::mutex> lock(mutex);
  CUpti_Activity* record = nullptr;
  CUptiResult status;
  while ((status = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
    if (!output) continue;
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
        record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      const auto* k = reinterpret_cast<const CUpti_ActivityKernel12*>(record);
      std::fprintf(output,"{\"kind\":\"kernel\",\"name\":\"%s\",\"start_ns\":%llu,\"end_ns\":%llu,\"grid\":[%d,%d,%d],\"block\":[%d,%d,%d],\"registers\":%u,\"dynamic_shared_bytes\":%u,\"graph_id\":%u,\"graph_node_id\":%llu,\"shared_memory_executed_bytes\":%u,\"local_bytes_per_thread\":%u}\n", k->name,
        static_cast<unsigned long long>(k->start),static_cast<unsigned long long>(k->end),
        k->gridX,k->gridY,k->gridZ,k->blockX,k->blockY,k->blockZ,
        k->registersPerThread,k->dynamicSharedMemory,k->graphId,
        static_cast<unsigned long long>(k->graphNodeId),k->sharedMemoryExecuted,k->localMemoryPerThread);
    }
  }
  size_t count=0;
  if (cuptiActivityGetNumDroppedRecords(context,stream,&count)==CUPTI_SUCCESS) dropped+=count;
  std::free(buffer);
}
}
extern "C" const char* beamz_profile_error() { return error_text; }
extern "C" int beamz_profile_start(const char* path) {
  if (output) return -1;
  output=std::fopen(path,"w");
  if (!output) return -2;
  dropped=0;
  int error=Check(cuptiActivityRegisterCallbacks(Request,Complete));
  if (!error) error=Check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  return error;
}
extern "C" int beamz_profile_stop() {
  int error=Check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  const int flushed=Check(cuptiActivityFlushAll(0));
  std::lock_guard<std::mutex> lock(mutex);
  if (output) { std::fprintf(output,"{\"kind\":\"collector\",\"dropped_records\":%llu}\n",static_cast<unsigned long long>(dropped)); std::fclose(output);output=nullptr; }
  return error ? error : flushed;
}
