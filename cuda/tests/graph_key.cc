#include <cstdio>

#include "graph.h"

int main() {
  BeamzDftGroupLaunch monitors{};
  BeamzProgramLaunch program{};
  program.monitors = &monitors;
  const auto baseline = BeamzGraphKey("test", nullptr, program);
  BeamzBuffer* buffers[] = {
      &monitors.indices, &monitors.weights, &monitors.frequencies,
      &monitors.component_masks, &monitors.counts, &monitors.codes,
      &monitors.windows, &monitors.dft_re, &monitors.dft_im,
      &monitors.dft_weight, &monitors.phase_sin, &monitors.phase_cos,
      &monitors.phase_window, &monitors.pair_samples, &monitors.time,
      &monitors.current_step, &monitors.elapsed_steps};
  int storage = 0;
  for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
    buffers[i]->data = &storage;
    const auto changed = BeamzGraphKey("test", nullptr, program);
    if (changed == baseline) {
      std::fprintf(stderr, "Graph key ignored monitor buffer %zu\n", i);
      return 1;
    }
    // Replay must read the current contents at the bound address; the value
    // itself must not generate a fresh graph key for every timestep.
    storage += 1;
    if (changed != BeamzGraphKey("test", nullptr, program)) return 2;
    buffers[i]->data = nullptr;
  }
  return 0;
}
