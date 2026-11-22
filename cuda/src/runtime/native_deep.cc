#include "native_deep.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

BeamzNativeDeepPlan BeamzBuildNativeDeepPlan(const BeamzBuffer* fields,
                                            const BeamzBuffer& indices,
                                            const int32_t* host_indices) {
  if (indices.rank != 4 || indices.element_type != kBeamzS32 || indices.dims[1] != 6)
    throw std::invalid_argument("deep observations require packed int32 indices[M,6,P,K]");
  BeamzNativeDeepPlan plan;
  int64_t sizes[6]{};
  int64_t prefix = 0;
  for (int c = 0; c < 6; ++c) {
    if (fields[c].rank != 3) throw std::invalid_argument("deep fields must have rank three");
    sizes[c] = 1;
    for (int axis = 0; axis < 3; ++axis) {
      const int64_t dimension = fields[c].dims[axis];
      if (dimension < 1 || dimension > INT32_MAX || sizes[c] > INT32_MAX / dimension)
        throw std::invalid_argument("deep field offsets exceed int32");
      sizes[c] *= dimension;
      plan.mask_shape[axis] = std::max(plan.mask_shape[axis], static_cast<int>(dimension));
    }
    plan.field_base[c] = prefix;
    prefix += sizes[c];
  }
  int64_t count = 1;
  for (int axis = 0; axis < 4; ++axis) {
    if (indices.dims[axis] < 0 ||
        (indices.dims[axis] && count > INT32_MAX / indices.dims[axis]))
      throw std::invalid_argument("deep observation records exceed int32");
    count *= indices.dims[axis];
  }
  std::vector<int64_t> records(static_cast<size_t>(count), -1);
  const int64_t component_stride = indices.dims[2] * indices.dims[3];
  for (int64_t i = 0; i < count; ++i) {
    const int component = static_cast<int>((i / component_stride) % 6);
    const int native = (component + 3) % 6;
    const int32_t logical = host_indices[i];
    if (logical >= 0 && logical < sizes[native]) {
      records[i] = plan.field_base[native] + logical;
      plan.keys.push_back(records[i]);
    }
  }
  std::sort(plan.keys.begin(), plan.keys.end());
  plan.keys.erase(std::unique(plan.keys.begin(), plan.keys.end()), plan.keys.end());
  if (plan.keys.size() > INT32_MAX)
    throw std::invalid_argument("deep unique observed cells exceed int32");
  plan.remapped_indices.resize(static_cast<size_t>(count), -1);
  for (int64_t i = 0; i < count; ++i)
    if (records[i] >= 0)
      plan.remapped_indices[i] = static_cast<int32_t>(
          std::lower_bound(plan.keys.begin(), plan.keys.end(), records[i]) - plan.keys.begin());
  int64_t mask_cells = 1;
  for (int dimension : plan.mask_shape) {
    if (mask_cells > INT32_MAX / dimension)
      throw std::invalid_argument("deep publication mask offsets exceed int32");
    mask_cells *= dimension;
  }
  plan.publication_words.resize((mask_cells + 3) / 4, 0);
  int component = 0;
  for (const int64_t key : plan.keys) {
    while (component < 5 && key >= plan.field_base[component + 1]) ++component;
    const int64_t local = key - plan.field_base[component];
    const int64_t nx = fields[component].dims[2], ny = fields[component].dims[1];
    const int64_t z = local / (ny * nx), y = (local / nx) % ny, x = local % nx;
    const int64_t flat = (z * plan.mask_shape[1] + y) * plan.mask_shape[2] + x;
    plan.publication_words[flat >> 2] |= (uint32_t{1} << component) << (8 * (flat & 3));
  }
  return plan;
}
