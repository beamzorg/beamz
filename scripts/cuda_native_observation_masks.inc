// Immutable observation-effect facts. Pack six field-presence bits into the
// existing coarse lookup word; raw leaf IDs and published values are unchanged.
inline BeamzNativeMovingSparseObservation BeamzBuildNativeMovingMaskedObservation(
    const BeamzBuffer* fields, const std::vector<int64_t>& keys) {
  auto plan = BeamzBuildNativeMovingSparseObservation(fields, keys);
  constexpr size_t width = 128 * 6;
  for (int32_t& encoded : plan.coarse) {
    if (encoded < 0) continue;
    if (encoded > (INT32_MAX >> 6))
      throw std::invalid_argument("masked observation slot exceeds int32");
    const size_t base = size_t(encoded) * width;
    if (base + width > plan.leaf.size())
      throw std::invalid_argument("masked observation leaf is outside allocation");
    unsigned mask = 0;
    for (size_t cell = 0; cell < 128; ++cell)
      for (unsigned field = 0; field < 6; ++field)
        if (plan.leaf[base + cell * 6 + field] >= 0) mask |= 1u << field;
    encoded = int32_t((unsigned(encoded) << 6) | mask);
  }
  return plan;
}
