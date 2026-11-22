// Capture an exact positive-zero initial-state fact before releasing caller
// storage. Reset then reconstructs those bytes without a private host snapshot.
inline bool BeamzNativeAllPositiveZero(const void* data, size_t bytes) {
  const auto* input = static_cast<const unsigned char*>(data);
  uint64_t combined = 0;
  size_t offset = 0;
  for (; offset + sizeof(uint64_t) <= bytes; offset += sizeof(uint64_t)) {
    uint64_t word;
    std::memcpy(&word, input + offset, sizeof(word));
    combined |= word;
  }
  for (; offset < bytes; ++offset) combined |= input[offset];
  return combined == 0;
}
