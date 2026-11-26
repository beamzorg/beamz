// Contiguous non-peer faces travel in both directions concurrently. Sparse
// packets are lossless and have bounded staging independent of the grid size.
struct TemporalFace {
  cudaMemcpy3DPeerParms region;
  size_t offset;
  size_t Bytes() const {
    const auto e = region.extent;
    return e.width * e.height * e.depth;
  }
  static bool Linear(const TemporalFace& face) {
    const auto& q = face.region;
    return !(q.srcPos.x || q.srcPos.y || q.dstPos.x || q.dstPos.y) && q.srcPtr.pitch == q.extent.width &&
           q.dstPtr.pitch == q.extent.width && q.srcPtr.ysize == q.extent.height &&
           q.dstPtr.ysize == q.extent.height;
  }
  static char* Address(cudaPitchedPtr ptr, size_t z) {
    return static_cast<char*>(ptr.ptr) + z * ptr.pitch * ptr.ysize;
  }
};

struct TemporalPeerEndpoint {
  const int device = CurrentDevice();
  Stream stream;
  Event received;
  struct Scratch {
    DeviceMemory send{beamz::cuda::halo::kBufferBytes}, receive{beamz::cuda::halo::kBufferBytes}, counter{4};
    unsigned* count = nullptr;
    Scratch() {
      Check(cudaMallocHost(reinterpret_cast<void**>(&count), sizeof(*count)), "allocate halo count");
    }
    ~Scratch() { cudaFreeHost(count); }
  };
  std::unique_ptr<Scratch> scratch;
  explicit TemporalPeerEndpoint(bool pack) {
    if (pack) scratch = std::make_unique<Scratch>();
  }
  ~TemporalPeerEndpoint() { cudaSetDevice(device); }

  void Select() const { Check(cudaSetDevice(device), "select halo device"); }
  void Wait(const Event& event) const {
    Select();
    Check(cudaStreamWaitEvent(stream.value, event.value, 0), "wait for halo");
  }
  void Mark() const {
    Select();
    Check(cudaEventRecord(received.value, stream.value), "publish halo");
  }
  void Pack(const TemporalFace& f, size_t offset, size_t bytes) {
    if (!bytes) return;
    Select();
    const auto& q = f.region;
    Check(beamz::cuda::halo::Pack(TemporalFace::Address(q.srcPtr, q.srcPos.z) + offset, bytes / 4,
                                  scratch->send.data, scratch->counter.data, stream.value),
          "pack sparse halo");
    Check(cudaMemcpyAsync(scratch->count, scratch->counter.data, 4, cudaMemcpyDeviceToHost, stream.value),
          "read halo count");
  }
  size_t Publish(const TemporalFace& f, size_t offset, size_t bytes, TemporalPeerEndpoint& dst,
                 bool inspect) const {
    if (!bytes) return 0;
    dst.Select();
    const auto& q = f.region;
    const char* src = TemporalFace::Address(q.srcPtr, q.srcPos.z) + offset;
    char* target = TemporalFace::Address(q.dstPtr, q.dstPos.z) + offset;
    const size_t packed = inspect ? size_t(*scratch->count) * sizeof(beamz::cuda::halo::Packet) : bytes;
    const bool use_packets = inspect && packed + packed / 10 < bytes;
    const size_t wire = use_packets ? packed : bytes;
    if (wire)
      Check(cudaMemcpyPeerAsync(use_packets ? dst.scratch->receive.data : target, dst.device,
                                use_packets ? scratch->send.data : src, device, wire, dst.stream.value),
            "transfer temporal halo");
    if (use_packets)
      Check(beamz::cuda::halo::Restore(dst.scratch->receive.data, *scratch->count, target, bytes / 4,
                                       dst.stream.value),
            "restore sparse halo");
    return wire;
  }
};

struct TemporalPeerTransport {
  std::unique_ptr<TemporalPeerEndpoint> left, right;
  int dense_windows = 0;
  TemporalPeerTransport(int a, int b, bool pack) {
    Check(cudaSetDevice(a), "select lower halo owner");
    left = std::make_unique<TemporalPeerEndpoint>(pack);
    Check(cudaSetDevice(b), "select upper halo owner");
    right = std::make_unique<TemporalPeerEndpoint>(pack);
  }
  void Exchange(const std::vector<TemporalFace>& up, const std::vector<TemporalFace>& down, Event& low,
                Event& high, int join_device, cudaStream_t join) {
    for (auto* p : {left.get(), right.get()}) {
      p->Wait(low);
      p->Wait(high);
    }
    size_t wire = 0, total = 0;
    const bool inspect = left->scratch && dense_windows == 0;
    for (size_t j = 0; j < up.size(); ++j) {
      const size_t a = up[j].Bytes(), b = down[j].Bytes(), limit = std::max(a, b);
      // Dense windows use one ordinary linear DMA per field and direction.
      const size_t chunk = inspect ? beamz::cuda::halo::kChunk : limit;
      for (size_t off = 0; off < limit; off += chunk) {
        const size_t na = std::min(chunk, a - std::min(a, off)), nb = std::min(chunk, b - std::min(b, off));
        if (inspect) {
          left->Pack(up[j], off, na);
          right->Pack(down[j], off, nb);
          for (auto* p : {left.get(), right.get()}) {
            p->Select();
            Check(cudaStreamSynchronize(p->stream.value), "finish halo count");
          }
        }
        wire += left->Publish(up[j], off, na, *right, inspect);
        wire += right->Publish(down[j], off, nb, *left, inspect);
        total += na + nb;
        if (inspect) {
          left->Mark();
          right->Mark();
          // The next chunk can reuse a send buffer after the peer has consumed it.
          left->Wait(right->received);
          right->Wait(left->received);
        }
      }
    }
    if (dense_windows)
      --dense_windows;
    else if (inspect && wire >= total - total / 10)
      dense_windows = 15;
    left->Mark();
    right->Mark();
    Check(cudaSetDevice(join_device), "select temporal join");
    for (auto* p : {left.get(), right.get()})
      Check(cudaStreamWaitEvent(join, p->received.value, 0), "join non-peer halo");
  }
};
