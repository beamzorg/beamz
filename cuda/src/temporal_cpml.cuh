// Included by update.cu. Independent overlapping two-step tiles. Every tile
// reads frozen fields/psi and writes only its rectangular ownership region.
struct PairCpmlPhase {
  FusedYeePhase field;
  void *psi[6], *final_psi[6];
  int psi_row[6], psi_plane[6];
  const float *profiles[18];
  int thickness, metallic_edges;
};

PairCpmlPhase MakePairCpmlPhase(const BeamzLaunch &launch) {
  PairCpmlPhase p{};
  p.field = MakeFusedYeePhase(launch);
  p.thickness = launch.uniform_cpml_thickness;
  p.metallic_edges = launch.metallic_edges;
  for (int t = 0; t < 6; ++t) {
    p.psi[t] = launch.inputs[31 + t].data;
    p.final_psi[t] = launch.outputs[3 + t].data;
    p.psi_row[t] = launch.inputs[31 + t].row_stride;
    p.psi_plane[t] = launch.inputs[31 + t].plane_stride;
    for (int q = 0; q < 3; ++q)
      p.profiles[3 * t + q] =
          static_cast<const float *>(launch.inputs[13 + 3 * t + q].data);
  }
  return p;
}

__host__ __device__ constexpr int PairPsiTerms(int mask) {
  return 2 * ((mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1));
}
__host__ __device__ constexpr int PairPsiSlot(int mask, int term) {
  int slot=0;
  for(int t=0;t<term;++t)
    slot += bool(mask & (1 << beamz::cuda::yee::CpmlAxis(t)));
  return slot;
}
struct PairCpmlGrid {
  int high[3], field_high[3], extent[3], boundary[3];
};

template <int X, int Y, int K, int Mask, bool Half> struct PairPsiStage {
  static constexpr int sx = X + 3 - K, sy = Y + 3 - K;
  static constexpr int offset = PairStage<X, Y, 3>::offset +
                                (K == 0 ? 0 : (Half ? 1 : 2) * PairPsiTerms(Mask) * (X + 3) * (Y + 3));
  using Value=std::conditional_t<Half,__nv_bfloat16,float>;
  __device__ static __forceinline__ Value &At(float *shared, int term, int z,
                                              int y, int x) {
    return reinterpret_cast<Value *>(shared+offset)[((PairPsiSlot(Mask,term)*2+(z&1))*sy+y)*sx+x];
  }
};

template <int X, int Y, int K, int Term, int Mask, bool Half, int Steps>
__device__ __forceinline__ float PairCpmlDerivative(
    float derivative, float *shared, const PairCpmlPhase &p, int c, int z, int y,
    int x, int iy, int ix, bool owned) {
  constexpr int axis = beamz::cuda::yee::CpmlAxis(Term);
  constexpr float sign = beamz::cuda::yee::CpmlSign(Term);
  if constexpr (!(Mask & (1 << axis))) return sign * derivative;
  const auto &out = p.field.outputs[c];
  const int coord = axis == 0 ? z : axis == 1 ? y : x;
  int packed;
  if (!beamz::cuda::yee::CpmlPackedCoordinate(
          coord, out.dims[axis], p.thickness, p.thickness, &packed))
    return sign * derivative;
  const int pz = axis == 0 ? packed : z;
  const int py = axis == 1 ? packed : y;
  const int px = axis == 2 ? packed : x;
  const auto &input = p.psi[Term];
  const int index = pz*p.psi_plane[Term] + py*p.psi_row[Term] + px;
  float old;
  if constexpr (K < 2) {
    if constexpr(Half) old=__bfloat162float(static_cast<const __nv_bfloat16 *>(input)[index]);
    else old=static_cast<const float *>(input)[index];
  } else {
    const auto value=PairPsiStage<X,Y,K-2,Mask,Half>::At(shared,Term,z,iy+1,ix+1);
    if constexpr(Half) old=__bfloat162float(value); else old=value;
  }
  const float next = beamz::cuda::yee::AdvanceCpmlPsi(
      p.profiles[3 * Term + 1][packed], old,
      p.profiles[3 * Term][packed], derivative);
  if constexpr (K < 2*(Steps-1)) {
    auto &value=PairPsiStage<X,Y,K,Mask,Half>::At(shared,Term,z,iy,ix);
    if constexpr(Half) value=__float2bfloat16_rn(next); else value=next;
  } else if (owned) {
    if constexpr(Half) static_cast<__nv_bfloat16 *>(p.final_psi[Term])[index]=__float2bfloat16_rn(next);
    else static_cast<float *>(p.final_psi[Term])[index]=next;
  }
  return beamz::cuda::yee::CorrectCpmlDerivative(
      sign, derivative, p.profiles[3 * Term + 2][packed], next);
}

template <bool Packed, bool Half, int X, int Y, int Z, int Mask, int Steps, int K = 0>
__device__ __forceinline__ void CpmlPairStages(
    float *shared, const PairCpmlPhase &h, const PairCpmlPhase &e,
    const PairFieldOutputs &final, const FusedHSources &hs,
    const FusedHSources &es, const BeamzBuffer &publication,
    const bool *active_sources, int ox, int oy, int oz, int wave, int end_x, int end_y, int end_z) {
  using Here = PairStage<X+2*(Steps-2),Y+2*(Steps-2),K>;
  constexpr int phase = K % 2, shift = phase == 1 ? 1 : 0;
  constexpr int sign = phase == 0 ? 1 : -1;
  const auto &p = phase == 0 ? h : e;
  const auto &f = p.field;
  const auto &sources = phase == 0 ? hs : es;
  const int z = oz + wave - Steps - K / 2;
  if (wave >= K) {
    for (int i = threadIdx.x; i < Here::sx * Here::sy; i += blockDim.x) {
      const int ix = i % Here::sx, iy = i / Here::sx;
      const int x = ox - Steps + (K + 1) / 2 + ix;
      const int y = oy - Steps + (K + 1) / 2 + iy;
      const bool owned = x >= ox && y >= oy && z >= oz && x < end_x &&
                         y < end_y && z < end_z;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float next = 0.f;
        const auto &out = f.outputs[c];
        if (BufferContains(out, z, y, x)) {
          float old;
          if constexpr (K < 2)
            old = Read3D(f.inputs[c], z, y, x);
          else
            old = PairStage<X, Y, K - 2>::At(shared, c, z, iy + 1, ix + 1);
          auto other = [&](int component, int dz, int dy, int dx) {
            if constexpr (K == 0)
              return Read3D(h.field.inputs[3 + component], z + dz, y + dy, x + dx);
            else
              return PairStage<X+2*(Steps-2),Y+2*(Steps-2),K-1>::At(
                  shared, component, z + dz, iy + shift + dy, ix + shift + dx);
          };
          auto diff = [&](int component, int axis) {
            const int coord = axis == 0 ? z : axis == 1 ? y : x;
            const int extent = f.inputs[3 + component].dims[axis];
            if constexpr (phase == 1) {
              if (coord == 0)
                return p.metallic_edges & (1 << (2 * axis))
                           ? beamz::cuda::yee::ScaleYeeDifference(other(component, 0, 0, 0), f.inv_resolution) : 0.f;
              if (coord == extent)
                return p.metallic_edges & (1 << (2 * axis + 1))
                           ? beamz::cuda::yee::ScaleYeeDifference(
                                 -other(component, axis == 0 ? -1 : 0,
                                        axis == 1 ? -1 : 0, axis == 2 ? -1 : 0),
                                 f.inv_resolution) : 0.f;
            }
            const float center = other(component, 0, 0, 0);
            const float adjacent = other(component, axis == 0 ? sign : 0,
                                          axis == 1 ? sign : 0, axis == 2 ? sign : 0);
            return beamz::cuda::yee::ScaleYeeDifference(
                phase == 0 ? adjacent - center : center - adjacent,
                f.inv_resolution);
          };
          constexpr int sa[] = {2, 0, 1}, sb[] = {1, 2, 0};
          constexpr int aa[] = {1, 0, 2}, ab[] = {0, 2, 1};
          const float da = diff(sa[c], aa[c]), db = diff(sb[c], ab[c]);
          float curl;
          // Unrolled c gives constant term IDs without device-side metadata.
          if (c == 0)
            curl = PairCpmlDerivative<X,Y,K,0,Mask,Half,Steps>(da,shared,p,c,z,y,x,iy,ix,owned) +
                   PairCpmlDerivative<X,Y,K,1,Mask,Half,Steps>(db,shared,p,c,z,y,x,iy,ix,owned);
          else if (c == 1)
            curl = PairCpmlDerivative<X,Y,K,2,Mask,Half,Steps>(da,shared,p,c,z,y,x,iy,ix,owned) +
                   PairCpmlDerivative<X,Y,K,3,Mask,Half,Steps>(db,shared,p,c,z,y,x,iy,ix,owned);
          else
            curl = PairCpmlDerivative<X,Y,K,4,Mask,Half,Steps>(da,shared,p,c,z,y,x,iy,ix,owned) +
                   PairCpmlDerivative<X,Y,K,5,Mask,Half,Steps>(db,shared,p,c,z,y,x,iy,ix,owned);
          const int logical = (z * static_cast<int>(out.dims[1]) + y) *
                                  static_cast<int>(out.dims[2]) + x;
          const float decay = phase == 1 && Packed ? 1.f : Read(f.inputs[6+c],z,y,x);
          const float coefficient = phase == 1 && Packed
              ? beamz::cuda::yee::PackedMaterialSource(f.inputs[6+c],f.inputs[9+c],logical)
              : Read(f.inputs[9+c],z,y,x);
          next = beamz::cuda::yee::AdvanceYeeField(phase,old,decay,coefficient,curl);
          if (p.metallic_edges && beamz::cuda::yee::PecConstrained(out,phase,c,p.metallic_edges,z,y,x))
            next = 0.f;
          if (active_sources[phase*3+c])
            next = c == 0 ? AddFusedHSource<0>(next,sources,z,y,x,K/2)
                 : c == 1 ? AddFusedHSource<1>(next,sources,z,y,x,K/2)
                          : AddFusedHSource<2>(next,sources,z,y,x,K/2);
          if (owned) {
            if constexpr (K >= 2*(Steps-1)) {
              const auto &dest = final.fields[phase*3+c];
              static_cast<float *>(dest.data)[BeamzOffset3D(dest,z,y,x)] = next;
            } else if (publication.data &&
                       static_cast<const int *>(publication.data)[
                           BeamzOffset3D(publication,z,y/8,x/16)]) {
              static_cast<float *>(out.data)[BeamzOffset3D(out,z,y,x)] = next;
            }
          }
        }
        if constexpr (K < 2*Steps-1) Here::At(shared,c,z,iy,ix) = next;
      }
    }
  }
  __syncthreads();
  if constexpr (K < 2*Steps-1)
    CpmlPairStages<Packed,Half,X,Y,Z,Mask,Steps,K+1>(shared,h,e,final,hs,es,publication,
                                     active_sources,ox,oy,oz,wave,end_x,end_y,end_z);
}

// The oriented y-face tile needs three resident 384-thread blocks on SM86.
// Bound registers so its 32-KiB shared stages, rather than registers, set residency.
template <bool Packed, bool Half, int X, int Y, int Z, int Mask, int Steps>
__global__ __launch_bounds__(
    (Steps == 2 && ((Mask == 4 && X == 15 && Y == 16) ||
                    (Mask == 2 && X == 16 && Y == 15))) ? 384 : 256,
    (Steps == 2 && Mask == 2 && X == 16 && Y == 15) ? 3 : 1)
void TemporalCpmlPair(
    const __grid_constant__ PairCpmlPhase h,
    const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,
    const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,
    const __grid_constant__ BeamzBuffer publication, PairCpmlGrid grid) {
  extern __shared__ float shared[];
  auto tile_coordinate = [&](int axis, int local, int size) {
    const int low_count=(grid.high[axis]+size-1)/size;
    if (Mask & (1 << axis))
      return local < low_count ? local * size
          : grid.boundary[axis] + (local-low_count)*size;
    return grid.high[axis] + local * size;
  };
  auto tile_end = [&](int axis, int local, int origin, int size) {
    const int low_count=(grid.high[axis]+size-1)/size;
    if (Mask & (1 << axis))
      return min(origin+size,local<low_count?grid.high[axis]:grid.extent[axis]);
    return min(origin+size,grid.boundary[axis]);
  };
  const int ox=tile_coordinate(2,blockIdx.x,X);
  const int oy=tile_coordinate(1,blockIdx.y,Y);
  const int oz=tile_coordinate(0,blockIdx.z,Z);
  const int end_x=tile_end(2,blockIdx.x,ox,X);
  const int end_y=tile_end(1,blockIdx.y,oy,Y);
  const int end_z=tile_end(0,blockIdx.z,oz,Z);
  __shared__ bool active_sources[6];
  if (threadIdx.x < 6) {
    const int c=threadIdx.x;
    const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);
    bool active=false;
    for(int s=0;s<g.coefficients.dims[0];++s)
      active |= starts[3*s]<oz+Z+2 && starts[3*s]+g.coefficients.dims[1]>oz-2 &&
                starts[3*s+1]<oy+Y+2 && starts[3*s+1]+g.coefficients.dims[2]>oy-2 &&
                starts[3*s+2]<ox+X+2 && starts[3*s+2]+g.coefficients.dims[3]>ox-2;
    active_sources[c]=active;
  }
  __syncthreads();
  for(int wave=0;wave<end_z-oz+2*Steps-1;++wave) {
    if constexpr (Mask == 0 && Steps == 2)
      PairStages<Packed,X,Y,Z,0,false>(shared,h.field,e.field,final,hs,es,publication,
          active_sources,ox,oy,oz,wave,h.thickness,grid.field_high[0]-h.thickness,grid.field_high[1]-h.thickness,grid.field_high[2]-h.thickness);
    else
      CpmlPairStages<Packed,Half,X,Y,Z,Mask,Steps>(shared,h,e,final,hs,es,publication,
                                      active_sources,ox,oy,oz,wave,end_x,end_y,end_z);
  }
}
