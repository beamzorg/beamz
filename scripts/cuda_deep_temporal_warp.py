"""Generate a warp-X/shared-Y temporal kernel over native BeamZ descriptors.

Research only. The generated header is included after temporal_cpml.cuh in a
translation unit that already defines BeamZ's real launch/field/source helpers.
No replacement material, boundary, source, or auxiliary-state ABI is invented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.cuda_stencil_trace import (
        Expr,
        Read,
        Update,
        emit_cuda_block,
        parameter,
        trace_cpml_step,
    )
except ModuleNotFoundError:
    from cuda_stencil_trace import (
        Expr,
        Read,
        Update,
        emit_cuda_block,
        parameter,
        trace_cpml_step,
    )

TERM_AXES = (1, 0, 0, 2, 2, 1)
TERM_COMPONENTS = (2, 1, 0, 2, 1, 0)


def lower_derivatives(updates):
    """Keep traced recurrence/arithmetic, admit native wall derivative semantics."""

    def visit(value):
        if (
            value.op == "mul_rn"
            and value.args[1].op == "coefficient"
            and value.args[1].args[0].name == "inv_spacing"
        ):
            reads = [arg.args[0] for arg in value.args[0].args]
            shifted = next(read for read in reads if any(read.offset))
            axis = next(i for i, d in enumerate(shifted.offset) if d)
            component = "xyz".index(shifted.field[1])
            term = next(
                t
                for t in range(6)
                if TERM_AXES[t] == axis and TERM_COMPONENTS[t] == component
            )
            return parameter(f"derivative_{term}")
        return Expr(
            value.op, tuple(visit(v) if isinstance(v, Expr) else v for v in value.args)
        )

    return tuple(Update(u.output, visit(u.value)) for u in updates)


def resources(tile, depth, mask, *, psi_history=None, field_frames=1):
    z, y, x = tile
    if depth < 1 or min(tile) < 1 or x + 2 * depth - 1 > 32:
        raise ValueError("The complete initial computation row must fit one warp")
    rows = y + 2 * depth - 1
    if rows * 32 > 1024:
        raise ValueError("More than1024threads required")
    if field_frames not in (1, 2):
        raise ValueError("Only one or two shared field frames are supported")
    areas = [
        (x + 2 * depth - 1 - k) * (y + 2 * depth - 1 - k) for k in range(2 * depth)
    ]
    psi = 2 * int(mask).bit_count()
    shared_psi = psi_history == "shared" or (
        psi_history is None and int(mask).bit_count() >= 2
    )
    field_bytes = field_frames * 2 * 4 * 32 * sum(y + 2 * depth - 1 - k for k in range(2 * depth - 1))
    psi_bytes = (
        4 * psi * 32 * sum(y + 2 * depth - 1 - k for k in range(2 * depth - 2))
        if shared_psi
        else 0
    )
    return dict(
        tile_zyx=tile,
        depth=depth,
        mask=mask,
        threads=rows * 32,
        shared_bytes=field_bytes + psi_bytes,
        shared_field_bytes=field_bytes,
        shared_psi_bytes=psi_bytes,
        psi_history="shared" if shared_psi else "register",
        y_component_history_scalars=2 * depth - 1,
        psi_history_scalars=0 if shared_psi else (2 * depth - 2) * psi,
        psi_current_scalars=2 * psi,
        field_current_history_scalars=12,
        executed_to_useful_half_updates=sum(
            a * (z + 2 * depth - 1 - k) for k, a in enumerate(areas)
        )
        / (2 * depth * z * y * x),
        note="Source-level liveness; ptxas register/spill counts are authoritative",
    )


HELPERS = r"""
#pragma once
#include "deep_temporal_observation.cuh"
// Requires real PairCpmlPhase/FusedYeePhase/PairFieldOutputs/FusedHSources,
// PairCpmlGrid, Read3D/Read/BufferContains and canonical Yee helpers.
namespace beamz { namespace cuda { namespace deep {
__device__ __forceinline__ float SafeRead(const BeamzBuffer &field,int z,int y,int x) {
  return BufferContains(field,z,y,x)?Read3D(field,z,y,x):0.f;
}
__device__ __forceinline__ float Difference(const PairCpmlPhase &p,int phase,int component,int axis,
    int z,int y,int x,float center,float adjacent) {
  const int coord=axis==0?z:axis==1?y:x;
  const int extent=p.field.inputs[3+component].dims[axis];
  if(phase && ((coord==0 && !(p.metallic_edges&(1<<(2*axis)))) ||
               (coord==extent && !(p.metallic_edges&(1<<(2*axis+1))))))return 0.f;
  return beamz::cuda::yee::ScaleYeeDifference(phase?center-adjacent:adjacent-center,p.field.inv_resolution);
}
__device__ __forceinline__ void Publish(const Observation &o, const BeamzBuffer &field,
    int c,int step,int z,int y,int x,float value) {
  if (!o.values) return;
  if (o.publication_words) {
    if(z<0||y<0||x<0||z>=o.mask_nz||y>=o.mask_ny||x>=o.mask_nx)return;
    const int cell=(z*o.mask_ny+y)*o.mask_nx+x;
    const uint32_t word=__ldg(o.publication_words+(cell>>2));
    if(!((word>>(8*(cell&3)))&(1u<<c)))return;
  }
  const int64_t key=o.field_base[c]+(static_cast<int64_t>(z)*field.dims[1]+y)*field.dims[2]+x;
  int low=0,high=o.key_count;
  while(low<high) { const int mid=(low+high)/2;if(o.keys[mid]<key)low=mid+1;else high=mid; }
  if(low<o.key_count && o.keys[low]==key)o.values[step*o.key_count+low]=value;
}
__device__ __forceinline__ int PsiIndex(const PairCpmlPhase &p,int term,int c,int z,int y,int x,int *packed) {
  if(!BufferContains(p.field.outputs[c],z,y,x))return -1;
  const int axis=beamz::cuda::yee::CpmlAxis(term);
  const int coord=axis==0?z:axis==1?y:x;
  if(!beamz::cuda::yee::CpmlPackedCoordinate(coord,p.field.outputs[c].dims[axis],p.thickness,p.thickness,packed))return -1;
  const int pz=axis==0?*packed:z,py=axis==1?*packed:y,px=axis==2?*packed:x;
  return pz*p.psi_plane[term]+py*p.psi_row[term]+px;
}
template<bool Half> __device__ __forceinline__ float ReadPsi(const PairCpmlPhase &p,int term,int z,int y,int x) {
  int packed;const int index=PsiIndex(p,term,term/2,z,y,x,&packed);
  if(index<0)return 0.f;
  if constexpr(Half)return __bfloat162float(static_cast<const __nv_bfloat16 *>(p.psi[term])[index]);
  else return static_cast<const float *>(p.psi[term])[index];
}
__device__ __forceinline__ float Profile(const PairCpmlPhase &p,int term,int kind,int z,int y,int x) {
  int packed;if(PsiIndex(p,term,term/2,z,y,x,&packed)<0)return kind==0?0.f:1.f;
  return p.profiles[3*term+kind][packed];
}
template<bool Half> __device__ __forceinline__ void WritePsi(const PairCpmlPhase &p,int term,int z,int y,int x,float value) {
  int packed;const int index=PsiIndex(p,term,term/2,z,y,x,&packed);if(index<0)return;
  if constexpr(Half)static_cast<__nv_bfloat16 *>(p.final_psi[term])[index]=__float2bfloat16_rn(value);
  else static_cast<float *>(p.final_psi[term])[index]=value;
}
"""


def generate(tile=(64, 8, 25), depth=4, mask=7, *, half=False, psi_history=None,  # noqa: C901
             scalar_h=False, interior=False, byte_packed=False, source_signature=63,
             field_frames=1):  # noqa: C901
    if not isinstance(source_signature, int) or not 0 <= source_signature <= 63:
        raise ValueError("Source signature must contain only the six H/E component bits")
    if interior and mask:
        raise ValueError("Unchecked interior lowering is admitted only for CPML mask zero")
    info = resources(tile, depth, mask, psi_history=psi_history, field_frames=field_frames)
    tz, ty, tx = tile
    stages = 2 * depth
    widths = [tx + stages - 1 - k for k in range(stages)]
    heights = [ty + stages - 1 - k for k in range(stages)]
    lows = [(k + 1) // 2 for k in range(stages)]
    offsets, cursor = [], 0
    for h in heights[:-1]:
        offsets.append(cursor)
        cursor += 2 * 32 * h
    frame_words = cursor
    cursor *= field_frames
    active_terms = [t for t, axis in enumerate(TERM_AXES) if mask & (1 << axis)]
    psi_offsets = []
    for height in heights[:-2]:
        psi_offsets.append(cursor)
        cursor += len(active_terms) * 32 * height
    updates = lower_derivatives(
        trace_cpml_step(
            active_axes=tuple(a for a in range(3) if mask & (1 << a)),
            precision="bfloat16" if half else "float32",
        )
    )

    def active(k):
        return f"qx>={lows[k]} && qx<{lows[k] + widths[k]} && qy>={lows[k]} && qy<{lows[k] + heights[k]}"

    def shared(k, c, dy=0, age=0):
        assert c != 1
        frame = f"((wave+{age})&1)*{frame_words}+" if field_frames == 2 else ""
        return f"s[{frame}{offsets[k]}+({c // 2}*{heights[k]}+qy+({dy})-{lows[k]})*32+qx-{lows[k]}]"

    def prev(k, c):
        assert c == 1
        return f"previous_y_{k}"

    def cur(f, c):
        return f"current_{f}_{c}"

    def psi(k, term):
        if info["psi_history"] == "register":
            return f"previous_psi_{k}_{term}"
        return f"s[{psi_offsets[k]}+({active_terms.index(term)}*{heights[k]}+qy-{lows[k]})*32+qx-{lows[k]}]"

    bodies = []
    for k in range(stages):
        phase = k % 2
        family = "E" if phase else "H"
        selected = tuple(
            u
            for u in updates
            if u.output.field.startswith(family)
            or u.output.field.startswith("psi_" + family)
        )

        def read(read, k=k, phase=phase, family=family):
            dz, dy, dx = read.offset
            if read.field.startswith("psi_"):
                t = int(read.field[5:])
                return (
                    f"ReadPsi<{str(half).lower()}>(p,{t},z,y,x)"
                    if k < 2
                    else psi(k - 2, t)
                )
            c = "xyz".index(read.field[1])
            if read.field.startswith(family):
                if k < 2:
                    return f"{'Read3D' if interior else 'SafeRead'}(p.field.inputs[{c}],z,y,x)"
                return f"old_{family}_{c}"
            if k == 0:
                return f"{'Read3D' if interior else 'SafeRead'}(p.field.inputs[{3 + c}],z+({dz}),y+({dy}),x+({dx}))"
            if not phase:
                if dz:
                    return cur("E", c)
                if dx:
                    return f"neighbor_{c}"
                if dy:
                    return shared(k - 1, c, dy, age=1)
                return f"old_E_{c}"
            if dz:
                return f"old_H_{c}"
            if dx:
                return f"neighbor_{c}"
            if dy:
                return shared(k - 1, c, dy)
            return cur("H", c)

        def coefficient(read, phase=phase):
            name = read.name
            if name == "inv_spacing":
                return "p.field.inv_resolution"
            if name.startswith("source_") or name.startswith("decay_"):
                source = name.startswith("source_")
                c = "xyz".index(name[-1])
                fallback = f"Read(p.field.inputs[{9 + c if source else 6 + c}],z,y,x)"
                if scalar_h and not phase:
                    fallback = f"(*static_cast<const float *>(p.field.inputs[{9 + c if source else 6 + c}].data))"
                value = (
                    f"(Packed ? beamz::cuda::yee::PackedMaterialSource(p.field.inputs[{6 + c}],p.field.inputs[{9 + c}],(z*p.field.outputs[{c}].dims[1]+y)*p.field.outputs[{c}].dims[2]+x) : {fallback})"
                    if source and phase
                    else ("1.f" if phase and not source else fallback)
                )
                if source and phase and byte_packed:
                    linear = f"((z*static_cast<int>(p.field.outputs[{c}].dims[1])+y)*static_cast<int>(p.field.outputs[{c}].dims[2])+x)"
                    typed = f"beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[{6+c}].data),static_cast<int>(p.field.inputs[{6+c}].dims[0]),static_cast<const uint32_t *>(p.field.inputs[{9+c}].data),{linear})"
                    value = f"(Packed ? {typed} : {fallback})"
                if phase and not source:
                    value = f"(Packed ? 1.f : {fallback})"
                return value if interior else f"(BufferContains(p.field.outputs[{c}],z,y,x) ? {value} : 0.f)"
            kind = (
                2
                if name.startswith("inv_kappa_")
                else (0 if name.startswith("a_") else 1)
            )
            term = int(name[-1])
            return f"Profile(p,{term},{kind},z,y,x)"

        def output(write):
            return (
                f"next_psi_{int(write.field[5:])}"
                if write.field.startswith("psi_")
                else f"next_{'xyz'.index(write.field[1])}"
            )

        code = []
        if not phase and k:
            for c in range(3):
                value = prev(k - 1, c) if c == 1 else shared(k - 1, c, age=1)
                code.append(f"if(wave>={k} && {active(k - 1)}) old_E_{c}={value};")
        elif phase:
            # Hy's previous z plane is held in the component-specific register history.
            code.append(f"old_H_1={prev(k - 1, 1)};")
        if k:
            vals = [f"old_E_{c}" if not phase else cur("H", c) for c in range(3)]
            for c, value in enumerate(vals):
                intrinsic = "__shfl_down_sync" if not phase else "__shfl_up_sync"
                code.append(
                    f"const float neighbor_{c}={intrinsic}(0xffffffffu,{value},1);"
                )
        code += [f"float next_{c}=0.f;" for c in range(3)]
        code += [f"float next_psi_{t}=0.f;" for t in active_terms]
        derivatives = []
        for t in range(6):
            axis, c = TERM_AXES[t], TERM_COMPONENTS[t]
            field = ("H" if phase else "E") + "xyz"[c]
            offset = tuple((-1 if phase else 1) if a == axis else 0 for a in range(3))
            center = read(Read(field, phase, (0, 0, 0)))
            adjacent = read(Read(field, phase, offset))
            derivatives.append(
                f"const float derivative_{t}=" + (
                    f"beamz::cuda::yee::ScaleYeeDifference({'('+center+')-('+adjacent+')' if phase else '('+adjacent+')-('+center+')'},p.field.inv_resolution);"
                    if interior else f"Difference(p,{phase},{c},{axis},z,y,x,{center},{adjacent});"
                )
            )
        code += [
            f"if(wave>={k} && {active(k)}) {{",
            *derivatives,
            emit_cuda_block(
                selected,
                read_name=read,
                coefficient_name=coefficient,
                write_name=output,
                prefix=f"stage{k}",
            )["source"],
        ]
        for c in range(3):
            code.append("if(true) {" if interior else f"if(BufferContains(p.field.outputs[{c}],z,y,x)) {{")
            code.append(
                f"const bool constrained_{c}=false;" if interior else f"const bool constrained_{c}=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[{c}],{phase},{c},p.metallic_edges,z,y,x); if(constrained_{c}) next_{c}=0.f;"
            )
            if source_signature & (1 << (phase * 3 + c)):
                code.append(
                    f"if(!constrained_{c} && active_sources[{phase * 3 + c}]) next_{c}=AddFusedHSource<{c}>(next_{c},sources,z,y,x,{k // 2});"
                )
            code += [
                "if(owned) {",
                f"Publish(observation,p.field.outputs[{c}],{phase * 3 + c},{k // 2},z,y,x,next_{c});",
            ]
            if k >= stages - 2:
                code.append(
                    f"static_cast<float *>(final.fields[{phase * 3 + c}].data)[BeamzOffset3D<false>(final.fields[{phase * 3 + c}],z,y,x)]=next_{c};"
                )
            code += ["}", "}"]
        for t in active_terms:
            if k >= stages - 2:
                code.append(
                    f"if(owned) WritePsi<{str(half).lower()}>(p,{t},z,y,x,next_psi_{t});"
                )
        code.append("}")
        if k >= 2:
            code.append(
                f"if(wave>={k - 2} && {active(k - 2)}) {prev(k - 2, 1)}={cur(family, 1)};"
            )
            for t in active_terms:
                code.append(
                    f"if(wave>={k - 2} && {active(k - 2)}) {psi(k - 2, t)}=current_psi_{family}_{t};"
                )
        if not phase:
            for c in (0, 2):
                code.append(f"if(wave>{k} && {active(k)}) old_H_{c}={shared(k, c, age=1)};")
        code.append(f"if(wave>={k} && {active(k)}) {{")
        code += [f"{cur(family, c)}=next_{c};" for c in range(3)]
        code += [f"current_psi_{family}_{t}=next_psi_{t};" for t in active_terms]
        code.append("}")
        if not phase:
            if field_frames == 1:
                code.append("__syncthreads();")
            for c in (0, 2):
                code.append(
                    f"if(wave>={k} && {active(k)}) {shared(k, c)}={cur('H', c)};"
                )
            if k:
                for c in (0, 2):
                    code.append(
                        f"if(wave>={k - 1} && {active(k - 1)}) {shared(k - 1, c)}={cur('E', c)};"
                    )
            code.append("__syncthreads();")
        bodies.append(
            "\n".join(
                [
                    "{",
                    f"const auto &p={'e' if phase else 'h'}; const auto &sources={'es' if phase else 'hs'};",
                    f"const int z=oz+wave-{depth}-{k // 2},x=ox-{depth}+qx,y=oy-{depth}+qy;",
                    "const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;",
                    *code,
                    "}",
                ]
            )
        )
    wave_decl = [
        f"float current_{f}_{c}=0.f,old_{f}_{c}=0.f;" for f in "HE" for c in range(3)
    ]
    decl = [f"float previous_y_{k}=0.f;" for k in range(stages - 1)]
    decl += [
        f"float previous_psi_{k}_{t}=0.f;"
        for k in range(stages - 2)
        for t in active_terms
        if info["psi_history"] == "register"
    ]
    wave_decl += [f"float current_psi_{f}_{t}=0.f;" for f in "HE" for t in active_terms]
    name = f"DeepWarpD{depth}M{mask}{'BF16' if half else 'F32'}"
    name += ("ScalarH" if scalar_h else "") + ("Interior" if interior else "")
    name += ("BytePacked" if byte_packed else "") + (f"Source{source_signature}" if source_signature != 63 else "")
    name += "DoubleFrame" if field_frames == 2 else ""
    kernel = f"""
constexpr int {name}Threads={info["threads"]};
constexpr int {name}SharedBytes={info["shared_bytes"]};
template<bool Packed> __global__ __launch_bounds__({info["threads"]},1) void {name}(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {{
  if(blockDim.x!={info["threads"]})return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {{const int low=(grid.high[axis]+size-1)/size;
    return ({mask}&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;}};
  auto end=[&](int axis,int block,int origin,int size) {{const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,({mask}&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);}};
  const int ox=coordinate(2,blockIdx.x,{tx}),oy=coordinate(1,blockIdx.y,{ty}),oz=coordinate(0,blockIdx.z,{tz});
  const int end_x=end(2,blockIdx.x,ox,{tx}),end_y=end(1,blockIdx.y,oy,{ty}),end_z=end(0,blockIdx.z,oz,{tz});
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){{const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+{tz + depth} && starts[3*i]+g.coefficients.dims[1]>oz-{depth} &&
                starts[3*i+1]<oy+{ty + depth} && starts[3*i+1]+g.coefficients.dims[2]>oy-{depth} &&
                starts[3*i+2]<ox+{tx + depth} && starts[3*i+2]+g.coefficients.dims[3]>ox-{depth};
    active_sources[c]=active;}}
  __syncthreads();
  {chr(10).join(decl)}
  for(int wave=0;wave<end_z-oz+{stages - 1};++wave){{
    {chr(10).join(wave_decl)}
    {chr(10).join(bodies)}
    if(wave>={stages - 2} && {active(stages - 2)}) previous_y_{stages - 2}=current_H_1;
  }}
}}
"""
    return HELPERS + kernel + "\n}}} // namespace beamz::cuda::deep\n", info


def generate_bundle(*, typed_mie=False, double_frames=False):
    """Whole-grid D4 native hook, with mask-specialized private CPML state."""
    parts = [HELPERS]
    estimates = []
    for mask in range(8):
        tile = (64, 8 if mask in (0, 1, 2, 4) else (3 if mask == 7 else 4), 25)
        source, info = generate(tile, 4, mask, scalar_h=typed_mie,
                                byte_packed=typed_mie, source_signature=20 if typed_mie else 63,
                                field_frames=2 if double_frames and mask in (0, 1, 2, 4) else 1)
        parts.append(
            source[len(HELPERS) : -len("\n}}} // namespace beamz::cuda::deep\n")]
        )
        estimates.append(info)
    parts.append("\n}}} // namespace beamz::cuda::deep\n")
    host = r"""
int BeamzDeepD4AdmissionKind() { return @ADMISSION_KIND@; }
int BeamzEnqueueDeepWarpD4(void *opaque_stream,const BeamzProgramLaunch &program,
    const BeamzBuffer *final_fields,int step,const beamz::cuda::deep::Observation &observation) {
  const auto stream=static_cast<cudaStream_t>(opaque_stream);
  const auto &h=program.h_ab,&e=program.e_ab;
  if(!final_fields || !program.source_groups || program.source_group_count!=9 ||
     h.uniform_cpml_thickness!=12 || e.uniform_cpml_thickness!=12 ||
     h.nterms!=6 || e.nterms!=6 || h.metric_kind!=0 || e.metric_kind!=0 ||
     !HasPackedLosslessMaterial(e))return cudaErrorInvalidValue;
  for(int c=0;c<3;++c) {
    if(program.source_groups[c].coefficients.dims[0])return cudaErrorInvalidValue;
    if(h.inputs[6+c].rank!=0 || h.inputs[9+c].rank!=0)return cudaErrorInvalidValue;
  }
  for(int c=0;c<6;++c) {
    if(final_fields[c].element_type!=kBeamzF32)return cudaErrorInvalidValue;
    for(int q=0;q<3;++q)
      if(final_fields[c].data==h.inputs[q].data || final_fields[c].data==e.inputs[q].data)return cudaErrorInvalidValue;
    if(h.inputs[31+c].element_type!=kBeamzF32 || e.inputs[31+c].element_type!=kBeamzF32 ||
       h.outputs[3+c].element_type!=kBeamzF32 || e.outputs[3+c].element_type!=kBeamzF32 ||
       h.inputs[31+c].data==h.outputs[3+c].data || e.inputs[31+c].data==e.outputs[3+c].data)return cudaErrorInvalidValue;
  }
  PairFieldOutputs final{}; FusedHSources hs{},es{};hs.step=es.step=step;
  PairCpmlGrid grid{};
  for(int c=0;c<6;++c){final.fields[c]=final_fields[c];for(int a=0;a<3;++a)
    grid.extent[a]=std::max(grid.extent[a],static_cast<int>(final_fields[c].dims[a]));}
  for(int c=0;c<3;++c){hs.groups[c]=program.source_groups[3+c];es.groups[c]=program.source_groups[6+c];}
  for(int a=0;a<3;++a){int limit=static_cast<int>(h.outputs[0].dims[a]);
    for(int c=1;c<3;++c)limit=std::min(limit,static_cast<int>(h.outputs[c].dims[a]));
    if(limit<32)return cudaErrorInvalidValue;
    grid.field_high[a]=limit;grid.high[a]=16;grid.boundary[a]=limit-16;}
  const auto hp=MakePairCpmlPhase(h),ep=MakePairCpmlPhase(e);
  using namespace beamz::cuda::deep;
@TYPED_ADMISSION@
@LAUNCHES@
  return cudaSuccess;
}
int BeamzEnqueueDeepWarpDft4(void *opaque_stream,const BeamzProgramLaunch &program,
    int step,const beamz::cuda::deep::Observation &observation,const BeamzBuffer &remapped_indices) {
  if(!program.monitors || !program.monitors->monitor_count)return cudaSuccess;
  if(!observation.values || observation.key_count<1 || remapped_indices.rank!=4 ||
     remapped_indices.element_type!=kBeamzS32)return cudaErrorInvalidValue;
  auto h=program.h_ab,e=program.e_ab;auto monitors=*program.monitors;
  monitors.indices=remapped_indices;
  BeamzBuffer field{};field.rank=3;field.element_type=kBeamzF32;
  field.dims[0]=field.dims[1]=1;field.dims[2]=observation.key_count;
  field.row_stride=field.plane_stride=observation.key_count;
  for(int substep=0;substep<4;++substep){
    field.data=observation.values+substep*observation.key_count;
    for(int c=0;c<3;++c)h.outputs[c]=e.outputs[c]=field;
    const auto error=BeamzEnqueueDftGroups(static_cast<cudaStream_t>(opaque_stream),h,e,monitors,step+substep);
    if(error!=cudaSuccess)return error;
  }
  return cudaSuccess;
}
"""
    launches = []
    for info in estimates:
        mask, (_, y, x) = info["mask"], info["tile_zyx"]
        name = f"DeepWarpD4M{mask}F32" + ("ScalarHBytePackedSource20" if typed_mie else "")
        name += "DoubleFrame" if double_frames and mask in (0, 1, 2, 4) else ""
        launches.append(f"""{{ const int sizes[3]={{64,{y},{x}}};int counts[3];bool active=true;
for(int a=0;a<3;++a){{counts[a]=({mask}&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}}
if(active){{{name}<true><<<dim3(counts[2],counts[1],counts[0]),{name}Threads,{name}SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}}}""")
    admission = r"""
  int source_signature=0;
  for(int c=0;c<6;++c)if(program.source_groups[3+c].coefficients.dims[0]>0)source_signature|=1<<c;
  if(source_signature!=20)return cudaErrorInvalidValue;
  for(int c=0;c<3;++c) {
    if(h.inputs[6+c].element_type!=kBeamzF32 || h.inputs[9+c].element_type!=kBeamzF32 ||
       e.inputs[6+c].element_type!=kBeamzF32 || e.inputs[6+c].rank!=1 ||
       e.inputs[6+c].dims[0]<1 || e.inputs[6+c].dims[0]>2147483647LL ||
       e.inputs[9+c].element_type!=kBeamzS32 || e.inputs[9+c].rank!=1)return cudaErrorInvalidValue;
    int64_t volume=1;
    for(int a=0;a<3;++a) {
      const auto n=e.outputs[c].dims[a];
      if(n<1 || n>2147483647LL/volume)return cudaErrorInvalidValue;
      volume*=n;
    }
    if(e.inputs[9+c].dims[0]<(volume+3)/4)return cudaErrorInvalidValue;
  }
""" if typed_mie else ""
    parts.append(host.replace("@LAUNCHES@", "\n".join(launches)).replace("@TYPED_ADMISSION@\n", admission).replace("@ADMISSION_KIND@", "1" if typed_mie else "0"))
    return "\n".join(parts), estimates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--tile", type=int, nargs=3, default=(64, 8, 25))
    p.add_argument("--mask", type=int, default=7)
    p.add_argument("--half", action="store_true")
    p.add_argument("--bundle", action="store_true")
    p.add_argument("--typed-mie", action="store_true", help="Opt-in source20/scalarH/byte-packed lowering with strict host admission")
    p.add_argument("--double-frames", action="store_true", help="Two shared field frames for core/faces; preserve post-publication barriers")
    args = p.parse_args()
    source, info = (
        generate_bundle(typed_mie=args.typed_mie, double_frames=args.double_frames)
        if args.bundle
        else generate(tuple(args.tile), args.depth, args.mask, half=args.half)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
