// Research lowering of validated full-bank edge packets. Queue order and
// canonical numerical primitives are unchanged; no axis bucket permutation.
#pragma once
#include <initializer_list>
#include "native_moving.h"
struct MstepCompactField {
  float* data;
  int z, y, x, row, plane;
};
struct MstepCompactPsi {
  const float *input, *a, *b, *inv;
  float* output;
  int row, plane;
  int low, high;
};
struct MstepCompactEdgeView {
  MstepCompactField own[3], other[3], output[3];
  MstepCompactPsi psi[6];
  const float *decay[3], *source[3], *tables[3];
  const uint32_t* codes[3];
  int table_sizes[3], thickness, metallic_edges;
  int decay_strides[3][3], source_strides[3][3];
  bool dense_e;
  float inv;
  int metric_kind;const float*metrics[3];
};
__device__ __forceinline__ float CompactCoefficient(const float* values,const int* strides,
                                                     int z,int y,int x) {
  return __ldg(values+z*strides[0]+y*strides[1]+x*strides[2]);
}
__device__ __forceinline__ int CompactEdgeOffset(
    const MstepCompactField&f,int z,int y,int x) {
  return z*f.plane+y*f.row+x;
}
template<int Phase,int Axis>
__device__ __forceinline__ float CompactEdgeDifference(
    const MstepCompactField&f,int z,int y,int x,float inv) {
  const int coordinate=Axis==0?z:(Axis==1?y:x);
  const int size=Axis==0?f.z:(Axis==1?f.y:f.x);
  const int stride=Axis==0?f.plane:(Axis==1?f.row:1);
  if constexpr(Phase==0) {
    if(coordinate+1>=size)return 0.f;
    const int offset=CompactEdgeOffset(f,z,y,x);
    return beamz::cuda::yee::ScaleYeeDifference(f.data[offset+stride]-f.data[offset],inv);
  } else {
    // This is BoundaryDifference<0,false>, not exterior zero-extension.
    if(coordinate==0||coordinate==size)return 0.f;
    const int offset=CompactEdgeOffset(f,z,y,x);
    return beamz::cuda::yee::ScaleYeeDifference(f.data[offset]-f.data[offset-stride],inv);
  }
}
template<int Term>
__device__ __forceinline__ float CompactEdgeCorrect(
    const MstepCompactEdgeView&v,float derivative,int z,int y,int x) {
  constexpr int axis=beamz::cuda::yee::CpmlAxis(Term);
  constexpr float sign=beamz::cuda::yee::CpmlSign(Term);
  const auto&target=v.output[Term/2];
  const int coordinate=axis==0?z:(axis==1?y:x);
  const int size=axis==0?target.z:(axis==1?target.y:target.x);
  int packed;
  if(coordinate<v.thickness)packed=coordinate;
  else if(coordinate>=size-v.thickness)packed=2*v.thickness+coordinate-size;
  else return sign*derivative;
  const auto&p=v.psi[Term];
  const int offset=(axis==0?packed:z)*p.plane+(axis==1?packed:y)*p.row+(axis==2?packed:x);
  const float next=beamz::cuda::yee::AdvanceCpmlPsi(p.b[packed],p.input[offset],p.a[packed],derivative);
  p.output[offset]=next;
  return beamz::cuda::yee::CorrectCpmlDerivative(sign,derivative,p.inv[packed],next);
}
template<int Phase,int Component,int Layout>
__device__ __forceinline__ void CompactEdgeComponent(
    const MstepCompactEdgeView&v,int z,int y,int x) {
  const auto&out=v.output[Component];
  if(z>=out.z||y>=out.y||x>=out.x)return;
  constexpr auto first=beamz::cuda::yee::FirstCurlTerm(Component);
  constexpr auto second=beamz::cuda::yee::SecondCurlTerm(Component);
  const int d0_coordinate=first.derivative_axis==0?z:(first.derivative_axis==1?y:x);
  const float d0_metric=v.metric_kind==0?v.inv:__ldg(v.metrics[first.derivative_axis]+(v.metric_kind==1?0:d0_coordinate));
  const float d0=CompactEdgeDifference<Phase,first.derivative_axis>(v.other[first.source_component],z,y,x,d0_metric);
  const int d1_coordinate=second.derivative_axis==0?z:(second.derivative_axis==1?y:x);
  const float d1_metric=v.metric_kind==0?v.inv:__ldg(v.metrics[second.derivative_axis]+(v.metric_kind==1?0:d1_coordinate));
  const float d1=CompactEdgeDifference<Phase,second.derivative_axis>(v.other[second.source_component],z,y,x,d1_metric);
  const float curl=CompactEdgeCorrect<2*Component>(v,d0,z,y,x)+CompactEdgeCorrect<2*Component+1>(v,d1,z,y,x);
  float decay,source;
  if constexpr(Phase==0) {
    decay=*v.decay[Component];source=*v.source[Component];
  } else {
    decay=1.f;
    const int linear=(z*out.y+y)*out.x+x;
    if constexpr(Layout==2) {
      const int h=reinterpret_cast<const int32_t*>(v.codes[Component])[linear>>5];
      source=v.tables[Component][h<0?~h:h+(linear&31)];
    } else source=beamz::cuda::yee::PackedMaterialSource(v.tables[Component],v.table_sizes[Component],v.codes[Component],linear);
  }
  out.data[CompactEdgeOffset(out,z,y,x)]=beamz::cuda::yee::AdvanceYeeField(
      Phase,v.own[Component].data[CompactEdgeOffset(v.own[Component],z,y,x)],decay,source,curl);
}
template<int Phase,int Layout>
__global__ void MstepCompactFullEdge(
    const __grid_constant__ MstepCompactEdgeView v,const int*cells,int count,
    int ny,int nx,MstepIndexDecode decode) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count)return;
  const int row=cells[i],z=MstepQuotient(static_cast<uint32_t>(row),decode.plane);
  const int rem=row-z*ny*nx,y=MstepQuotient(static_cast<uint32_t>(rem),decode.x),x=rem-y*nx;
  CompactEdgeComponent<Phase,0,Layout>(v,z,y,x);
  CompactEdgeComponent<Phase,1,Layout>(v,z,y,x);
  CompactEdgeComponent<Phase,2,Layout>(v,z,y,x);
}
inline bool MakeMstepCompactEdge(const BeamzLaunch&p,MstepCompactEdgeView*v,int*layout,
                                 bool dense_e=false,
                                 const beamz::cuda::moving::CpmlWidth* widths=nullptr) {
  if(p.metric_kind<0||p.metric_kind>2||
     (widths ? p.nterms!=0&&p.nterms!=6 : p.metallic_edges||p.nterms!=6||p.uniform_cpml_thickness!=12))return false;
  *v={};v->thickness=p.uniform_cpml_thickness;v->inv=p.inv_resolution;*layout=0;
  v->metallic_edges=p.metallic_edges;
  v->metric_kind=p.metric_kind;
  if(p.metric_kind)for(int axis=0;axis<3;++axis){
    const auto&m=p.metrics[axis];
    if(!m.data||m.element_type!=kBeamzF32||m.rank!=(p.metric_kind==1?0:1))return false;
    if(p.metric_kind==2){
      int required=INT32_MAX;for(int c=0;c<3;++c)required=std::min(required,int(p.outputs[c].dims[axis]));
      if(m.dims[0]<required+p.phase)return false;
    }
    v->metrics[axis]=static_cast<const float*>(m.data);
  }
  auto field=[](const BeamzBuffer&b,MstepCompactField*f) {
    if(b.rank!=3||b.element_type!=kBeamzF32||b.compact_slots)return false;
    int64_t cells=1;
    for(int a=0;a<3;++a){if(b.dims[a]<1||b.dims[a]>INT32_MAX||cells>INT32_MAX/b.dims[a])return false;cells*=b.dims[a];}
    if(cells>INT32_MAX||b.row_stride<b.dims[2]||b.plane_stride<b.row_stride*b.dims[1]||
       static_cast<int64_t>(b.plane_stride)*b.dims[0]>INT32_MAX)return false;
    *f={static_cast<float*>(b.data),static_cast<int>(b.dims[0]),static_cast<int>(b.dims[1]),static_cast<int>(b.dims[2]),b.row_stride,b.plane_stride};return true;
  };
  const auto coefficient=[](const BeamzBuffer& b,const BeamzBuffer& target,const float** value,int* strides){
    if(!b.data||b.compact_slots||b.element_type!=kBeamzF32||(b.rank!=0&&b.rank!=3))return false;
    *value=static_cast<const float*>(b.data);
    if(!b.rank)return true;
    for(int d=0;d<3;++d)if(b.dims[d]!=1&&b.dims[d]!=target.dims[d])return false;
    strides[0]=b.dims[0]==1?0:b.plane_stride;
    strides[1]=b.dims[1]==1?0:b.row_stride;
    strides[2]=b.dims[2]==1?0:1;
    return true;
  };
  for(int c=0;c<3;++c) {
    if(!field(p.inputs[c],&v->own[c])||!field(p.inputs[3+c],&v->other[c])||!field(p.outputs[c],&v->output[c]))return false;
    for(int a=0;a<3;++a)if(p.inputs[c].dims[a]!=p.outputs[c].dims[a])return false;
    if(widths&&(p.phase==0||dense_e)) {
      const auto& target=p.outputs[c];
      if(!coefficient(p.inputs[6+c],target,&v->decay[c],v->decay_strides[c])||
         !coefficient(p.inputs[9+c],target,&v->source[c],v->source_strides[c]))return false;
      if(p.phase!=0){*layout=3;v->dense_e=true;}
    } else if(p.phase==0) {
      for(int k:{6+c,9+c})if(p.inputs[k].rank!=0||p.inputs[k].element_type!=kBeamzF32)return false;
      v->decay[c]=static_cast<const float*>(p.inputs[6+c].data);v->source[c]=static_cast<const float*>(p.inputs[9+c].data);
    } else if(dense_e) {
      const auto& decay = p.inputs[6+c];
      const auto& source = p.inputs[9+c];
      if (decay.rank != 0 || decay.element_type != kBeamzF32 ||
          source.rank != 3 || source.element_type != kBeamzF32 ||
          source.compact_slots || source.row_stride != source.dims[2] ||
          source.plane_stride != source.dims[1] * source.dims[2]) return false;
      for (int axis = 0; axis < 3; ++axis)
        if (source.dims[axis] != p.outputs[c].dims[axis]) return false;
      v->decay[c] = static_cast<const float*>(decay.data);
      v->source[c] = static_cast<const float*>(source.data);
      *layout = 3;
    } else {
      const auto&t=p.inputs[6+c];const auto&codes=p.inputs[9+c];
      if(t.rank!=1||t.element_type!=kBeamzF32||t.dims[0]<1||t.dims[0]>INT32_MAX||
         codes.element_type!=kBeamzS32||(codes.rank!=1&&codes.rank!=2))return false;
      if(c&&*layout!=codes.rank)return false;*layout=codes.rank;
      v->tables[c]=static_cast<const float*>(t.data);v->codes[c]=static_cast<const uint32_t*>(codes.data);v->table_sizes[c]=t.dims[0];
    }
  }
  for(int term=0;term<p.nterms;++term) {
    const int axis=beamz::cuda::yee::CpmlAxis(term);
    const auto&target=p.outputs[term/2];const auto&in=p.inputs[31+term];const auto&out=p.outputs[3+term];
    const int low=widths?widths[term].low:12, high=widths?widths[term].high:12;
    const int64_t packed=int64_t(low)+high;
    if(low<0||high<0||packed>target.dims[axis]||in.rank!=3||out.rank!=3||
       (in.element_type!=kBeamzF32&&(!widths||in.element_type!=kBeamzBF16))||
       out.element_type!=in.element_type||in.compact_slots||out.compact_slots)return false;
    if(widths&&p.uniform_cpml_thickness>0&&(low!=p.uniform_cpml_thickness||high!=p.uniform_cpml_thickness))return false;
    for(int a=0;a<3;++a)if(in.dims[a]!=(a==axis?packed:target.dims[a])||out.dims[a]!=in.dims[a])return false;
    if(!packed)continue;
    // Canonical CorrectCpml uses dense packed-slab offsets, regardless of
    // general field padding. Require that representation for both psi banks.
    for(const auto*psi:{&in,&out})if(psi->row_stride!=psi->dims[2]||psi->plane_stride!=psi->dims[1]*psi->dims[2])return false;
    const float*profiles[3];
    for(int k=0;k<3;++k) {
      const auto&b=p.inputs[13+3*term+k];
      if(b.element_type!=kBeamzF32||b.rank!=3||b.dims[axis]!=packed)return false;
      for(int a=0;a<3;++a)if(a!=axis&&b.dims[a]!=1)return false;
      profiles[k]=static_cast<const float*>(b.data);
    }
    v->psi[term]={static_cast<const float*>(in.data),profiles[0],profiles[1],profiles[2],static_cast<float*>(out.data),out.row_stride,out.plane_stride,low,high};
  }
  return true;
}
inline bool TryMstepCompactEdge(cudaStream_t stream,const BeamzLaunch&p,
    const BeamzBuffer&cells,int ny,int nx) {
  MstepCompactEdgeView v;int layout;
  if(!MakeMstepCompactEdge(p,&v,&layout))return false;
  const int count=cells.dims[0],blocks=(count+255)/256;
  const auto decode=MakeMstepIndexDecode(ny,nx);
  const auto*rows=static_cast<const int*>(cells.data);
  if(p.phase==0)MstepCompactFullEdge<0,0><<<blocks,256,0,stream>>>(v,rows,count,ny,nx,decode);
  else if(layout==2)MstepCompactFullEdge<1,2><<<blocks,256,0,stream>>>(v,rows,count,ny,nx,decode);
  else MstepCompactFullEdge<1,1><<<blocks,256,0,stream>>>(v,rows,count,ny,nx,decode);
  return true;
}
