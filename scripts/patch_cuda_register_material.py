"""Add exact private uniform-E column facts to a copied adaptive D2 candidate.

High bit20 is private device metadata. Input arrays and the public packet ABI
are unchanged. Only the register-core wrapper consumes this bit; ordinary
kernels continue to use the original four quadrant ownership bits. Unsupported
material encodings keep the original decoder.
"""

import argparse
from pathlib import Path

FACTS = r"""
  // RESEARCH_UNIFORM_D2: certify the entire column including every halo stage.
  void PrepareUniformD2() {
    if(!mstep_||workspace_count_!=12||!attributes_.cpml_enabled)return;
    const int bounds_index=monitor_offset_+16,mask_index=bounds_index+1;
    const auto&bounds=host_[bounds_index];const auto&masks=host_[mask_index];
    if(bounds.descriptor.rank!=1||bounds.bytes!=8||bounds.descriptor.element_type!=kBeamzS32||
       masks.descriptor.rank!=2||masks.descriptor.element_type!=kBeamzS32)return;
    const auto*limits=static_cast<const int32_t*>(bounds.data);
    if(limits[0]<0||limits[1]<=limits[0]||limits[1]>INT32_MAX-2)return;
    uint32_t lookup[3][256]{};uint32_t common[3]{};int dims[3][3]{};
    const uint32_t*codes[3]{};
    for(int c=0;c<3;++c){
      const auto&t=host_[37+c];const auto&code=host_[40+c];const auto&f=host_[3+c].descriptor;
      if(t.descriptor.rank!=1||t.descriptor.element_type!=kBeamzF32||t.descriptor.dims[0]<1||t.descriptor.dims[0]>256||
         code.descriptor.rank!=1||code.descriptor.element_type!=kBeamzS32||f.rank!=3||f.element_type!=kBeamzF32)return;
      int64_t count=1;
      for(int a=0;a<3;++a){if(f.dims[a]<1||f.dims[a]>INT32_MAX)return;dims[c][a]=int(f.dims[a]);if(count>INT32_MAX/f.dims[a])return;count*=f.dims[a];}
      if(code.bytes!=size_t((count+3)/4)*4||t.bytes!=size_t(t.descriptor.dims[0])*4)return;
      std::memcpy(lookup[c],t.data,t.bytes);common[c]=lookup[c][t.descriptor.dims[0]-1];codes[c]=static_cast<const uint32_t*>(code.data);
    }
    const int64_t rows=masks.descriptor.dims[0],columns=masks.descriptor.dims[1];
    if(rows<1||columns<1||rows>INT32_MAX/columns||masks.bytes!=size_t(rows*columns)*4)return;
    const auto*original=static_cast<const int32_t*>(masks.data);
    for(int64_t i=0;i<rows*columns;++i)if(original[i]<0||original[i]>15)return;
    uniform_d2_masks_.assign(original,original+rows*columns);
    for(int64_t by=0;by<rows;++by)for(int64_t bx=0;bx<columns;++bx){
      const size_t tile=by*columns+bx;if(!original[tile])continue;bool same=true;
      for(int c=0;c<3&&same;++c){
        // A coarse tile covers every16x8 sub-tile and the largest32x16 core.
        // Bounds are deliberately wider than E1/E2 coefficient trajectories.
        const int z0=std::max(0,limits[0]-2),z1=std::min(dims[c][0],limits[1]+2);
        const int y0=int(std::max<int64_t>(0,by*16-2)),y1=int(std::min<int64_t>(dims[c][1],by*16+18));
        const int x0=int(std::max<int64_t>(0,bx*32-2)),x1=int(std::min<int64_t>(dims[c][2],bx*32+34));
        for(int z=z0;z<z1&&same;++z)for(int y=y0;y<y1&&same;++y)for(int x=x0;x<x1;++x){
          const int i=(z*dims[c][1]+y)*dims[c][2]+x;const unsigned code=(codes[c][i>>2]>>(8*(i&3)))&255u;
          if(lookup[c][code]!=common[c]){same=false;break;}
        }
      }
      if(same){uniform_d2_masks_[tile]|=1<<20;++uniform_d2_tiles_;}
    }
    Check(cudaMemcpyAsync(base_inputs_[mask_index].data,uniform_d2_masks_.data(),masks.bytes,cudaMemcpyHostToDevice,stream_.value),"upload private uniform-E facts");
  }
  std::vector<int32_t> uniform_d2_masks_;
  size_t uniform_d2_tiles_=0;
"""


def once(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f"Expected one frozen anchor: {old!r}")
    return s.replace(old, new, 1)


def patch(source):
    if source.resolve() == (Path(__file__).resolve().parents[1] / "cuda").resolve():
        raise ValueError("Use a copied source tree")
    p = source / "src/mstep_register.cuh"
    s = p.read_text()
    s = once(
        s,
        "template <int X, int Y, int Layout, bool LosslessH>",
        "template <int X, int Y, int Layout, bool LosslessH, bool UniformE>",
    )
    old = "const float source = MstepMaterialSource<Layout>(v, c, e_offset[c][slot]);"
    s = once(
        s,
        old,
        "const float source = UniformE ? __ldg(v.tables[c]+v.table_sizes[c]-1) : MstepMaterialSource<Layout>(v, c, e_offset[c][slot]);",
    )
    old = "const float source = MstepMaterialSource<Layout>(v, c, e_linear);"
    s = once(
        s,
        old,
        "const float source = UniformE ? __ldg(v.tables[c]+v.table_sizes[c]-1) : MstepMaterialSource<Layout>(v, c, e_linear);",
    )
    old = """ if(admitted)MstepRegisterColumnBody<X,Y,Layout,true>(v,masks,bounds,ntx,segment);
 else MstepRegisterColumnBody<X,Y,Layout,false>(v,masks,bounds,ntx,segment);"""
    new = """ const int ox=blockIdx.x*X,oy=blockIdx.y*Y;
 const bool uniform=(masks[(oy/16)*ntx+ox/32]&(1<<20))!=0;
 if(admitted){
  if(uniform)MstepRegisterColumnBody<X,Y,Layout,true,true>(v,masks,bounds,ntx,segment);
  else MstepRegisterColumnBody<X,Y,Layout,true,false>(v,masks,bounds,ntx,segment);
 }else{
  if(uniform)MstepRegisterColumnBody<X,Y,Layout,false,true>(v,masks,bounds,ntx,segment);
  else MstepRegisterColumnBody<X,Y,Layout,false,false>(v,masks,bounds,ntx,segment);
 }"""
    s = once(s, old, new)
    p.write_text(s)
    p = source / "src/native_extension.cc"
    s = p.read_text()
    s = once(
        s,
        "    ValidateClock(monitor_offset_ + 13, kBeamzF32);",
        "    PrepareUniformD2();\n    ValidateClock(monitor_offset_ + 13, kBeamzF32);",
    )
    s = once(s, " private:\n", " private:\n" + FACTS)
    s = once(
        s,
        '    result["input_count"] = host_.size();',
        '    result["uniform_d2_tiles"] = uniform_d2_tiles_;\n    result["input_count"] = host_.size();',
    )
    p.write_text(s)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-source", type=Path, required=True)
    patch(p.parse_args().candidate_source)
