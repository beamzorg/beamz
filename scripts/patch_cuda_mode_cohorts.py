"""Join adjacent source-bearing cohorts inside each existing ready antichain.

Mode 4 joins source-only and source+observer IDs using Policy4; mode 5 also
joins boundary/source IDs using Policy5. Immutable tile flags retain all CPML,
source and observer effects. The fork/join waits use the same effective ranges
as the launches, including an empty first cohort. No tile or epoch is reordered
across a dependency level, and no new device storage is allocated.

BEAMZ_CUDA_MODE_POLICY_JOIN is captured during preparation (0, 4 or 5).
"""
from pathlib import Path
from build_cuda_block_uniform import once


def patch(source: Path):
    path = source / "src/native_extension.cc"
    text = path.read_text()
    text = '#include <cstdlib>\n' + text
    anchor = "    moving_workspace_.source_presence_mask=BeamzNativeMovingSourcePresenceMask(validate.program.source_groups,validate.program.source_group_count);"
    text = once(text, anchor, anchor + """
    const char*join_mode=std::getenv("BEAMZ_CUDA_MODE_POLICY_JOIN");
    if(join_mode){
      if(std::strcmp(join_mode,"4")==0)moving_workspace_.data_flags|=16u;
      else if(std::strcmp(join_mode,"5")==0)moving_workspace_.data_flags|=32u;
      else if(std::strcmp(join_mode,"0")!=0)throw std::invalid_argument("mode policy join must be0/4/5");
    }
""")
    text = once(text, '    result["moving_uniform_noio_tiles"]=moving_uniform_noio_;',
        '    result["moving_mode_policy_join"]=(moving_workspace_.data_flags&32u)?5:(moving_workspace_.data_flags&16u)?4:0;\n    result["moving_uniform_noio_tiles"]=moving_uniform_noio_;')
    text = once(text, '  module.attr("moving_mode_source54_six")=true;',
                '  module.attr("moving_mode_cohort_join")=true;\n  module.attr("moving_mode_source54_six")=true;')
    path.write_text(text)
    path = source / "src/asymmetric_native.cuh"
    text = path.read_text()
    anchor = "inline cudaError_t LaunchRanks(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){\n#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES"
    text = once(text, anchor, anchor + """
 const int source_join_last=(IsotropicH&&w.source_presence_mask==54u&&ModeSource54SixEligible(p))?
     ((w.data_flags&32u)?5:(w.data_flags&16u)?4:0):0;
""")
    text = once(text,
        "  for(int policy=0;policy<6;++policy){const int start=offsets[policy],count=offsets[policy+1]-start;",
        """  for(int policy=0;policy<6;++policy){
   if(source_join_last&&policy>3&&policy<=source_join_last)continue;
   const int start=offsets[policy],count=offsets[source_join_last&&policy==3?source_join_last+1:policy+1]-start;""")
    text = once(text,
        "    case 3:status=LaunchSourcePending<Depth,Layout,IsotropicH,3>(branch,p,w.tile_ids_device+start,count,w.source_presence_mask,bool(w.data_flags&8u));break;",
        """    case 3:
     if(source_join_last==5)status=LaunchSourcePending<Depth,Layout,IsotropicH,5>(branch,p,w.tile_ids_device+start,count,w.source_presence_mask,false);
     else if(source_join_last==4)status=LaunchSourcePending<Depth,Layout,IsotropicH,4>(branch,p,w.tile_ids_device+start,count,w.source_presence_mask,false);
     else status=LaunchSourcePending<Depth,Layout,IsotropicH,3>(branch,p,w.tile_ids_device+start,count,w.source_presence_mask,bool(w.data_flags&8u));
     break;""")
    text = once(text,
        "   if(offsets[policy]==offsets[policy+1])continue;",
        """   if(source_join_last&&policy>3&&policy<=source_join_last)continue;
   if(offsets[policy]==offsets[source_join_last&&policy==3?source_join_last+1:policy+1])continue;""")
    path.write_text(text)
