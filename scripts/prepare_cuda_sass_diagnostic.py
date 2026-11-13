"""Add a host-only, opt-in profiling range around one actual NoIO launch."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha

HOOK=r'''
// Host-only diagnostics, outside graph capture and excluded from benchmark use.
struct ResearchSassScope{
 cudaStream_t stream;bool active=false;int(*finish)()=nullptr;const char*(*last_error)()=nullptr;
 ResearchSassScope(cudaStream_t s,int depth,int cube,int count,bool permitted=true):stream(s){
  const char*library=std::getenv("BEAMZ_SASS_RANGE_LIBRARY");if(!permitted||!library||depth!=16||cube!=512)return;
  static int matching=0;const int current=matching++;
  const char*requested=std::getenv("BEAMZ_SASS_MATCH_INDEX");const int target=requested?std::atoi(requested):64;
  if(current!=target)return;
  cudaStreamCaptureStatus capturing;auto status=cudaStreamIsCapturing(stream,&capturing);
  if(status!=cudaSuccess||capturing!=cudaStreamCaptureStatusNone)throw std::runtime_error("Software SASS diagnostics require direct launches");
  if(cudaStreamSynchronize(stream)!=cudaSuccess)throw std::runtime_error("Pre-profile stream sync failed");
  void*handle=dlopen(library,RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());
  auto begin=reinterpret_cast<int(*)(const char*)>(dlsym(handle,"beamz_sass_begin"));
  finish=reinterpret_cast<int(*)()>(dlsym(handle,"beamz_sass_end"));
  last_error=reinterpret_cast<const char*(*)()>(dlsym(handle,"beamz_sass_error"));
  if(!begin||!finish||!last_error)throw std::runtime_error("Missing software SASS diagnostic entry point");
  const char*output=std::getenv("BEAMZ_SASS_RANGE_OUTPUT");if(!output)throw std::runtime_error("Software SASS output required");
  if(begin(output))throw std::runtime_error(last_error());
  std::fprintf(stderr,"SASS diagnostic launch index%d depth%d cube%d CTAs%d\n",current,depth,cube,count);active=true;
 }
 ~ResearchSassScope()noexcept(false){if(active){if(cudaStreamSynchronize(stream)!=cudaSuccess)throw std::runtime_error("Software SASS target failed");if(finish())throw std::runtime_error(last_error());}}
};
'''


def stage(source,output,family='noio',expected_parent=PARENT_CONSUMER,expected_kernels=66):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied diagnostic source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=expected_parent:raise ValueError('Explicit frozen parent consumer required')
    text=original.replace('#include <limits>','#include <limits>\n#include <dlfcn.h>\n#include <cstdlib>\n#include <cstdio>\n#include <stdexcept>')
    anchor='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\ninline cudaError_t Launch('
    if text.count(anchor)!=1:raise ValueError('Exact profiling insertion changed')
    text=text.replace(anchor,HOOK+'\n'+anchor)
    if family=='noio':
        anchor='   AsymmetricNoIOFixedCube<Depth,512><<<uniform_count,1024,FixedCubeSharedBytes,stream>>>(p,ids);'
        hook='   ResearchSassScope diagnostic(stream,Depth,512,uniform_count);\n'
    elif family=='bulk':
        anchor=' Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<count,(Fast?1024:512),SharedBytes,stream>>>(p,ids);'
        if text.count(anchor)==0:anchor=' Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<count,(Fast?1024:512),LaunchBytes,stream>>>(p,ids);'
        hook=' ResearchSassScope diagnostic(stream,Depth,p.physical_n[0],count,Fast&&Layout==1&&IsotropicH&&p.physical_n[1]==512&&p.physical_n[2]==512);\n'
    elif family=='source':
        anchor='   AsymmetricSourcePending20<Depth,Policy><<<count,1024,SourcePendingSharedBytes,stream>>>(p,ids);'
        hook='   ResearchSassScope diagnostic(stream,Depth,p.physical_n[0],count,Policy==5&&p.physical_n[1]==512&&p.physical_n[2]==512);\n'
    else:raise ValueError('Explicit diagnostic family required')
    if text.count(anchor)!=1:raise ValueError('Exact selected kernel launch changed')
    text=text.replace(anchor,hook+anchor)
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    extension=(output/'src/native_extension.cc').read_text()
    graph='Check(BeamzLaunchGraph(stream_.value,key,(a.cuda_flags&kBeamzGraphCache)!=0,a.graph_cache_capacity,enqueue),"execute native moving graph");'
    if extension.count(graph)!=1:raise ValueError('Exact moving graph dispatch site required')
    # Graph-cache flags are required by packet admission and disabling the
    # cache still captures a one-shot graph. Bypass only this host dispatch
    # when the explicit diagnostic library environment variable is set.
    extension=extension.replace(graph,'if(std::getenv("BEAMZ_SASS_RANGE_LIBRARY"))Check(enqueue(),"execute diagnostic direct moving launches");else '+graph)
    extension='#include <cstdlib>\n'+extension
    (output/'src/native_extension.cc').write_text(extension)
    cmake=(output/'CMakeLists.txt').read_text()+'\nif(TARGET _cuda_native)\n target_link_libraries(_cuda_native PRIVATE ${CMAKE_DL_LIBS})\nendif()\n';(output/'CMakeLists.txt').write_text(cmake)
    proof=dict(schema='beamz.host-only-SASS-diagnostic.v2',family=family,parent_sha256=expected_parent,
               host_extension_sha256=sha(extension.encode()),direct_dispatch_diagnostic_env_only=True,
               consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               performance_candidate=False,required_device_gate='Exact selected-kernel SASS identity to parent before profiling',expected_kernels=expected_kernels,GPU_executed=False)
    (output.parent/'sass-diagnostic-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--expected-consumer-sha256',default=PARENT_CONSUMER);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-kernels',type=int,default=66);p.add_argument('--family',choices=('noio','bulk','source'),default='noio');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.family,a.expected_consumer_sha256,a.expected_kernels)))
