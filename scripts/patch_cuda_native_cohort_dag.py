"""Copy-only exact per-cohort dependency scheduler for native temporal CUDA.

Kernel Parameters, canonical point code, fields, imports, trails and all CTA
barriers remain unchanged. Only host graph dependencies/event allocation change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXPECTED = {
    "asymmetric_native.cuh": "f0657105ea5a57219d48a7a067d1ef53f6246676579e722ffd0670a14d3d8f49",
    "native_extension.cc": "8f122cdbb850374f44c8e59c3282274b744ebff0ef5a907540a245a53a730bcf",
    "native_moving.h": "3932dc6c86898d1090e9d9efba2a176ad9300a8f63cefe6430dfa0901dc7345b",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(t, a, b):
    if t.count(a) != 1:
        raise ValueError(f"Expected exact one cohort patch anchor:{a!r}")
    return t.replace(a, b, 1)


MEMBERS = """  BeamzNativeMovingCohorts moving_cohort_plan_;
  std::vector<std::unique_ptr<MovingDependencyEvent>>moving_cohort_events_;
  std::vector<void*>moving_cohort_event_handles_;
"""
PREPARE = """    moving_cohort_plan_=BeamzBuildNativeMovingCohorts(moving_plan_);
    const size_t cohort_count=moving_cohort_plan_.offsets.size()-1;
    while(moving_cohort_events_.size()<cohort_count)moving_cohort_events_.push_back(std::make_unique<MovingDependencyEvent>());
    moving_cohort_event_handles_.clear();moving_cohort_event_handles_.reserve(cohort_count);
    for(size_t i=0;i<cohort_count;++i)moving_cohort_event_handles_.push_back(moving_cohort_events_[i]->value);
    moving_workspace_.cohort_offsets_host=moving_cohort_plan_.offsets.data();
    moving_workspace_.cohort_dependencies_host=moving_cohort_plan_.dependencies.data();
    moving_workspace_.cohort_dependency_count=moving_cohort_plan_.dependencies.size();
    moving_workspace_.cohort_last_policy_host=moving_cohort_plan_.last_policy_cohort.data();
    moving_workspace_.cohort_events_host=moving_cohort_event_handles_.data();
"""
VALIDATE = """  if(!w.cohort_offsets_host||!w.cohort_events_host||!w.cohort_last_policy_host||w.cohort_dependency_count<0||
     (w.cohort_dependency_count&&!w.cohort_dependencies_host)||w.cohort_offsets_host[0]!=0||w.cohort_offsets_host[w.rank_count*6]!=w.cohort_dependency_count)return false;
  for(int cohort=0;cohort<w.rank_count*6;++cohort){
   const int a=w.cohort_offsets_host[cohort],b=w.cohort_offsets_host[cohort+1];if(a<0||b<a||b>w.cohort_dependency_count)return false;
   for(int i=a;i<b;++i){const int producer=w.cohort_dependencies_host[i];if(producer<0||producer/6>=cohort/6||!w.cohort_events_host[producer])return false;}
   const int r=cohort/6,policy=cohort%6;const auto*o=w.policy_offsets_host+r*7;
   if(o[policy]!=o[policy+1]&&!w.cohort_events_host[cohort])return false;
  }
  for(int policy=0;policy<6;++policy){int last=-1;for(int r=0;r<w.rank_count;++r){const auto*o=w.policy_offsets_host+r*7;if(o[policy]!=o[policy+1])last=r*6+policy;}if(w.cohort_last_policy_host[policy]!=last)return false;}
"""


def transform(c, e, h, body):
    anchor = (
        "template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks"
    )
    c = once(c, anchor, body + "\n" + anchor)
    anchor = "inline cudaError_t LaunchRanks(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){"
    c = once(
        c,
        anchor,
        anchor
        + "\n#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES\n if(w.parallel_policies)return LaunchCohortRanks<Depth,Layout,IsotropicH>(stream,p,w);\n#endif",
    )
    anchor = "  for(int policy=0;policy<6;++policy)if(!w.policy_streams[policy]||!w.policy_done_events[policy])return false;"
    c = once(c, anchor, anchor + "\n" + VALIDATE)
    h = once(
        h,
        "  bool parallel_policies=false;",
        """  const int32_t* cohort_offsets_host=nullptr;
  const int32_t* cohort_dependencies_host=nullptr;
  const int32_t* cohort_last_policy_host=nullptr;
  void*const* cohort_events_host=nullptr;
  int cohort_dependency_count=0;
  bool parallel_policies=false;""",
    )
    e = once(
        e,
        '#include "native_moving.h"',
        '#include "native_moving.h"\n#include "native_cohort_dependencies.h"',
    )
    e = once(
        e,
        "  BeamzNativeMovingPlan moving_plan_;",
        MEMBERS + "  BeamzNativeMovingPlan moving_plan_;",
    )
    e = once(
        e,
        "    moving_workspace_.policy_fork_event=moving_policy_fork_->value;moving_workspace_.parallel_policies=true;",
        "    moving_workspace_.policy_fork_event=moving_policy_fork_->value;moving_workspace_.parallel_policies=true;\n"
        + PREPARE,
    )
    anchor = "    append_bytes(&moving_workspace_.parallel_policies,sizeof(bool));"
    e = once(
        e,
        anchor,
        anchor
        + """
    const uint32_t cohort_schema=1;append_bytes(&cohort_schema,sizeof(cohort_schema));
    append_bytes(moving_cohort_plan_.offsets.data(),moving_cohort_plan_.offsets.size()*4);
    append_bytes(moving_cohort_plan_.dependencies.data(),moving_cohort_plan_.dependencies.size()*4);
    append_bytes(moving_cohort_plan_.last_policy_cohort.data(),moving_cohort_plan_.last_policy_cohort.size()*4);
    for(void*handle:moving_cohort_event_handles_)append_bytes(&handle,sizeof(handle));""",
    )
    anchor = '    result["moving_parallel_policies_compiled"]='
    if e.count(anchor) != 1:
        raise ValueError("Metadata anchor changed")
    e = e.replace(
        anchor,
        """    result["moving_cohort_scheduler"]="dependency_dag_v1";
    result["moving_cohort_count"]=moving_cohort_plan_.active_cohorts;
    result["moving_cohort_dependency_edges"]=moving_cohort_plan_.dependencies.size();
    result["moving_cohort_event_count"]=moving_cohort_event_handles_.size();
    result["moving_cohort_host_bytes"]=(moving_cohort_plan_.offsets.size()+moving_cohort_plan_.dependencies.size()+moving_cohort_plan_.last_policy_cohort.size())*4+moving_cohort_event_handles_.size()*sizeof(void*);
    result["moving_cohort_max_distance"]=moving_cohort_plan_.maximum_dependency_distance;
"""
        + anchor,
        1,
    )
    # Extend CPU resource planning without allocating or touching any GPU.
    anchor = '  size_t bulk=0;for(int count:p.bulk_counts)bulk+=count;r["bulk_tile_count"]=bulk;return r;'
    e = once(
        e,
        anchor,
        """  const auto cohort=BeamzBuildNativeMovingCohorts(p);
  r["cohort_active_count"]=cohort.active_cohorts;r["cohort_cross_stream_edges"]=cohort.dependencies.size();r["cohort_max_distance"]=cohort.maximum_dependency_distance;
"""
        + anchor,
    )
    e = once(
        e,
        "NB_MODULE(_cuda_native, module) {",
        'NB_MODULE(_cuda_native, module) {\n  module.attr("moving_cohort_scheduler")="dependency_dag_v1";',
    )
    # Prepare admission changed, but all device definitions before MakeSource
    # must remain byte-for-byte. The host launcher is the sole execution change.
    return c, e, h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src = a.source.resolve()
    out = a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        out.exists()
        or src == out
        or out == canonical
        or src in out.parents
        or out in src.parents
    ):
        p.error("New disjoint private copied source required")
    for name, value in EXPECTED.items():
        if sha((src / "src" / name).read_bytes()) != value:
            raise ValueError(f"Exact frozen Observer9f6a parent required:{name}")
    c = (src / "src/asymmetric_native.cuh").read_text()
    e = (src / "src/native_extension.cc").read_text()
    h = (src / "src/native_moving.h").read_text()
    body = Path(__file__).with_name("cuda_native_cohort_schedule.inc").read_text()
    newc, newe, newh = transform(c, e, h, body)
    assert (
        newc[: newc.index("inline bool MakeSource(")]
        == c[: c.index("inline bool MakeSource(")]
    )
    shutil.copytree(src, out)
    for name, value in [
        ("asymmetric_native.cuh", newc),
        ("native_extension.cc", newe),
        ("native_moving.h", newh),
    ]:
        (out / "src" / name).write_text(value)
    builder = Path(__file__).with_name("cuda_native_cohort_dependencies.h")
    shutil.copy2(builder, out / "src/native_cohort_dependencies.h")
    proof = {
        "parent_sources": EXPECTED,
        "consumer_sha256": sha(newc.encode()),
        "extension_sha256": sha(newe.encode()),
        "workspace_header_sha256": sha(newh.encode()),
        "builder_sha256": sha(builder.read_bytes()),
        "schedule_sha256": sha(body.encode()),
        "device_prefix_bytewise_unchanged": True,
        "expected_kernels": 52,
        "event_allocation": "rank_count*6 persistent host-owned handles; reused only after final macro join",
        "dependencies": "actual7neighborRAW+actualphysicalownerpreviousproducerWAW+alloldconsumersWAR; same-policy implicitstream order",
        "scope": "CPU-proved host scheduler candidate, no GPU performance/correctness claim until gates",
    }
    (out.parent / "cohort-source-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
