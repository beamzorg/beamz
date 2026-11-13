"""Emit same-rank policy fork/join launches; kernel arithmetic stays unchanged.

Requires matching persistent streams/events in the private native workspace.
All auxiliary branches join the main stream before another rank or return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MACRO = "BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one parallel-policy anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source):
    source = once(
        source,
        "#pragma once",
        f"#pragma once\n#ifndef {MACRO}\n#define {MACRO} 0\n#endif",
    )
    source = once(
        source,
        " if(!w.policy_offsets_host)return false;",
        """ if(!w.policy_offsets_host)return false;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
 if(w.parallel_policies){
  if(!w.policy_fork_event)return false;
  for(int policy=0;policy<6;++policy)if(!w.policy_streams[policy]||!w.policy_done_events[policy])return false;
 }
#endif""",
    )
    start = source.index("inline cudaError_t LaunchRanks(")
    end = source.index(
        "template<int Depth,int Layout>\ninline cudaError_t DispatchClass", start
    )
    body = source[start:end]
    body = once(
        body,
        " for(int rank=0;rank<w.rank_count;++rank){const auto*offsets=w.policy_offsets_host+rank*7;",
        """ for(int rank=0;rank<w.rank_count;++rank){const auto*offsets=w.policy_offsets_host+rank*7;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
  const bool parallel=w.parallel_policies;
  if(parallel){const auto status=cudaEventRecord(static_cast<cudaEvent_t>(w.policy_fork_event),stream);if(status!=cudaSuccess)return status;}
#endif""",
    )
    body = once(
        body,
        "   if(!count)continue;cudaError_t status=cudaSuccess;",
        """   if(!count)continue;cudaError_t status=cudaSuccess;cudaStream_t branch=stream;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
   if(parallel){branch=static_cast<cudaStream_t>(w.policy_streams[policy]);status=cudaStreamWaitEvent(branch,static_cast<cudaEvent_t>(w.policy_fork_event),0);if(status!=cudaSuccess)return status;}
#endif""",
    )
    for policy in range(6):
        fast = "true" if policy == 0 else "false"
        body = once(
            body,
            f"Launch<Depth,Layout,{fast},IsotropicH,{policy}>(stream,p,",
            f"Launch<Depth,Layout,{fast},IsotropicH,{policy}>(branch,p,",
        )
    body = once(
        body,
        "   if(status!=cudaSuccess)return status;\n  }\n }\n return cudaSuccess;\n#else",
        """   if(status!=cudaSuccess)return status;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
   if(parallel){status=cudaEventRecord(static_cast<cudaEvent_t>(w.policy_done_events[policy]),branch);if(status!=cudaSuccess)return status;}
#endif
  }
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
  if(parallel)for(int policy=0;policy<6;++policy){
   if(offsets[policy]==offsets[policy+1])continue;
   const auto status=cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(w.policy_done_events[policy]),0);if(status!=cudaSuccess)return status;
  }
#endif
 }
 return cudaSuccess;
#else""",
    )
    return source[:start] + body + source[end:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve():
        p.error("Input is frozen; output must differ")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(patch(a.input.read_text()))
    print(
        json.dumps(
            {
                "input": str(a.input),
                "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
                "output": str(a.output),
                "output_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
                "parallel_macro_default": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
