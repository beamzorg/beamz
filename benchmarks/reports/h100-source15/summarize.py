"""Summarize measured full public capacity results without extrapolation."""

import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
case = root / "raw/public-eight-z-15b-cuda_streamed-propagated.json"
data = json.loads(case.read_text())
stages = data["stages"]
runs = [
    s
    for s in stages
    if s["stage"].startswith("run_") and not s["stage"].endswith("failed")
]
warm = [s["seconds"] for s in runs[1:]]
result = {
    "status": data["status"],
    "failure_stage": data.get("failure_stage"),
    "cells": data["cells"],
    "final_step": data.get("final_step"),
    "steps_per_segment": data["args"]["steps"],
    "segment_seconds": [s["seconds"] for s in runs],
    "warm_median_gcups": data["cells"]
    * data["args"]["steps"]
    / statistics.median(warm)
    / 1e9
    if warm
    else None,
    "total_stepping_seconds": sum(s["seconds"] for s in runs),
    "warm_runtime_cv": statistics.pstdev(warm) / statistics.mean(warm)
    if warm
    else None,
    "cuda_schedule": data.get("cuda_schedule"),
    "executable_bytes": data.get("executable_bytes"),
    "monitor_signal": data.get("monitor_signal"),
    "host_peak_gib": max(s["max_rss_bytes"] for s in stages) / 2**30,
    "stages": [
        {
            "stage": s["stage"],
            "seconds": s["seconds"],
            "gpus": [
                {
                    "id": g["id"],
                    "live_gib": g["stats"]["bytes_in_use"] / 2**30,
                    "peak_gib": g["stats"]["peak_bytes_in_use"] / 2**30,
                }
                for g in s["gpus"]
            ],
        }
        for s in stages
    ],
}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "stages"}, indent=2))
