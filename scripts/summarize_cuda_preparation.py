"""Summarize fresh-process CUDA preparation traces, including failed trials."""

import argparse
import csv
import json
from pathlib import Path

from scripts.summarize_hopper_cubes import timed_telemetry


def summarize(root):
    rows = []
    for path in sorted(root.rglob("cube-*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue  # A running trial may be between writes.
        if "side" not in data:
            continue
        exit_path = path.with_name(f"cube-{data['side']}-exit.json")
        exit_code = (
            json.loads(exit_path.read_text())["exit_code"]
            if exit_path.exists()
            else None
        )
        stages = data.get("stages", [])
        program = next((s for s in stages if s["stage"] == "program_built"), None)
        compiled = next((s for s in stages if s["stage"] == "compiled"), None)
        trace_path = path.with_suffix(".stages.jsonl")
        trace = []
        if trace_path.exists():
            for line in trace_path.read_text().splitlines():
                try:
                    trace.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        diagnostic = [s for s in trace if s["stage"] == "_launch_power_diagnostics_3d"]
        diagnostic_s = (
            diagnostic[-1]["unix_time"] - diagnostic[0]["unix_time"]
            if len(diagnostic) == 2
            else None
        )
        runtime = [s for s in stages if s["stage"].startswith("iteration_")]
        temp = data.get("executable_memory", {}).get("temp_size_in_bytes", 0)
        timezone_path = path.parent / "telemetry-timezone.json"
        offset = (
            json.loads(timezone_path.read_text())["utc_offset_seconds"]
            if timezone_path.exists()
            else 0
        )
        telemetry_data = dict(
            data,
            sample_windows_unix=[
                [start + offset, end + offset]
                for start, end in data.get("sample_windows_unix", [])
            ],
        )
        rows.append(
            dict(
                variant=path.parent.name,
                side=data["side"],
                cells=data["cells"],
                status="failed" if exit_code else data.get("status", data["stage"]),
                exit_code=exit_code,
                allocator=data["allocator_env"].get(
                    "XLA_PYTHON_CLIENT_ALLOCATOR", "bfc"
                ),
                setup_s=data.get("setup_s"),
                source_diagnostic_s=diagnostic_s,
                xla_compile_s=data.get("compile_s"),
                setup_peak_live_gib=(
                    program["memory_stats"]["peak_bytes_in_use"] / 2**30
                )
                if program
                else None,
                xla_peak_live_gib=(
                    compiled["memory_stats"]["peak_bytes_in_use"] / 2**30
                )
                if compiled
                else None,
                overall_peak_live_gib=max(
                    s["memory_stats"]["peak_bytes_in_use"] for s in stages + trace
                )
                / 2**30,
                runtime_live_plus_temp_estimate_gib=(
                    max(s["memory_stats"]["bytes_in_use"] for s in runtime) + temp
                )
                / 2**30
                if runtime
                else None,
                timed_nvml_peak_gib=timed_telemetry(path, telemetry_data)[
                    "timed_nvml_memory_max_gib"
                ],
                host_peak_gib=max(
                    [data["host_peak_rss_bytes"]]
                    + [s.get("host_peak_rss_bytes", 0) for s in trace]
                )
                / 2**30,
                gcups=data.get("gcups"),
                finite=data.get("finite_complete_state"),
                error=data.get(
                    "error",
                    f"Process exited {exit_code}; last stage {trace[-1]['stage']} {trace[-1]['event']}"
                    if exit_code and trace
                    else "",
                ),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
