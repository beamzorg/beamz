#!/usr/bin/env python3
"""Audit executed macro/fallback witnesses in already accepted GPU reports."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--kind", choices=("axis", "dag"), required=True)
    p.add_argument("--scheduler", default="dependency_dag_v1")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    digest = hashlib.sha256(a.module.read_bytes()).hexdigest()
    result = {"module_sha256": digest, "kind": a.kind, "passed": False, "files": []}
    try:
        for name in ("probes", "blocks", "constant-blocks", "nonisotropic"):
            path = a.root / (name + ".json")
            report = json.loads(path.read_text())
            if not report["passed"] or report["module_sha256"] != digest:
                raise ValueError(f"Unaccepted or different module report: {path}")
            rows = [row for case in report["cases"] for row in case.get("results", [case])]
            witnesses = []
            for row in rows:
                if not row["branch_witness"]["positive_macro"]:
                    raise ValueError("A metadata witness did not execute a macro")
                m = row["metadata"]
                if a.kind == "axis":
                    counts = list(m["moving_noio_axis_tiles_zyx"])
                    if len(counts) != 3 or any(x < 0 for x in counts):
                        raise ValueError("Invalid axis counters")
                    if (name == "probes" and not all(x > 0 for x in counts)) or (
                            name != "probes" and any(counts)):
                        raise ValueError(f"Unexpected active/fallback axis counters: {counts}")
                    witnesses.append(counts)
                else:
                    if (m["moving_cohort_scheduler"] != a.scheduler
                            or m["moving_cohort_count"] <= 0
                            or m["moving_cohort_dependency_edges"] <= 0
                            or m["moving_cohort_event_count"] < m["moving_cohort_count"]):
                        raise ValueError("Dependency DAG was not active")
                    witnesses.append({key: m[key] for key in (
                        "moving_cohort_count", "moving_cohort_dependency_edges",
                        "moving_cohort_event_count", "moving_cohort_max_distance")})
            result["files"].append({"path": str(path), "witnesses": witnesses})
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "kind": a.kind}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
