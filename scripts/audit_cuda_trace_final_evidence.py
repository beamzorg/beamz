"""Read-only independent evidence audit; writes only separate audit artifacts."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def parity_summary(value):
    leaves = value.get("leaves", [])
    if isinstance(leaves, dict):
        leaves = list(leaves.values())
    return {"passed": value.get("passed"), "bitwise_exact": value.get("bitwise_exact", value.get("all30_bitwise")),
            "leaves": len(leaves), "every_leaf_bitwise_finite": len(leaves) == 30 and all(
                x.get("bitwise_exact") and x.get("finite") is True for x in leaves)}


def comparison(path):
    a = load(path)
    assert a["control_identity_verified"] is True
    pairs = []
    for pair in a["pairs"]:
        outputs = {}
        for role in ("baseline", "candidate"):
            ref = pair[role]; report_path = Path(ref["path"]); r = load(report_path)
            if role == "baseline":
                assert ref["module_sha256"] == a["expected_control_sha256"]
            fixture_path = Path(r["fixture"]); f = load(fixture_path)
            attrs = f["attributes"]; spec = f["fixture"]["spec"]
            steps = sum(r["chunks"])
            assert steps == attrs["nsteps"]
            assert ref["sha256"] == sha(report_path)
            assert r["module_sha256"] == ref["module_sha256"]
            assert r["fixture_sha256"] == sha(fixture_path)
            module = Path(r["module"])
            assert module.exists() and sha(module) == r["module_sha256"]
            first, replay = parity_summary(r["parity"]), parity_summary(r["repeat_parity"])
            assert r["status"] == "ok" and first["every_leaf_bitwise_finite"] and replay["every_leaf_bitwise_finite"]
            assert r["host_inputs_unchanged"] and r["jax_imported"] is False
            assert spec["pml_cells"] == 12 and spec["pml_policy"] == "fixed_cells"
            outputs[role] = {"report": str(report_path), "report_sha256": sha(report_path),
                "fixture": str(fixture_path), "fixture_sha256": sha(fixture_path),
                "module_sha256": r["module_sha256"], "module_hash_verified": True,
                "shape_zyx": spec["shape_zyx"], "steps": steps, "planned_full_steps": spec["timesteps"],
                "complete_physical_duration": steps == spec["timesteps"],
                "duration_fs": steps*attrs["dt"]*1e15, "cpml_cells_per_face": spec["pml_cells"],
                "wall_s": r["median_wall_s"], "wall_samples": len(r["samples_wall_s"]),
                "first_all30": first, "replay_all30": replay,
                "persistent_method_witness_positive": all(x.get("persistent_body_positive") for x in r.get("persistent_method_witnesses", [])) if "persistent_method_witnesses" in r else None,
                "host_inputs_unchanged": True, "timed_worker_imported_JAX": False}
        assert outputs["baseline"]["fixture_sha256"] == outputs["candidate"]["fixture_sha256"]
        actual = outputs["baseline"]["wall_s"]/outputs["candidate"]["wall_s"]
        assert abs(actual-pair["speedup"]) < 1e-12
        pairs.append({"speedup": actual, **outputs})
    speeds = [x["speedup"] for x in pairs]
    assert statistics.median(speeds) == a["median_paired_speedup"]
    complete = all(x["baseline"]["complete_physical_duration"] for x in pairs)
    return {"comparison": str(path), "sha256": sha(path), "reported_scope": a["scope"],
            "audited_scope": "complete30fs-solve" if complete else "256step-prefix-screen-NOT-full-solve",
            "generic_complete_scope_label_is_misleading": not complete and "Complete" in a["scope"],
            "fresh_process_pairs": len(pairs), "speedup": a["median_paired_speedup"],
            "observed_range": a["observed_range"], "range_is_confidence_interval": False,
            "expected_control_sha256": a.get("expected_control_sha256"),
            "control_identity_verified": a.get("control_identity_verified"), "pairs": pairs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--area", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args();a.output.mkdir(parents=True, exist_ok=True)
    area = a.area
    full = [comparison(area/name/"prefix-comparison.json") for name in (
        "source-uniform-profiles-full512", "source-uniform-profiles-full564")]
    cumulative = [comparison(q) for q in sorted((area/"cumulative-gpu-sixth").glob("*comparison.json"))]
    nearest = [comparison(q) for q in sorted((area/"nearest3-final-gpu").glob("*comparison.json"))]
    resources = []
    for name in ("source-uniform-profiles-v1", "persistent-specialized-g4-cumulative-v1",
                 "persistent-paired-special-phase-v5-cumulative", "persistent-paired-special-phase-v5-E2-cumulative",
                 "persistent-paired-special-phase-v5-E4-cumulative", "noio-fixed-cubes-cached-v1",
                 "persistent-cumulative-nearest3-v1"):
        folder = area/name; rows = load(folder/"ptxas-resources.json");manifest = load(folder/"manifest.json")
        nonzero = [r for r in rows if any(r.get(k) for k in ("stack", "spill_stores", "spill_loads"))]
        resources.append({"candidate": name, "module_sha256": manifest["module_sha256"],
            "manifest_sha256": sha(folder/"manifest.json"), "ptxas_resource_report_sha256": sha(folder/"ptxas-resources.json"),
            "entries": len(rows), "strict_zero_stack_and_spill": not nonzero,
            "nonzero_entries": nonzero, "explicit_exploratory_budget_bytes": manifest.get("persistent_exploratory_stack_and_spill_budget_bytes"),
            "immutable_compile_manifest_GPU_executed_flag_is_not_runtime_evidence": True})
    rejections = load(area/"resource-rejections.json")["rows"]
    later = []
    for row in rejections:
        matches = [q for q in area.glob("*/*comparison.json") if q.name.startswith(row["candidate"]+"-")]
        if matches:
            later.append({"candidate": row["candidate"], "strict_gate": row["gate"],
                          "later_measurement_paths": [str(q) for q in matches],
                          "interpretation": "Strict rejection describes that admission attempt; later bounded-spill execution must not be erased."})
    validations = []
    for q in sorted((area/"source-uniform-profiles-validation").glob("*.json")):
        if q.name in ("plan.json", "validate-processes.json"):continue
        r = load(q)
        clock_checks = r.get("checks", [])
        passed = r.get("passed")
        if q.name == "clock13.json":
            passed = r.get("status") == "ok" and len(clock_checks) == 13 and all(
                x.get("passed") and x.get("all30_exact") and x.get("elapsed_exact") and x.get("origin_exact") for x in clock_checks)
        assert passed is True
        validations.append({"path": str(q), "sha256": sha(q), "passed": r.get("passed"),
                            "cases": len(r.get("cases", [])), "schema": r.get("schema"),
                            "top_level_passed_absent": "passed" not in r,
                            "audited_passed": passed, "clock_checks": len(clock_checks)})
    sanitizer_process = load(area/"validation-gpu-third/processes.json")
    sanitizer_executions = [{"label": r["label"], "returncode": r["returncode"], "command_prefix": r["command"][:5],
                            "wall_s": r["wall_s"]} for r in sanitizer_process["records"] if r["label"].startswith("combined-")]
    assert len(sanitizer_executions) == 4 and all(r["returncode"] == 0 for r in sanitizer_executions)
    sanitizer_jsons = [{"path": str(q), "sha256": sha(q), "passed": load(q).get("passed")}
                       for q in sorted((area/"validation-gpu-third").glob("combined-*.json"))]
    historical = load(Path(".cache/main-vs-temporal-2026-11-12/summary.json"))
    historical_provenance = load(Path(".cache/main-vs-temporal-2026-11-12/provenance.json"))
    pending = []
    for name in ("fixed-cube-full512", "fixed-cube-full564", "final-direct-full512", "final-direct-full564"):
        folder = area/name; results = sorted(folder.glob("*comparison.json"))
        pending.append({"campaign": name, "comparison_present_at_snapshot": bool(results),
                        "interpretation": "Audit separately when complete; no speed claim from plan or partial log alone.",
                        "completed_results": [comparison(q) for q in results]})
    result = {"schema": "beamz.final-independent-research-evidence-audit.v1",
        "audit_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result": "Audited existing measurements support modest full-solve gain; new cumulative persistent screens did not beat current control. No2x or several-fold additional improvement established.",
        "current_control": {"sha256": load(area/"control-contract.json")["control_sha256"],
            "identity": "Current research selected52865, NOT stock main", "hardware": "Local RTX3090,250W,FP32,CPML12/face", "remote_compute": False},
        "complete_full_ABBA": full, "cumulative_prefix_screens": cumulative,
        "nearest3_prefix_screens": nearest,
        "resource_admission": resources,
        "strict_rejection_rows": {"path": str(area/"resource-rejections.json"), "sha256": sha(area/"resource-rejections.json"),
            "row_count": len(rejections), "never_infer_measured_slowdown_from_compiler_rejection": True,
            "strict_rejection_names_later_measured_under_other_admission": later},
        "best5fb_validation_reports": validations, "best5fb_independent_profiles_and_sanitizer_JSON": sanitizer_jsons,
        "sanitizer_process_ledger_path": str(area/"validation-gpu-third/processes.json"),
        "sanitizer_process_ledger_sha256": sha(area/"validation-gpu-third/processes.json"),
        "sanitizer_and_independent_profile_process_execution": sanitizer_executions,
        "historical_main_context": {"main_commit": historical["main_commit"],
            "recorded_utc": historical["finished_utc"], "source_provenance": historical_provenance["source"],
            "then_default_main_s": historical["rows"][0]["warmed_s"], "then_explicit_main_mstep_s": historical["rows"][1]["warmed_s"],
            "then_research_s": historical["rows"][2]["warmed_s"],
            "then_research_module_sha256": historical["rows"][2]["module_sha256"],
            "then_speedup_vs_default": historical["temporal_speedup_vs_main_default_cuda"],
            "then_speedup_vs_mstep": historical["temporal_speedup_vs_main_explicit_mstep"],
            "limit": "Earlier one-repeat comparison against pinned main, different research binary and date. No fresh paired current5fb-vs-main run here; multiplying ratios is not a new controlled measurement."},
        "startup_zero": "Phasev3/startup-prev WaitProducer(step0) could return successful ready0; if(!seen) incorrectly skipped remaining parent tags. Actual extracted CPU witness reproduced it. Fixedv4 checks abort_local and waits all tags; independent delayed5ms/10ms witness passed. Do not promote v3 merely because selected GPU inputs passed.",
        "formal_limits": [
            "CPU owner/ring/controller proofs certify documented assumptions and exact source bindings; not exhaustive numerical or hardware race proof",
            "Exact all30 first/replay and nonzero DFT witnesses support schedule equivalence; do not establish analytic Mie scattering accuracy or reproduce an unpublished creator benchmark",
            "Mixed-cache stress660000 checks and cumulative stress4x10240000 checks passed; finite stress does not replace PTX causality proof or certify universal scheduling",
            "Cumulative PTX proof allows per-writer fence removal only with complete producer/consumer joins, strong release/acquire, exact epoch/reset and full WAR retirement",
            "Small-fixture sanitizers are bounded coverage; racecheck is not a general proof for all cross-CTA global accesses",
            "ABBA2 pairs give observed range, not confidence interval; prefixes use one fresh pair with three inner samples",
            "Prepared warmed solver timings exclude frontend/compile/upload/readback; current native workers did not import JAX",
            "Worktree contains extensive pre-existing production/research changes. This audit is research evidence review, not a whole-tree PR approval or production installation certification"],
        "pending_fixed_cube_campaigns": pending,
        "reviewer_GPU_jobs": 0, "reviewer_production_or_report_edits": 0}
    scoped_path = area / "scoped-final-gpu/processes.json"
    scoped = load(scoped_path)["records"] if scoped_path.exists() else []
    for record in scoped:
        if record.get("clean_in_scope"):
            assert record["returncode"] == 0 and record["all30_helper_passed"] and record["explicit_zero_summary"]
            assert record["oracle_manifest_unchanged"]
    result["fixed_cube_scoped_sanitizers"] = {
        "path": str(scoped_path), "sha256": sha(scoped_path) if scoped_path.exists() else None,
        "clean_in_scope": sum(bool(r.get("clean_in_scope")) for r in scoped),
        "scope": "Exact actual-shape/depth kernel, requested one matching direct launch; reference Program prepared separately; all30 finite/bitwise checked; not whole-grid or graph-mode sanitizer coverage",
        "records": scoped,
    }
    result["audit_refresh"] = "Root refreshed the independent review script with final result paths; script performs no GPU work. Original independent audit snapshot remains separate."
    result["hardware_stress"] = [{"path": str(area/name/"result.json"), "sha256": sha(area/name/"result.json"),
                                  "result": load(area/name/"result.json")}
                                 for name in ("mixed-psi-stress", "fence-coalescing-stress")]
    out = a.output/"audit.json";out.write_text(json.dumps(result, indent=2)+"\n")
    lines = ["Independent evidence audit", "", result["result"], "",
             "| Evidence | Audited scope | Bound control | Speedup |", "|---|---|---|---:|"]
    completed_fixed = [r for campaign in pending for r in campaign["completed_results"]]
    for r in full+cumulative+nearest+completed_fixed:
        lines.append(f"| {Path(r['comparison']).parent.name}/{Path(r['comparison']).name} | {r['audited_scope']} | {(r['expected_control_sha256'] or 'unspecified')[:12]} | {r['speedup']:.6f}× |")
    lines += ["", "5fb full solves:512³/12347steps and564³/13601steps,680nm including12 CPML cells per face,FP32,30fs. Two fresh-process ABBA pairs each; all30 outputs and replay are bitwise exact. Observed ranges are not confidence intervals.",
        "", "Cumulative rank reports use256 executed steps despite generic ‘Complete identical physics’ labels. All four are measured performance regressions, with correct first/replay outputs. Their bounded8/16-byte spill admission is exploratory;5fb has62 strict-zero entries.",
        "", "Compiler rejection is not measured slowdown. The strict rejection ledger contains at least one name later timed under bounded-spill admission; classify by exact module SHA and admission attempt.",
        "", "Historical main means pinned d5282752 from the prior day. Its275.3076s default/195.6657s explicit Mstep results were compared with a different155.6933s research binary, not current5fb. No fresh current-vs-main comparison was run here.",
        "", "Startup-zero defect was reproduced and fixed: ready0 is valid success; abort_local determines failure. Phasev3 remains unsuitable for promotion. V4 waits all delayed parent tags.",
        "", "Fixed-cube full campaigns are recorded only if completed comparison JSON exists at this audit snapshot; plans and partial logs establish no outcome.",
        "", "PTX proofs, finite GPU stress, bitwise equivalence and small sanitizers have distinct scopes. They do not prove analytic Mie accuracy, universal hardware scheduling, or equivalence to unpublished FDTDX rewrite results.",
        "", "See audit.json for source/report/module hashes and detailed gates. Reviewer ran no GPU jobs and changed no production files or primary report."]
    (a.output/"audit.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output": str(out), "full_comparisons": len(full), "cumulative_screens": len(cumulative),
                      "misleading_prefix_scope_labels": sum(r["generic_complete_scope_label_is_misleading"] for r in cumulative)}))


if __name__ == "__main__":
    main()
