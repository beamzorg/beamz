"""Compare full notebook outputs, physical state and measured execution costs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compare_arrays(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(
            shape_dtype_match=False,
            shape_a=list(a.shape),
            shape_b=list(b.shape),
            dtype_a=str(a.dtype),
            dtype_b=str(b.dtype),
        )
    exact = True
    max_abs = 0.0
    peak = 0.0
    sum_diff = 0.0
    sum_ref = 0.0
    finite = True
    for start in range(0, a.size, 1 << 20):
        x = a.reshape(-1)[start : start + (1 << 20)]
        y = b.reshape(-1)[start : start + (1 << 20)]
        exact &= np.array_equal(x.view(np.uint8), y.view(np.uint8))
        finite &= bool(np.isfinite(x).all() and np.isfinite(y).all())
        dx = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)
        dy = y.astype(dx.dtype)
        d = np.abs(dy - dx)
        if d.size:
            max_abs = max(max_abs, float(d.max()))
            peak = max(peak, float(np.abs(dx).max()))
        sum_diff += float(np.sum(d * d))
        sum_ref += float(np.sum(np.abs(dx) ** 2))
    return dict(
        shape_dtype_match=True,
        shape=list(a.shape),
        dtype=str(a.dtype),
        finite=finite,
        bitwise_exact=bool(exact),
        max_abs_error=max_abs,
        reference_peak=peak,
        peak_scaled_max_error=max_abs / max(peak, 1e-300),
        relative_l2=(sum_diff / max(sum_ref, 1e-300)) ** 0.5,
    )


def archive(a, b):
    from scripts.cuda_fixture_store import ArrayReader, drop_pages

    def opened(path):
        if path.is_dir():
            reader = ArrayReader(path)
            return reader, list(reader.metadata)
        reader = np.load(path)
        return reader, list(reader.files)

    x, names = opened(a)
    y, other = opened(b)
    if sorted(names) != sorted(other):
        raise ValueError("Archive leaves differ")
    rows = {}
    for name in names:
        left, right = x[name], y[name]
        rows[name] = compare_arrays(left, right)
        drop_pages(left)
        drop_pages(right)
        del left, right
    if hasattr(x, "close"):
        x.close()
    if hasattr(y, "close"):
        y.close()
    return dict(
        leaves=rows,
        leaf_count=len(rows),
        all_bitwise=all(r.get("bitwise_exact", False) for r in rows.values()),
        all_finite=all(r.get("finite", False) for r in rows.values()),
    )


def unavailable(area, main, candidate):
    a = json.loads((main / "results.json").read_text())
    execution = json.loads((main / "execution.json").read_text())
    import nbformat

    n = nbformat.read(
        Path(execution["root"]) / "examples/notebooks/cosine_waveguide_crossing.ipynb",
        as_version=4,
    )
    end_cell = len(n.cells) + 1
    measured = dict(
        notebook_total_s=next(
            x["start_s"] for x in execution["cell_starts"] if x["index"] == end_cell
        ),
        first_public_s=a["runs"][0]["public_wall_s"],
        first_engine_s=a["runs"][0]["engine_s"],
        first_other_public_s=a["runs"][0]["public_wall_s"] - a["runs"][0]["engine_s"],
        warm_public_s=[x["public_wall_s"] for x in a["runs"][1:]],
        warm_engine_s=[x["engine_s"] for x in a["runs"][1:]],
        warm_public_median_s=float(
            np.median([x["public_wall_s"] for x in a["runs"][1:]])
        ),
        warm_engine_median_s=float(np.median([x["engine_s"] for x in a["runs"][1:]])),
        memory=a["memory"],
    )
    attempts = []
    for name in ("bounded-run", "bounded-run-v2", "bounded-run-v3"):
        path = area / name
        guard = path / "host-memory-guard.json"
        attempts.append(
            dict(
                directory=str(path),
                completed=False,
                reason="host_memory_guard"
                if guard.exists()
                else "kernel_host_RAM_OOM_kill",
                guard=json.loads(guard.read_text()) if guard.exists() else None,
            )
        )
    result = dict(
        schema="beamz.cosine-cuda-stabilization.v1",
        status="native_comparison_unavailable",
        scope="Original full cosine notebook:1ps,101wavelengths,49million cells,FP32,CPML12; local RTX3090 at250W.",
        provenance=json.loads((area / "provenance.json").read_text()),
        main=measured,
        main_notebook=str(main / "cosine_waveguide_crossing.ipynb"),
        main_notebook_completed=True,
        main_state_validation=json.loads(
            (area / "main-state-validation.json").read_text()
        ),
        main_spectrum=json.loads((area / "main-spectrum-summary.json").read_text()),
        main_three_repeats_field_DFT_bitwise=True,
        grid_shape_xyz=a["grid_shape"],
        steps=a["steps"],
        metric_kind=a["metric_kind"],
        native=dict(
            completed=False,
            solver_executed=False,
            temporal_executed=False,
            notebook_total_s=None,
            engine_s=None,
            accuracy_vs_main=None,
            compatibility="Frozen moving engine requires metric_kind0 (uniform isotropic Cartesian); notebook resolves rectilinear (metric_kind2).",
            fallback="Explicit ordinary-native prototype attempted; frontend source/monitor preparation exceeded available host RAM before reaching its solver.",
            attempts=attempts,
            placement_correction="Notebook already resolved to default GPU preparation; explicit default request did not change effective placement.",
        ),
        memory_definition="NVML sampled per-process GPU residency (including context and JAX pool), host RSS,50ms interval. Main peak spans cold execution and three repeats. Native partial failures do not establish a completed-run peak.",
        accuracy_definition="No native output exists to compare. All30 main state leaves and10 main observable exports are finite; three repeats match original raw field DFTs bitwise.",
        source_diagnosis="Preparation trace stops inside sources_monitors_and_layout. Rectilinear local_3d_phasor_context returns None, retaining full-volume complex scratch in launch diagnostics. This code path and host OOM evidence motivate bounded source preparation; no new source-calibration algorithm was validated.",
        next_required=[
            "Support graded derivative metrics in the traced temporal kernels.",
            "Bound rectilinear mode-source preparation without changing launch calibration or the mesh.",
        ],
        speedup=None,
        comparison_valid=False,
        RunPod=False,
        installed_default_changed=False,
    )
    (area / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], main=measured)))
    return result


def summarize(area, candidate_name="bounded-run-v3"):
    area = area.resolve()
    main = area / "main-run"
    bounded = area / candidate_name
    if not (bounded / "results.json").exists():
        return unavailable(area, main, bounded)
    a = json.loads((main / "results.json").read_text())
    b = json.loads((bounded / "results.json").read_text())
    ea = json.loads((main / "execution.json").read_text())
    eb = json.loads((bounded / "execution.json").read_text())
    assert ea["notebook_sha256"] == eb["notebook_sha256"]
    assert (
        a["grid_shape"] == b["grid_shape"]
        and a["steps"] == b["steps"]
        and a["frequencies"] == b["frequencies"]
    )
    outputs = archive(main / "arrays.npz", bounded / "arrays.npz")
    state = archive(
        main / "state" if (main / "state").is_dir() else main / "state.npz",
        bounded / "state" if (bounded / "state").is_dir() else bounded / "state.npz",
    )
    # The appended export cell starts after the original notebook. Its driver
    # timestamp includes kernel startup/imports and excludes exports and repeats.
    import nbformat

    original = nbformat.read(
        ea.get(
            "notebook_path",
            str(
                Path(ea["root"]) / "examples/notebooks/cosine_waveguide_crossing.ipynb"
            ),
        ),
        as_version=4,
    )
    end_cell = len(original.cells) + 1
    rows = []
    for trial, execution in ((a, ea), (b, eb)):
        warm = trial["runs"][1:]
        rows.append(
            dict(
                variant=trial["variant"],
                notebook_total_s=next(
                    r["start_s"]
                    for r in execution["cell_starts"]
                    if r["index"] == end_cell
                ),
                notebook_body_s=trial["notebook_wall_s"],
                first_public_s=trial["runs"][0]["public_wall_s"],
                first_engine_s=trial["runs"][0]["engine_s"],
                warm_public_s=[r["public_wall_s"] for r in warm],
                warm_engine_s=[r["engine_s"] for r in warm],
                warm_public_median_s=float(
                    np.median([r["public_wall_s"] for r in warm])
                ),
                warm_engine_median_s=float(np.median([r["engine_s"] for r in warm])),
                memory=trial["memory"],
                cold_notebook_memory=trial.get("cold_notebook_memory", trial["memory"]),
                monitor_decode_s=trial["runs"][0].get("monitor_decode_s"),
                result_build_s=trial["runs"][0].get("result_build_s"),
            )
        )
    result = dict(
        scope="Regular-grid cosine crossing: 30 nm, 1 ps, 101 wavelengths, CPML12 on every face, three-frequency broadband mode source, FP32. Default origin/main CUDA versus frozen bounded temporal CUDA with an explicit adapter.",
        provenance=json.loads((area / "provenance.json").read_text()),
        rows=rows,
        grid_shape=a["grid_shape"],
        steps=a["steps"],
        metric_kind=a["metric_kind"],
        outputs=outputs,
        state=state,
        notebook_speedup=rows[0]["notebook_total_s"] / rows[1]["notebook_total_s"],
        warm_engine_speedup=rows[0]["warm_engine_median_s"]
        / rows[1]["warm_engine_median_s"],
        warm_public_speedup=rows[0]["warm_public_median_s"]
        / rows[1]["warm_public_median_s"],
        memory_definition="50ms NVML per-process resident GPU memory, including JAX pool, native allocations and context; unrelated desktop/ComfyUI excluded. Host RAM via RSS.",
        engine_definition="Both engines use synchronized host-wall execution time, including field updates, CPML, sources, DFT and clock advance; setup and result decoding are excluded. Native CUDA-event timing is retained separately. Cold native wall time can include first graph construction; warm repeats reduce preparation differences.",
        temporal_active=b.get("temporal_active"),
        fallback_reason=b.get("fallback_reason"),
        candidate_directory=str(bounded),
        accuracy_definition="Agreement with origin/main on the identical discretization, not an independent physical convergence study.",
    )
    (area / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for path, label, style in (
        (main, "origin/main", "-"),
        (bounded, "Bounded CUDA", "--"),
    ):
        with np.load(path / "arrays.npz") as z:
            wavelength = 299792458 / z["freqs"] * 1e6
            axes[0].plot(
                wavelength,
                10 * np.log10(np.maximum(z["through"], 1e-30)),
                style,
                label=label,
            )
            axes[1].plot(
                wavelength,
                10
                * np.log10(
                    np.maximum(z["cross"] / np.maximum(z["through"], 1e-30), 1e-30)
                ),
                style,
                label=label,
            )
    for ax, title, ylabel in zip(
        axes,
        ("Through transmission", "Cross-port crosstalk"),
        ("Transmission (dB)", "Cross / through (dB)"),
        strict=True,
    ):
        ax.set(xlabel="Wavelength (µm)", ylabel=ylabel, title=title)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(area / "accuracy.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                notebook_speedup=result["notebook_speedup"],
                warm_engine_speedup=result["warm_engine_speedup"],
                outputs_bitwise=outputs["all_bitwise"],
                state_bitwise=state["all_bitwise"],
                rows=rows,
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--area", type=Path, required=True)
    p.add_argument("--candidate-name", default="bounded-run-v3")
    a = p.parse_args()
    summarize(a.area, a.candidate_name)
