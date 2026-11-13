#!/usr/bin/env python3
"""Adapter-configured common discrete data for released FDTDX 0.6.3.

Uses released run_fdtd/update/curl unchanged. Only Source/Detector extension
objects and public material/PML coefficient fields are configured here.
Importing this module never initializes JAX or a GPU.

Contract data is NPZ with a JSON manifest: shape_zyx, dx_m, dt_s, steps,
frequency_hz, epsilon mapping Ex/Ey/Ez to array keys, source term descriptors,
monitor gather descriptors, and cpml descriptors. All indices refer to complete
BeamZ z/y/x component supports; conversion rejects absent terminal-node reads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def verify_common_data(contract, data):
    """Fail closed on altered data or canonical work labels before setup/timing."""
    if contract.get("schema") != "beamz.fdtdx-common-discrete.v1":
        raise ValueError("Unsupported common discrete contract schema")
    recorded = contract["array_sha256"]
    if set(recorded) != set(data):
        raise ValueError("Canonical archive array names differ from declared hashes")
    actual = {key: array_digest(data[key]) for key in recorded}
    if actual != recorded:
        raise ValueError("Canonical array content hash mismatch")
    groups = {
        "electric_coefficients": list(contract["inv_epsilon"].values()),
        "sources": [s[k] for s in contract["sources"] for k in ("coeff", "waveform")],
        "monitors": [
            v[k]
            for m in contract["monitors"]
            for v in m["components"].values()
            for k in ("indices", "weights")
        ],
        "cpml": [
            v
            for a in contract["cpml"].values()
            for s in a.values()
            for f in s.values()
            for v in f.values()
        ],
    }
    hashes = {
        name: hashlib.sha256(
            json.dumps(
                {key: actual[key] for key in sorted(names)}, sort_keys=True
            ).encode()
        ).hexdigest()
        for name, names in groups.items()
    }
    if hashes != contract["canonical"]["data_hashes"]:
        raise ValueError(
            "Canonical coefficient/source/monitor/CPML group hash mismatch"
        )
    points = []
    for monitor in contract["monitors"]:
        if set(monitor["components"]) != set(COMPONENTS):
            raise ValueError("Every common face must sample all six field components")
        _, _, indices, _ = monitor_gathers(monitor, data, contract["shape_zyx"])
        points.append(indices.shape[-1])
        if (
            monitor.get("points") != indices.shape[-1]
            or monitor.get("record_interval", 1) != 1
        ):
            raise ValueError(
                "Declared monitor point count or recording interval differs"
            )
        if monitor.get("component_mask", [1] * 6) != [1] * 6:
            raise ValueError("Common comparison cannot omit detector components")
    expected = {
        "comparison_mode": "adapter-configured common discrete data; released kernels unchanged",
        "shape_zyx": list(contract["shape_zyx"]),
        "dx_m": float(contract["dx_m"]),
        "dt_s": float(contract["dt_s"]),
        "steps": int(contract["steps"]),
        "pml_cells_each_face": 12,
        "field_dtype": "float32",
        "frequency_hz": float(contract["frequency_hz"]),
        "source_terms": len(contract["sources"]),
        "detector_points": points,
        "detector_components": list(COMPONENTS),
        "dft_interval": 1,
        "dft_complex_accumulations": int(contract["steps"]) * 6 * sum(points),
        "detector_phase": "exp(+i*omega*(n+1)*dt), all six; no H time averaging",
        "clock_map": "FD H[n]=BeamZ H[n]; FD E[n]=BeamZ E[n-1]; E[-1]=0",
        "data_hashes": hashes,
        "gpu_count": 1,
        "gradients": False,
    }
    if expected != contract["canonical"]:
        raise ValueError("Canonical work labels disagree with resolved shared arrays")
    return expected


def component_shape(shape_zyx, component):
    shape = list(map(int, shape_zyx))
    offsets = {
        "Ex": (0, 0, 0.5),
        "Ey": (0, 0.5, 0),
        "Ez": (0.5, 0, 0),
        "Hx": (0.5, 0.5, 0),
        "Hy": (0.5, 0, 0.5),
        "Hz": (0, 0.5, 0.5),
    }[component]
    return tuple(n + int(offset == 0) for n, offset in zip(shape, offsets, strict=True))


def source_coordinates(term, data, shape_zyx):
    """Return global xyz source coordinates and coefficients, with no fitting."""
    if "indices_zyx" in term:
        zyx = np.asarray(data[term["indices_zyx"]], dtype=np.int64)
        if zyx.shape[0] != 3:
            raise ValueError("Source coordinates must have shape (3, points)")
    else:
        starts, sizes = term["slab_starts"], term["slab_sizes"]
        zyx = np.stack(
            np.meshgrid(
                *[np.arange(s, s + n) for s, n in zip(starts, sizes, strict=True)],
                indexing="ij",
            )
        )
        zyx = zyx.reshape(3, -1)
    if np.any(zyx < 0) or np.any(zyx >= np.asarray(shape_zyx)[:, None]):
        raise ValueError("Source touches absent FDTDX terminal field nodes")
    if np.iscomplexobj(data[term["coeff"]]):
        raise ValueError("Common source coefficients must be real update terms")
    coeff = np.asarray(data[term["coeff"]], dtype=np.float32).reshape(-1)
    if coeff.size != zyx.shape[1]:
        raise ValueError("Source coefficient/coordinate count differs")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("Nonfinite common source coefficient")
    return zyx[::-1].astype(np.int32), coeff


def monitor_gathers(monitor, data, shape_zyx):
    """Convert common full-support weighted gathers into a minimal local region."""
    coords, weights = [], []
    for component in COMPONENTS:
        item = monitor["components"][component]
        indices = np.asarray(data[item["indices"]], dtype=np.int64)
        weight = np.asarray(data[item["weights"]], dtype=np.float32)
        layout = item.get("layout", "point_neighbor")
        if layout == "point_neighbor":
            indices, weight = indices.T, weight.T
        elif layout != "neighbor_point":
            raise ValueError(f"Unknown monitor gather layout: {layout}")
        if indices.shape != weight.shape or indices.ndim != 2:
            raise ValueError("Monitor indices/weights must share (neighbors, points)")
        if not np.all(np.isfinite(weight)):
            raise ValueError("Nonfinite monitor weights")
        active = weight != 0
        indices = np.where(active, indices, 0)
        zyx = np.stack(np.unravel_index(indices, component_shape(shape_zyx, component)))
        if np.any(zyx[:, active] >= np.asarray(shape_zyx)[:, None]):
            raise ValueError("Monitor reads absent FDTDX terminal field nodes")
        # Zero-weight padding must use legal indices and cannot inflate the
        # detector's bounding box to include an unrelated origin cell.
        if not np.any(active):
            raise ValueError("All-zero component monitor interpolation")
        zyx[:, ~active] = zyx[:, active][:, :1]
        coords.append(zyx[::-1])
        weights.append(weight)
    all_coords = np.concatenate([xyz.reshape(3, -1) for xyz in coords], axis=1)
    lower = all_coords.min(axis=1)
    upper = all_coords.max(axis=1) + 1
    shape_xyz = tuple(upper - lower)
    point_counts = {weight.shape[1] for weight in weights}
    if len(point_counts) != 1:
        raise ValueError("Monitor component point counts differ")
    neighbors = max(weight.shape[0] for weight in weights)
    points = point_counts.pop()
    local_indices = np.zeros((6, neighbors, points), np.int32)
    local_weights = np.zeros((6, neighbors, points), np.float32)
    for c, (xyz, weight) in enumerate(zip(coords, weights, strict=True)):
        local_indices[c, : len(weight)] = np.ravel_multi_index(
            tuple(xyz - lower[:, None, None]), shape_xyz
        )
        local_weights[c, : len(weight)] = weight
    return tuple(map(int, lower)), tuple(map(int, upper)), local_indices, local_weights


def _extension_classes(jax, jnp):
    from fdtdx.core.jax.pytrees import autoinit, field, frozen_field
    from fdtdx.objects.detectors.detector import Detector
    from fdtdx.objects.sources.source import Source

    @autoinit
    class CommonSource(Source):
        indices: jax.Array = field()
        coefficients: jax.Array = field()
        samples: jax.Array = field()
        component: int = frozen_field()
        electric: bool = frozen_field()
        slab_starts: tuple[int, int, int] | None = frozen_field(default=None)
        slab_sizes: tuple[int, int, int] | None = frozen_field(default=None)

        def _inject(self, values, time_step, inverse):
            # E-first release step n computes common E_n; H computes H_(n+.5).
            # The released H update deliberately calls sources at n+.5;
            # exported common waveforms already contain each phase's physical
            # sampling offset, so recover the integer canonical phase counter.
            phase_step = jnp.floor(time_step).astype(jnp.int32)
            sample_index = phase_step - int(self.electric)
            amplitude = jnp.where(
                sample_index >= 0, self.samples[jnp.maximum(sample_index, 0)], 0
            )
            delta = self.coefficients * amplitude * (-1 if inverse else 1)
            if self.slab_starts is not None:
                region = tuple(
                    slice(start, start + size)
                    for start, size in zip(
                        self.slab_starts, self.slab_sizes, strict=True
                    )
                )
                return values.at[self.component, *region].add(delta)
            return values.at[self.component, *self.indices].add(delta)

        def update_E(
            self, E, inv_permittivities, inv_permeabilities, time_step, inverse
        ):
            del inv_permittivities, inv_permeabilities
            return self._inject(E, time_step, inverse) if self.electric else E

        def update_H(
            self, H, inv_permittivities, inv_permeabilities, time_step, inverse
        ):
            del inv_permittivities, inv_permeabilities
            return H if self.electric else self._inject(H, time_step, inverse)

    @autoinit
    class CommonDetector(Detector):
        gather_indices: jax.Array = field()
        gather_weights: jax.Array = field()
        frequency_hz: float = frozen_field()
        point_count: int = frozen_field()

        def _num_latent_time_steps(self):
            return 1

        def _shape_dtype_single_time_step(self):
            return {
                "phasor": jax.ShapeDtypeStruct((6, self.point_count), jnp.complex64)
            }

        def update(self, time_step, E, H, state, inv_permittivity, inv_permeability):
            del inv_permittivity, inv_permeability
            fields = jnp.concatenate((E, H), axis=0).reshape(6, -1)
            gathered = jax.vmap(lambda f, idx: f[idx])(fields, self.gather_indices)
            samples = jnp.sum(gathered * self.gather_weights, axis=1)
            time = (time_step + 1).astype(jnp.float32) * self._config.time_step_duration
            phase = jnp.exp(2j * jnp.pi * self.frequency_hz * time)
            return {"phasor": state["phasor"] + (samples * phase)[None]}

    return CommonSource, CommonDetector


def build_case(contract, data, *, backend="gpu", verified=None):
    """Configure released kernels using shared immutable discrete arrays."""
    import fdtdx
    import jax
    import jax.numpy as jnp
    from fdtdx.constants import eta0

    verified = verify_common_data(contract, data) if verified is None else verified
    if importlib.metadata.version("fdtdx") != "0.6.3":
        raise ValueError("This audited adapter admits only fdtdx==0.6.3")
    shape_zyx = tuple(map(int, contract["shape_zyx"]))
    if len(set(shape_zyx)) != 1 or contract["pml_cells"] != 12:
        raise ValueError("Expected cubic common workload with exactly CPML12")
    if len(contract["monitors"]) != 6:
        raise ValueError("Common comparison requires six six-component DFT faces")
    dx, dt, steps = (
        float(contract["dx_m"]),
        float(contract["dt_s"]),
        int(contract["steps"]),
    )
    config = fdtdx.SimulationConfig(
        time=steps * dt,
        grid=fdtdx.UniformGrid(spacing=dx),
        courant_factor=0.95,
        backend=backend,
        dtype=jnp.float32,
        gradient_config=None,
        symmetry=(0, 0, 0),
    )
    if config.backend != backend:
        raise RuntimeError(f"Requested {backend}, resolved {config.backend}")
    if config.time_steps_total != steps or not np.isclose(
        config.time_step_duration, dt, rtol=1e-12, atol=0
    ):
        raise ValueError("Resolved FDTDX timestep/step count differs from contract")
    Source, Detector = _extension_classes(jax, jnp)
    volume = fdtdx.SimulationVolume(
        name="common_volume", partial_grid_shape=shape_zyx[::-1]
    )
    bounds, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="pml", thickness=12),
        volume,
    )
    objects_list = [volume, *bounds.values()]
    descriptors = []
    frequency = float(contract["frequency_hz"])
    for i, term in enumerate(contract["sources"]):
        component = term["component"]
        if term["timing"] not in ("h", "e") or term["timing"] != component[0].lower():
            raise ValueError(
                "Common adapter currently requires canonical h/e source phases"
            )
        xyz, coefficients = source_coordinates(term, data, shape_zyx)
        electric = component.startswith("E")
        if not electric:
            coefficients = coefficients * np.float32(eta0)
        samples = np.asarray(data[term["waveform"]], np.float32)
        if samples.shape != (steps,):
            raise ValueError(
                "Source waveform must contain exactly common_steps samples"
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("Nonfinite common source waveform")
        slab_starts, slab_sizes = None, None
        source_indices = xyz
        if "slab_starts" in term:
            slab_starts = tuple(map(int, term["slab_starts"][::-1]))
            slab_sizes = tuple(map(int, term["slab_sizes"][::-1]))
            coefficients = coefficients.reshape(term["slab_sizes"]).transpose(2, 1, 0)
            source_indices = np.zeros((3, 0), np.int32)
        lo = xyz.min(axis=1)
        hi = xyz.max(axis=1) + 1
        source = Source(
            name=f"common_source_{i}",
            wave_character=fdtdx.WaveCharacter(frequency=frequency),
            indices=jnp.asarray(source_indices),
            coefficients=jnp.asarray(coefficients),
            samples=jnp.asarray(samples),
            component=COMPONENTS.index(component) % 3,
            electric=electric,
            slab_starts=slab_starts,
            slab_sizes=slab_sizes,
            partial_grid_shape=tuple(map(int, hi - lo)),
        )
        if not np.array_equal(
            np.asarray(source.coefficients), coefficients
        ) or not np.array_equal(np.asarray(source.samples), samples):
            raise ValueError("Configured source differs from canonical update data")
        objects_list.append(source)
        constraints.append(
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(-1, -1, -1),
                other_positions=(-1, -1, -1),
                grid_margins=tuple(map(int, lo)),
            )
        )
    for monitor in contract["monitors"]:
        lo, hi, indices, weights = monitor_gathers(monitor, data, shape_zyx)
        detector = Detector(
            name=monitor["name"],
            exact_interpolation=False,
            plot=False,
            dtype=jnp.complex64,
            gather_indices=jnp.asarray(indices),
            gather_weights=jnp.asarray(weights),
            frequency_hz=frequency,
            point_count=indices.shape[-1],
            partial_grid_shape=tuple(b - a for a, b in zip(lo, hi, strict=True)),
        )
        if not np.array_equal(
            np.asarray(detector.gather_indices), indices
        ) or not np.array_equal(np.asarray(detector.gather_weights), weights):
            raise ValueError(
                "Configured detector gather differs from canonical sampling data"
            )
        objects_list.append(detector)
        constraints.append(
            detector.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(-1, -1, -1),
                other_positions=(-1, -1, -1),
                grid_margins=lo,
            )
        )
        descriptors.append(
            {
                "name": monitor["name"],
                "region_xyz": [lo, hi],
                "points": indices.shape[-1],
            }
        )
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, key = fdtdx.place_objects(
        object_list=objects_list, config=config, constraints=constraints, key=key
    )
    if (
        tuple(objects.volume.grid_shape) != shape_zyx[::-1]
        or tuple(arrays.fields.E.shape) != (3, *shape_zyx[::-1])
        or tuple(arrays.fields.H.shape) != tuple(arrays.fields.E.shape)
    ):
        raise ValueError(
            "Actual release field/volume shape differs from canonical work"
        )
    if (
        len(objects.sources) != len(contract["sources"])
        or len(objects.detectors) != 6
        or len(objects.pml_objects) != 6
    ):
        raise ValueError("Actual release source/detector/CPML object counts differ")
    for detector, descriptor in zip(objects.detectors, descriptors, strict=True):
        expected_region = tuple(zip(*descriptor["region_xyz"], strict=True))
        if tuple(detector.grid_slice_tuple) != expected_region:
            raise ValueError(
                "Placed detector region differs from canonical gather coordinates"
            )
        if not np.all(np.asarray(detector._is_on_at_time_step_arr)):
            raise ValueError("A common detector is not active on every timestep")
    # Source extension has no solved profile to apply; its common data is final.
    epsilon = np.stack(
        [
            np.asarray(data[contract["epsilon"][c]], np.float32)[
                : shape_zyx[0], : shape_zyx[1], : shape_zyx[2]
            ].transpose(2, 1, 0)
            for c in COMPONENTS[:3]
        ]
    )
    if np.any(epsilon <= 0) or not np.all(np.isfinite(epsilon)):
        raise ValueError("Common Yee epsilon must be finite and positive")
    if "inv_epsilon" in contract:
        inverse_epsilon = np.stack(
            [
                np.asarray(data[contract["inv_epsilon"][c]], np.float32)[
                    : shape_zyx[0], : shape_zyx[1], : shape_zyx[2]
                ].transpose(2, 1, 0)
                for c in COMPONENTS[:3]
            ]
        )
    else:
        inverse_epsilon = np.float32(1) / epsilon
    if np.any(inverse_epsilon < 0) or not np.all(np.isfinite(inverse_epsilon)):
        raise ValueError("Common inverse epsilon must be finite and nonnegative")
    arrays = arrays.aset("inv_permittivities", jnp.asarray(inverse_epsilon))
    if not np.array_equal(np.asarray(arrays.inv_permittivities), inverse_epsilon):
        raise ValueError("Configured release inverse epsilon differs from shared data")
    # Fields are dimensionless a/b/inv_kappa arrays on each public PML object.
    configured = []
    for obj in objects.objects:
        if not isinstance(obj, fdtdx.PerfectlyMatchedLayer):
            configured.append(obj)
            continue
        axis = "xyz"[obj.axis]
        side = "low" if obj.direction == "-" else "high"
        spec = contract["cpml"][axis][side]
        for family in ("E", "H"):
            for common, native in (
                ("a", "pml_a_"),
                ("b", "pml_b_"),
                ("inv_kappa", "inv_kappa_"),
            ):
                values = np.asarray(data[spec[family][common]], np.float32).reshape(-1)
                if values.size != 12:
                    raise ValueError(
                        "Export each CPML face profile as exactly12 values"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError("Nonfinite common CPML profile")
                dims = [1, 1, 1]
                dims[obj.axis] = 12
                obj = obj.aset(native + family, jnp.asarray(values.reshape(dims)))
                if not np.array_equal(
                    np.asarray(getattr(obj, native + family)).reshape(-1), values
                ):
                    raise ValueError(
                        "Configured release CPML differs from shared profile"
                    )
        obj = obj.aset("kappa_start", 1.0).aset("kappa_end", 2.0)
        configured.append(obj)
    objects = objects.aset("object_list", configured)

    def run(initial):
        return fdtdx.run_fdtd(
            arrays=initial, objects=objects, config=config, key=key, show_progress=False
        )

    return {
        "arrays": arrays,
        "objects": objects,
        "config": config,
        "key": key,
        "run": run,
        "monitors": descriptors,
        "eta0": float(eta0),
        "params": params,
        "actual_canonical": verified,
        "actual_native": {
            "api": "fdtdx.run_fdtd, published unmodified update/curl",
            "shape_xyz": list(shape_zyx[::-1]),
            "field_shape": [3, *shape_zyx[::-1]],
            "dtype": str(arrays.fields.E.dtype),
            "steps": config.time_steps_total,
            "dt_s": config.time_step_duration,
            "pml_cells_each_face": 12,
            "electric_coefficients": "shared component Yee inverse epsilon including pinned zeros; cropped and transposed"
            if "inv_epsilon" in contract
            else "shared component Yee epsilon, cropped and transposed, reciprocal FP32",
            "magnetic_units": "normalized eta0*H_SI",
            "source_clock": "H[n]=commonH[n]; E[n]=commonE[n-1]; E[-1]=0",
            "final_component_times_dt": {"E": steps - 1, "H": steps - 0.5},
            "detector_phase": "exp(+i*omega*(n+1)*dt), all components",
            "detector_spatial_operator": "shared weighted Yee gathers; no native H time averaging",
            "detector_points": [m["points"] for m in descriptors],
            "source_terms": len(contract["sources"]),
            "gradient_config": None,
            "show_progress": False,
            "coefficient_and_profile_readback": "exact",
        },
    }


def synchronize(value):
    import jax

    return jax.block_until_ready(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-output", "--gate-output", dest="state_output", type=Path
    )
    parser.add_argument(
        "--phase", choices=("validate", "benchmark"), default="validate"
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--public-samples", type=int, default=3)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.backend == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    contract = json.loads(args.contract.read_text())
    data_path = args.data or Path(contract["data_file"])
    if not data_path.is_absolute():
        data_path = args.contract.parent / data_path
    data_sha = hash_file(data_path)
    if contract.get("data_sha256", data_sha) != data_sha:
        raise ValueError("Canonical data file SHA differs from contract")
    with np.load(data_path, allow_pickle=False) as data:
        verified = verify_common_data(contract, data)
    # Match BeamZ's scene-to-first scope: imports/context are process startup,
    # while scene construction/device data placement remain in measured setup.
    import fdtdx
    import jax

    del fdtdx
    if len(jax.devices(args.backend)) != 1:
        raise ValueError("Comparison requires exactly one requested device")
    memory_sampler = None
    started = time.perf_counter()
    with np.load(data_path, allow_pickle=False) as data:
        case = build_case(contract, data, backend=args.backend, verified=verified)
    synchronize(case["arrays"])
    report = {
        "status": "ok",
        "phase": args.phase,
        "engine": {
            "name": "fdtdx",
            "version": "0.6.3",
            "release_commit": "59ec558484d709d511cd70e3a8a45c05e662cfda",
        },
        "backend": args.backend,
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "data_sha256": data_sha,
        "actual_work": {
            "canonical": case["actual_canonical"],
            "native": case["actual_native"],
        },
        "setup_s": time.perf_counter() - started,
        "steps": case["config"].time_steps_total,
        "dt_s": case["config"].time_step_duration,
        "monitors": case["monitors"],
        "adapter_sha256": hash_file(__file__),
    }
    if not args.setup_only:
        import jax

        if args.backend == "gpu":
            from scripts.benchmark_cuda_revision import DeviceMemorySampler

            # Matched scope across engines: all input audits/readbacks and
            # scene setup are complete; capture compilation plus execution.
            memory_sampler = DeviceMemorySampler(1)
        public = jax.jit(case["run"])
        started = time.perf_counter()
        lowered = public.lower(case["arrays"])
        report["lower_s"] = time.perf_counter() - started
        started = time.perf_counter()
        executable = lowered.compile()
        report["compile_s"] = time.perf_counter() - started
        started = time.perf_counter()
        result = synchronize(executable(case["arrays"]))
        report["first_executable_s"] = time.perf_counter() - started
        report["cold_scene_to_first_s"] = (
            report["setup_s"]
            + report["lower_s"]
            + report["compile_s"]
            + report["first_executable_s"]
        )
        durations = []
        count = args.warmups + args.samples if args.phase == "benchmark" else 0
        for sample in range(count):
            result = None
            started = time.perf_counter()
            result = synchronize(executable(case["arrays"]))
            elapsed = time.perf_counter() - started
            if sample >= args.warmups:
                durations.append(elapsed)
        report["warm_executable_s"] = durations
        public_durations = []
        if args.phase == "benchmark":
            # Populate the public jit dispatch cache outside measured calls.
            result = None
            result = synchronize(public(case["arrays"]))
            for _ in range(args.public_samples):
                result = None
                started = time.perf_counter()
                result = synchronize(public(case["arrays"]))
                public_durations.append(time.perf_counter() - started)
        report["timing"] = {
            "warm_solve_samples_s": durations,
            "warm_public_samples_s": public_durations,
            "cold_scene_to_first_s": report["cold_scene_to_first_s"],
            "compile_s": report["compile_s"],
            "lower_s": report["lower_s"],
            "setup_s": report["setup_s"],
            "synchronization": "all_returned_device_leaves",
            "reset_policy": "same_immutable_initial_state_each_call",
            "public_scope": "warmed jitted fdtdx.run_fdtd wrapper, progress and gradients disabled",
        }
        report["memory"] = {
            "device_global": memory_sampler.finish()
            if memory_sampler
            else {"scope": "CPU validation; device sampler disabled"},
            "allocator": jax.devices()[0].memory_stats(),
            "allocator_scope": "Device-lifetime allocator statistics can include scene setup; distinct from sampled compile/execution peak",
        }
        if memory_sampler:
            report["memory"]["device_global"]["scope"] = (
                "Compile + first/warm execution only; excludes scene setup, input audits, and validation export. "
                + report["memory"]["device_global"]["scope"]
            )
        if args.state_output:
            _, output = result
            payload = {
                "E_xyz": np.asarray(output.fields.E),
                "H_si_xyz": np.asarray(output.fields.H) / case["eta0"],
            }
            for monitor in case["monitors"]:
                values = np.array(output.detector_states[monitor["name"]]["phasor"])[0]
                values[3:] /= case["eta0"]
                payload[monitor["name"] + "_phasor_raw"] = values.copy()
                values[:3] *= np.complex64(
                    np.exp(
                        -2j
                        * np.pi
                        * float(contract["frequency_hz"])
                        * float(contract["dt_s"])
                    )
                )
                payload[monitor["name"] + "_phasor"] = values
            args.state_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.state_output, **payload)
    elif memory_sampler:
        report["memory"] = {"device_global": memory_sampler.finish()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
