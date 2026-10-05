"""Two-layer, full-vector 3D inverse design for the issue #240 notebook.

The compiled JAX FDTD transition is differentiated with chunk rematerialization.
All optimization-specific machinery stays here; no alternative Maxwell solver is
introduced. Distances in Config are micrometres; Beamz receives SI units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.ndimage import map_coordinates

import beamz as bz
from beamz.lattice import component_axis_offsets_3d, yee_plane_coordinates_3d
from beamz.optimization.autodiff import transform_density
from beamz.simulation.execute import build_scan, initial_program_state


def gaussian_overlap_power(fields, target_e, weights, impedance):
    """Power in a unit-power, y-polarized +z paraxial target (SI units).

    Fields must already be colocated in space and time. The target has
    Ey=target_e and Hx=-target_e/impedance. Both E and H are required to
    distinguish upward from downward radiation.
    """
    amplitude = (
        jnp.sum(
            (fields[1] / impedance - fields[3]) * jnp.conj(target_e) * weights, axis=-1
        )
        / 4
    )
    return jnp.abs(amplitude) ** 2


def input_mode_diagnostics(fields, spec):
    """Check TE polarization and y parity on the straight-guide calibration plane."""
    fields = np.asarray(fields)
    ny = (
        spec.sample_region.axis_interval("y").stop
        - spec.sample_region.axis_interval("y").start
    )
    ey = fields[1].reshape(spec.freq_count, -1, ny)
    ey_power, ez_power = np.sum(abs(ey) ** 2), np.sum(abs(fields[2]) ** 2)
    return dict(
        te_fraction=float(ey_power / max(ey_power + ez_power, 1e-30)),
        ey_symmetry_error=float(
            np.linalg.norm(ey - ey[:, :, ::-1]) / max(np.linalg.norm(ey), 1e-30)
        ),
    )


@dataclass(frozen=True)
class Config:
    """Open demonstration stack, not the proprietary GlobalFoundries stack."""

    dx: float = 0.05
    aperture: float = 3.2
    waist: float = 1.5
    si_thickness: float = 0.22
    oxide_gap: float = 0.10
    poly_thickness: float = 0.16
    box_thickness: float = 2.0
    pml: float = 0.6
    run_time_fs: float = 220.0
    chunk_steps: int = 32
    wavelengths: tuple[float, ...] = (1.54, 1.55, 1.56)
    filter_radii: tuple[float, float] = (0.20, 0.10)
    learning_rate: float = 0.025
    steps_per_beta: int = 16
    betas: tuple[float, ...] = (2.0, 4.0, 8.0)
    seed: int = 240

    def __post_init__(self):
        for name in (
            "dx",
            "aperture",
            "waist",
            "si_thickness",
            "oxide_gap",
            "poly_thickness",
            "box_thickness",
            "pml",
            "run_time_fs",
            "learning_rate",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.chunk_steps < 1 or self.steps_per_beta < 1:
            raise ValueError("chunk_steps and steps_per_beta must be positive")
        if not self.betas or any(b <= 0 or not np.isfinite(b) for b in self.betas):
            raise ValueError("betas must be finite and positive")
        if not self.wavelengths or any(w <= 0 for w in self.wavelengths):
            raise ValueError("wavelengths must be positive")
        if len(self.filter_radii) != 2 or min(self.filter_radii) <= 0:
            raise ValueError("provide two positive layer filter radii")


class GratingProblem:
    """Reciprocal waveguide-to-Gaussian objective with a fixed input calibration."""

    def __init__(self, config=None):
        self.config = c = config if config is not None else Config()
        self.freqs = bz.LIGHT_SPEED / (np.asarray(c.wavelengths) * bz.um)
        # Grid-align the aperture and domain; design coordinates use cell centers.
        self.n = int(round(c.aperture / c.dx))
        self.aperture = self.n * c.dx
        self.domain = (
            np.ceil(
                np.array(
                    [
                        self.aperture + 2 * c.pml + 2.0,
                        self.aperture + 2 * c.pml + 0.8,
                        c.box_thickness
                        + c.si_thickness
                        + c.oxide_gap
                        + c.poly_thickness
                        + 2 * c.pml
                        + 1.4,
                    ]
                )
                / c.dx
            )
            * c.dx
        )
        self.z_si = -self.domain[2] / 2 + c.pml + c.box_thickness
        self.layers = (
            (self.z_si, c.si_thickness),
            (self.z_si + c.si_thickness + c.oxide_gap, c.poly_thickness),
        )
        self.x0 = -self.aperture / 2 + 0.6
        self.x1 = self.x0 + self.aperture
        self.source_x = -self.domain[0] / 2 + c.pml + 0.3
        self.input_x = self.source_x + 0.5
        self.fiber_z = self.layers[1][0] + c.poly_thickness + 0.5
        self.simulation = self.make_simulation()
        self.program = self.simulation.compile(num_steps=c.chunk_steps, backend="jax")
        self.initial = initial_program_state(self.program, t=0.0, current_step=0)
        self.chunks, self.tail = divmod(self.simulation.num_steps, c.chunk_steps)
        self.scan = build_scan(self.program)
        self.tail_scan = (
            build_scan(self.simulation.compile(num_steps=self.tail, backend="jax"))
            if self.tail
            else None
        )
        self.base = self.program.coefficients
        # Native rasterization supplies staggered layer fill fractions, including
        # subcell thicknesses. The two independent 2D densities only modulate these.
        self.layer_delta = []
        for layer in range(2):
            filled = self.make_simulation(filled_layer=layer).compile(
                num_steps=c.chunk_steps, backend="jax"
            )
            self.layer_delta.append(
                tuple(
                    jnp.asarray(getattr(filled.coefficients, f"e_permittivity_{a}"))
                    - jnp.asarray(getattr(self.base, f"e_permittivity_{a}"))
                    for a in "xyz"
                )
            )
        self.sample_coords = []
        for axis in "xyz":
            shape = getattr(self.base, f"e_permittivity_{axis}").shape
            offsets = component_axis_offsets_3d("E" + axis)
            x = (np.arange(shape[2]) + offsets["x"]) * c.dx - self.domain[0] / 2
            y = (np.arange(shape[1]) + offsets["y"]) * c.dx - self.domain[1] / 2
            yy, xx = np.meshgrid(
                (y + self.aperture / 2) / c.dx - 0.5,
                (x - self.x0) / c.dx - 0.5,
                indexing="ij",
            )
            self.sample_coords.append(
                jnp.asarray(np.stack((yy, xx)), dtype=jnp.float32)
            )
        self.fiber_spec = self.program.monitors[0]
        self.gaussian = self._gaussian_weights()
        self.incident = None
        self.evaluate = jax.jit(self._evaluate)
        self.value_gradient = jax.jit(jax.value_and_grad(self._loss, has_aux=True))

    def make_simulation(self, *, filled_layer=None, reference=False):
        c = self.config
        oxide = bz.Material(permittivity=1.44**2)
        silicon = bz.Material(permittivity=3.48**2)
        design = bz.Design(background=oxide)

        def box(center, size):
            return bz.Box(
                center=tuple(np.asarray(center) * bz.um),
                size=tuple(np.asarray(size) * bz.um),
                material=silicon,
            )

        # Silicon handle below the BOX; outgoing guide extends through left PML.
        bottom = -self.domain[2] / 2
        substrate_top = self.z_si - c.box_thickness
        design += box(
            (0, 0, (bottom + substrate_top) / 2),
            (self.domain[0], self.domain[1], substrate_top - bottom),
        )
        left, right = (
            -self.domain[0] / 2,
            (self.domain[0] / 2 if reference else self.x0),
        )
        design += box(
            ((left + right) / 2, 0, self.z_si + c.si_thickness / 2),
            (right - left, 0.5, c.si_thickness),
        )
        if filled_layer is not None:
            z, thickness = self.layers[filled_layer]
            design += box(
                ((self.x0 + self.x1) / 2, 0, z + thickness / 2),
                (self.aperture, self.aperture, thickness),
            )
        source = bz.ModeSource(
            center=(self.source_x * bz.um, 0, (self.z_si + c.si_thickness / 2) * bz.um),
            size=(0, 1.5 * bz.um, 1.4 * bz.um),
            direction="+",
            source_time=bz.GaussianPulse(
                freq0=bz.LIGHT_SPEED / (1.55 * bz.um),
                fwidth=0.25 * bz.LIGHT_SPEED / (1.55 * bz.um),
                offset=3.0,
            ),
            mode_spec=bz.ModeSpec(num_modes=4, polarization="te", target_neff=2.5),
        )
        if reference:
            monitors = [
                bz.FieldMonitor(
                    center=(
                        self.input_x * bz.um,
                        0,
                        (self.z_si + c.si_thickness / 2) * bz.um,
                    ),
                    size=(0, 1.5 * bz.um, 1.4 * bz.um),
                    freqs=self.freqs,
                    fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                    name="incident",
                )
            ]
        else:
            monitors = [
                bz.FieldMonitor(
                    center=(0, 0, self.fiber_z * bz.um),
                    size=(
                        (self.domain[0] - 2 * c.pml - 0.2) * bz.um,
                        (self.domain[1] - 2 * c.pml - 0.2) * bz.um,
                        0,
                    ),
                    freqs=self.freqs,
                    fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                    name="fiber",
                )
            ]
        return bz.Simulation(
            size=tuple(self.domain * bz.um),
            design=design,
            sources=[source],
            monitors=monitors,
            boundaries=[bz.PML(thickness=c.pml * bz.um, formulation="cpml")],
            grid_spec=bz.GridSpec.uniform(c.dx * bz.um, courant=0.8),
            run_time=c.run_time_fs * 1e-15,
        )

    def _gaussian_weights(self):
        c = self.config
        region = self.fiber_spec.sample_region
        monitor = self.simulation.monitors[0]
        # Simulation stores translated monitor coordinates already. Adding the
        # public-to-grid offset again would move the Gaussian off the aperture.
        center = monitor.center
        y, x = yee_plane_coordinates_3d(center, monitor.size, "z", region)
        self.fiber_x_um = x / bz.um - self.domain[0] / 2
        self.fiber_y_um = y / bz.um - self.domain[1] / 2
        yy, xx = np.meshgrid(
            y / bz.um - self.domain[1] / 2,
            x / bz.um - self.domain[0] / 2,
            indexing="ij",
        )
        g = np.exp(-((xx - (self.x0 + self.x1) / 2) ** 2 + yy**2) / c.waist**2)
        # Unit-power upward Gaussian: Hx = -Ey/Z. Normalize over the *entire*
        # transverse plane analytically, never inflate power by renormalizing a
        # truncated aperture. This is a paraxial target in homogeneous oxide.
        impedance = np.sqrt(bz.MU_0 / bz.EPS_0) / 1.44
        norm = np.sqrt(2 * impedance / (np.pi * (c.waist * bz.um) ** 2 / 2))
        return jnp.asarray(g.ravel() * norm, dtype=jnp.float32)

    def packed_fields(self, state, spec):
        start = spec.dft_value_offset
        stop = start + 6 * spec.freq_count * spec.dft_point_count
        fields = (
            state.dft_vec_re[start:stop] + 1j * state.dft_vec_im[start:stop]
        ).reshape(6, spec.freq_count, spec.dft_point_count)
        # H is half a timestep behind E; match Beamz's public flux convention.
        phase = jnp.exp(-1j * jnp.pi * jnp.asarray(self.freqs) * self.simulation.dt)
        return fields.at[3:].multiply(phase[None, :, None])

    @staticmethod
    def quadrature_weights(spec):
        return (
            jnp.asarray(spec.integration_weights)
            if spec.integration_weights.size
            else jnp.float32(spec.power_scale)
        )

    def calibrate(self):
        """Fixed, positive input spectrum from the identical straight guide."""
        simulation = self.make_simulation(reference=True)
        program = simulation.compile(backend="jax")
        state = initial_program_state(program, t=0.0, current_step=0)
        state = build_scan(program)(state, program.coefficients)
        fields = self.packed_fields(state, program.monitors[0])
        # A mode search near silicon's bulk index can select a substrate mode if
        # the padded mode plane reaches the handle. Verify the launched TE mode,
        # not just its positive power, before accepting a normalization spectrum.
        self.input_mode_diagnostics = input_mode_diagnostics(
            fields, program.monitors[0]
        )
        if (
            self.input_mode_diagnostics["te_fraction"] < 0.5
            or self.input_mode_diagnostics["ey_symmetry_error"] > 0.05
        ):
            raise RuntimeError(
                f"Input is not the even TE guide mode: {self.input_mode_diagnostics}"
            )
        # Same raw DFT convention as the objective, so temporal normalization
        # cancels exactly. x-normal power: 1/2 Re(Ey Hz* - Ez Hy*).
        weights = self.quadrature_weights(program.monitors[0])
        power = 0.5 * jnp.sum(
            jnp.real(fields[1] * fields[5].conj() - fields[2] * fields[4].conj())
            * weights,
            axis=-1,
        )
        power = np.asarray(power)
        if not np.all(np.isfinite(power)) or np.any(power <= 0):
            raise RuntimeError(f"Invalid incident calibration: {power}")
        self.incident = jnp.asarray(power)
        return power

    def initial_density(self):
        rng = np.random.default_rng(self.config.seed)
        density = 0.5 + 0.02 * rng.standard_normal((2, self.n, self.n))
        return jnp.asarray(density, dtype=jnp.float32)

    def physical_density(self, density, beta):
        # Mirror geometry about the guide axis; layers remain independent.
        symmetric = (density + density[:, ::-1, :]) / 2
        return jnp.stack(
            [
                transform_density(
                    symmetric[layer],
                    jnp.ones((self.n, self.n), dtype=bool),
                    beta,
                    0.5,
                    max(1, round(radius / self.config.dx)),
                    filter_type="conic",
                )
                for layer, radius in enumerate(self.config.filter_radii)
            ]
        )

    def coefficients(self, physical):
        values = {}
        for i, axis in enumerate("xyz"):
            epsilon = jnp.asarray(getattr(self.base, f"e_permittivity_{axis}"))
            for layer in range(2):
                plane = map_coordinates(
                    physical[layer], self.sample_coords[i], order=1, mode="nearest"
                )
                epsilon = epsilon + self.layer_delta[layer][i] * plane[None, :, :]
            values[f"e_permittivity_{axis}"] = epsilon
        return self.base._replace(**values)

    def forward(self, physical):
        coefficients = self.coefficients(physical)

        # Recompute each chunk in reverse mode, storing only chunk boundaries.
        # No buffer donation: every optimization evaluation starts from zero.
        def chunk(state, _):
            return self.scan(state, coefficients), None

        state, _ = jax.lax.scan(
            jax.checkpoint(chunk), self.initial, None, length=self.chunks
        )
        if self.tail_scan is not None:
            state = jax.checkpoint(self.tail_scan)(state, coefficients)
        return state

    def _evaluate(self, density, beta):
        state = self.forward(self.physical_density(density, beta))
        fields = self.packed_fields(state, self.fiber_spec)
        impedance = np.sqrt(bz.MU_0 / bz.EPS_0) / 1.44
        # Lorentz power overlap with unit-power +z target:
        # a = 1/4 integral(E x H_target* + E_target* x H).n dA.
        weights = self.quadrature_weights(self.fiber_spec)
        return (
            gaussian_overlap_power(fields, self.gaussian, weights, impedance)
            / self.incident
        )

    def _loss(self, density, beta):
        efficiency = self._evaluate(density, beta)
        # Smooth worst-case log loss; temperature=0.1, approaches minimax as it
        # decreases. This is explicitly not the paper's epigraph/GCMMA algorithm.
        losses = -jnp.log(jnp.maximum(efficiency, 1e-12))
        loss = 0.1 * jax.scipy.special.logsumexp(losses / 0.1)
        return loss, efficiency

    def check_gradient(self, density, beta=2.0, step=0.005):
        """Central differences along independent directions in both layers."""
        (_, _), gradient = self.value_gradient(density, jnp.float32(beta))
        rows = []
        for layer in range(2):
            direction = np.zeros(density.shape, dtype=np.float32)
            # Check each layer in its gradient direction to avoid cancellation.
            direction[layer] = np.asarray(gradient[layer])
            direction /= max(float(np.linalg.norm(direction)), 1e-30)
            direction = jnp.asarray(direction)
            plus = self._loss(density + step * direction, jnp.float32(beta))[0]
            minus = self._loss(density - step * direction, jnp.float32(beta))[0]
            finite_difference = float((plus - minus) / (2 * step))
            autodiff = float(jnp.sum(gradient * direction))
            error = abs(autodiff - finite_difference) / max(
                abs(autodiff), abs(finite_difference), 1e-8
            )
            rows.append(
                dict(
                    layer=layer,
                    autodiff=autodiff,
                    finite_difference=finite_difference,
                    relative_error=error,
                )
            )
        return rows


def optimize(problem, output):
    """Save every evaluated design and the next density after each iteration."""
    c = problem.config
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problem.calibrate()
    density = problem.initial_density()
    checks = problem.check_gradient(density, c.betas[0])
    print("Gradient checks:", checks, flush=True)
    if any(row["relative_error"] > 0.05 or row["autodiff"] <= 0 for row in checks):
        raise RuntimeError("Full FDTD gradient check failed")
    optimizer = optax.adam(c.learning_rate)
    history, designs = [], []
    started = time.monotonic()
    for beta in c.betas:
        optimizer_state = optimizer.init(density)
        for _ in range(c.steps_per_beta):
            (loss, efficiency), gradient = problem.value_gradient(
                density, jnp.float32(beta)
            )
            loss, efficiency = float(loss), np.asarray(efficiency)
            if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                raise RuntimeError("Nonfinite objective or gradient")
            history.append([beta, loss, *efficiency])
            designs.append(np.asarray(density).copy())
            print(
                f"{len(history):03d} beta={beta:g} loss={loss:.5f} eta={efficiency}",
                flush=True,
            )
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, density
            )
            density = jnp.clip(optax.apply_updates(density, updates), 0, 1)
            np.savez_compressed(
                output / "history.npz",
                history=np.asarray(history),
                designs=np.asarray(designs),
                next_density=np.asarray(density),
            )
    # The last updated design must be evaluated; history's last row predates it.
    final = np.asarray(problem.evaluate(density, jnp.float32(c.betas[-1])))
    physical = np.asarray(problem.physical_density(density, c.betas[-1]))
    np.savez_compressed(
        output / "final.npz",
        density=np.asarray(density),
        physical=physical,
        efficiency=final,
        incident=np.asarray(problem.incident),
    )
    report = dict(
        config=asdict(c),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device=str(jax.devices()[0]),
        device_kind=jax.devices()[0].device_kind,
        jax_version=jax.__version__,
        grid=list(problem.simulation.grid.shape),
        steps=problem.simulation.num_steps,
        gradient_checks=checks,
        input_mode=problem.input_mode_diagnostics,
        initial_efficiency=[float(v) for v in history[0][2:]],
        final_efficiency=final.tolist(),
        elapsed_seconds=time.monotonic() - started,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return density, np.asarray(history), report


def validate_design(config, density, output):
    """Re-evaluate the final density with longer time, binary material and finer mesh.

    This reports sensitivity, not a convergence certificate. No optimizer is run
    on the validation mesh. The latent density is interpolated to that mesh and
    physical filter radii are kept fixed.
    """
    from dataclasses import replace

    from scipy.ndimage import zoom

    output = Path(output)
    spectrum_config = replace(
        config,
        run_time_fs=2 * config.run_time_fs,
        wavelengths=tuple(np.linspace(1.50, 1.60, 21)),
    )
    spectrum_problem = GratingProblem(spectrum_config)
    spectrum_problem.calibrate()
    beta = jnp.float32(config.betas[-1])
    spectrum = np.asarray(spectrum_problem.evaluate(jnp.asarray(density), beta))
    physical = spectrum_problem.physical_density(jnp.asarray(density), beta)
    binary = jnp.asarray(physical >= 0.5, dtype=jnp.float32)
    binary_state = jax.jit(spectrum_problem.forward)(binary)
    fields = spectrum_problem.packed_fields(binary_state, spectrum_problem.fiber_spec)
    impedance = np.sqrt(bz.MU_0 / bz.EPS_0) / 1.44
    binary_efficiency = np.asarray(
        gaussian_overlap_power(
            fields,
            spectrum_problem.gaussian,
            spectrum_problem.quadrature_weights(spectrum_problem.fiber_spec),
            impedance,
        )
        / spectrum_problem.incident
    )
    np.savez_compressed(
        output / "spectrum.npz",
        wavelengths=spectrum_config.wavelengths,
        efficiency=spectrum,
        binary_efficiency=binary_efficiency,
        binary=np.asarray(binary),
        fiber_fields=np.asarray(fields),
        fiber_x_um=spectrum_problem.fiber_x_um,
        fiber_y_um=spectrum_problem.fiber_y_um,
    )
    # Drop executable references before compiling the larger validation grid.
    del spectrum_problem, binary_state
    jax.clear_caches()
    fine_config = replace(config, dx=config.dx / 2, run_time_fs=2 * config.run_time_fs)
    fine = GratingProblem(fine_config)
    fine.calibrate()
    fine_density = zoom(
        np.asarray(density),
        (1, fine.n / density.shape[1], fine.n / density.shape[2]),
        order=1,
    )
    fine_efficiency = np.asarray(fine.evaluate(jnp.asarray(fine_density), beta))
    data = dict(
        wavelengths=list(config.wavelengths),
        long_time_efficiency=np.interp(
            config.wavelengths, spectrum_config.wavelengths, spectrum
        ).tolist(),
        fine_mesh_efficiency=fine_efficiency.tolist(),
        fine_dx_um=fine_config.dx,
        run_time_fs=fine_config.run_time_fs,
    )
    (output / "validation.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/multilayer_grating")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--validate", action="store_true", help="validate saved final design"
    )
    args = parser.parse_args()
    if args.validate:
        report = json.loads((args.output / "report.json").read_text())
        config = Config(**report["config"])
        with np.load(args.output / "final.npz") as saved:
            density = saved["density"]
        print(validate_design(config, density, args.output), flush=True)
        return
    c = Config(dx=0.10, steps_per_beta=2, betas=(2.0,)) if args.smoke else Config()
    print("JAX devices:", jax.devices(), flush=True)
    problem = GratingProblem(c)
    print(
        "Grid:",
        problem.simulation.grid.shape,
        "steps:",
        problem.simulation.num_steps,
        flush=True,
    )
    optimize(problem, args.output)


if __name__ == "__main__":
    main()
