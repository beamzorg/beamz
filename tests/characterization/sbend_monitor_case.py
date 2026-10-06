"""S-bend regression geometry from beamzorg/beamz#309; no GDS/PDK required."""

from __future__ import annotations

import dataclasses

import numpy as np

import beamz as bz
from beamz.analysis import s_parameters
from beamz.devices.sources.time import gaussian_band_pulse


class Probe:
    def __init__(
        self,
        scene,
        mesh,
        *,
        widths=(1.35, 2.7, 4.0),
        distances=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
    ):
        self.scene, self.mesh = scene, mesh
        self.widths, self.distances = widths, distances

    def run(self):
        scene = self.scene
        design = bz.Design(
            width=scene["size_m"][0],
            height=scene["size_m"][1],
            depth=scene["size_m"][2],
            material=bz.Material(permittivity=scene["background_epsilon"]),
            structures=[
                bz.Polygon(
                    vertices=s["vertices_m"],
                    z=s["z_m"],
                    depth=s["depth_m"],
                    material=bz.Material(permittivity=s["epsilon"]),
                )
                for s in scene["structures"]
            ],
        )
        dx, dt = bz.dxdt(
            1.55e-6,
            n_max=scene["n_max"],
            dims=3,
            safety_factor=0.999,
            points_per_wavelength=self.mesh,
        )
        frequencies = 299792458 / (np.array([1.6, 1.55, 1.5]) * 1e-6)
        pulse = gaussian_band_pulse(
            frequencies,
            carrier_frequency=299792458 / 1.55e-6,
            dt=dt,
            run_after_sources_uoc=90,
            max_output_distance_um=3,
        )
        mode = bz.ModeSpec(polarization="te")

        def port(original, name, inward_offset_um):
            p = scene["ports"][original]
            center = list(p["center_m"])
            center[0] += (1 if p["direction"] == "+" else -1) * inward_offset_um * 1e-6
            return bz.Port(
                center=tuple(center),
                size=tuple(p["size_m"]),
                direction=p["direction"],
                name=name,
                mode_spec=mode,
            )

        src = port("opt1", "source", -1.4)
        input_port = port("opt1", "input", -0.5)
        outputs = []
        for width in self.widths:
            for distance in self.distances:
                name = f"w{width:g}_d{distance:g}"
                base = port("opt2", name, -distance)
                size = (0, width * 1e-6, width * 2.196 / 2.7 * 1e-6)
                outputs.append(
                    bz.Port(
                        center=base.center,
                        size=size,
                        direction=base.direction,
                        name=name,
                        mode_spec=mode,
                    )
                )
        source = bz.ModeSource(
            center=src.center,
            size=src.size,
            direction=src.direction,
            mode_spec=mode,
            source_time=bz.SampledSignal(
                values=pulse.signal,
                quadrature=pulse.signal_quadrature,
                dt=dt,
                freq0=299792458 / 1.55e-6,
            ),
        )
        sim = bz.Simulation(
            design=design,
            sources=[source],
            monitors=[p.to_monitor(frequencies) for p in [input_port, *outputs]],
            boundaries=[
                bz.PML(
                    edges=("left", "right", "top", "bottom", "front", "back"),
                    thickness=1e-6,
                )
            ],
            time=pulse.time,
            resolution=dx,
            setup_device="cpu",
        )
        results = sim.run(
            termination=bz.AutoTermination(
                min_steps=int(np.ceil((pulse.source_end_time + pulse.tail_time) / dt)),
                field_decay=1e-4,
            )
        )
        extracted = s_parameters(
            results,
            source_port="input",
            ports=[input_port, *outputs],
            frequencies=frequencies,
        )
        return {
            "wavelength_um": (299792458 / frequencies / 1e-6).tolist(),
            "s21_db": {
                p.name: (
                    20 * np.log10(np.abs(extracted.s_matrix[(p.name, "input")]))
                ).tolist()
                for p in outputs
            },
            "valid_mask": np.asarray(extracted.diagnostics["valid_mask"]).tolist(),
            "termination": dataclasses.asdict(results.termination),
        }
