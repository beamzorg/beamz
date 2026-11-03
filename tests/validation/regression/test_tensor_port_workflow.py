"""End-to-end numerical evidence for the explicitly supported tensor ports."""

from dataclasses import replace

import numpy as np
import pytest

import beamz as bz
from beamz.analysis.sparameters import _extract_port_waves_dft
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    Box,
    ExtrudedPolygon,
    Grid,
    Material,
    Object,
    Polygon,
    RasterOptions,
    Scene,
    rasterize,
)
from beamz.simulation.results import SimulationResults


def run(
    axis="x",
    direction="+",
    spacing_nm=100,
    *,
    cross=True,
    smoothing="farjadpour_full",
    smoothed=False,
    rectilinear=False,
    graded_normal=False,
    backend="jax",
    source_power=1.0,
    aperture_um=2.2,
):
    axes = {"x": "xyz", "y": "yxz", "z": "zxy"}[axis]
    order = [axes.index(a) for a in "xyz"]

    def point(values):
        return tuple(np.asarray(values)[order] * 1e-6)

    epsilon = np.array([[6.0, 0.2, 0.1], [0.2, 5.0, 0.3], [0.1, 0.3, 4.0]])[
        np.ix_(order, order)
    ]
    if not cross:
        epsilon = np.diag(np.diag(epsilon))
    scene = Scene(
        (Material(), Material(epsilon_r=tuple(map(tuple, epsilon)))),
        (Object(Box(point((0, 1.5, 1.5)), point((6.4, 2.1, 2.1))), 1),),
    )
    if smoothed:
        if axis != "z":
            raise ValueError("The oblique-interface benchmark is z-extruded")
        angle = np.pi / 6
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        vertices = (
            np.array([[-0.3, -0.2], [0.3, -0.2], [0.3, 0.2], [-0.3, 0.2]]) @ rotation.T
            + 1.8
        )
        polygon = Polygon(tuple(map(tuple, vertices * 1e-6)))
        scene = Scene(
            (Material(), Material(6.0)),
            (Object(ExtrudedPolygon(polygon, 0.0, 6.4e-6), 1),),
        )
    extent = point((6.4, 3.6, 3.6))
    shape = tuple(round(v / (spacing_nm * 1e-9)) for v in extent)
    grid = Grid.uniform((0, 0, 0), extent, shape)
    if rectilinear:
        edges = []
        for a, length, count in zip("xyz", extent, shape, strict=True):
            u = np.linspace(0, 1, count + 1)
            values = length * u
            if a != axis:
                values = length * (u + 0.08 * np.sin(2 * np.pi * u))
            elif graded_normal:
                # Local port stencils are uniform; a smooth grid perturbation
                # in the device interior exercises longitudinal metric updates.
                in_bump = (values > 3.0e-6) & (values < 3.4e-6)
                values[in_bump] += (
                    0.02e-6 * np.sin(np.pi * (values[in_bump] - 3.0e-6) / 0.4e-6) ** 2
                )
            edges.append(values)
        grid = Grid(*edges)
    material = MaterialGrid.from_raster_result(
        rasterize(
            scene,
            grid,
            options=RasterOptions(smoothing=smoothing),
        )
    )
    f = bz.LIGHT_SPEED / 1.55e-6
    sign = 1 if direction == "+" else -1
    source_x = 2.0 if sign == 1 else 4.4
    # These accuracy checks need decayed solver boundaries beyond the measured
    # aperture. Padding is explicit now; the public default adds no halo.
    spec = bz.ModeSpec(
        num_modes=2, target_neff=2.0, polarization=None, solver_padding_cells=12
    )
    size = point((0, aperture_um, aperture_um))
    source = bz.ModeSource(
        center=point((source_x, 1.8, 1.8)),
        size=size,
        direction=direction,
        source_time=bz.GaussianPulse(freq0=f, fwidth=0.1 * f, offset=5 / (2 * np.pi)),
        mode_spec=spec,
        power=source_power,
    )
    ports = []
    for name, distance in [
        ("back", -0.6),
        ("near", 0.6),
        ("far", 2.0 if rectilinear else 1.4),
        ("shifted", 2.023 if rectilinear else 1.423),
    ]:
        ports.append(
            bz.Port(
                center=point((source_x + sign * distance, 1.8, 1.8)),
                size=size,
                direction="+",
                name=name,
                mode_spec=spec,
            )
        )
    normal_edges = {
        "x": ["left", "right"],
        "y": ["bottom", "top"],
        "z": ["front", "back"],
    }[axis]
    other_edges = [
        e
        for e in ["left", "right", "bottom", "top", "front", "back"]
        if e not in normal_edges
    ]
    sim = bz.Simulation(
        material_grid=material,
        sources=[source],
        monitors=[p.to_monitor([f]) for p in ports]
        + (
            [
                bz.FluxMonitor(
                    center=p.center, size=p.size, freqs=[f], name=p.name + "_flux"
                )
                for p in ports
                if p.name in {"near", "far"}
            ]
            if rectilinear
            else []
        ),
        boundaries=[
            bz.PML(edges=normal_edges, thickness=1.2e-6, formulation="cpml"),
            bz.PML(edges=other_edges, thickness=0.6e-6, formulation="cpml"),
        ],
        run_time=200e-15,
        setup_device="cpu",
    )
    preview = (
        source.solve_modes(sim, freqs=[f], field_representation="launch")
        if rectilinear
        else None
    )
    result = sim.run(backend=backend, progress=False)
    secondary = [
        replace(
            p,
            name=p.name + "_mode1",
            monitor_name=p.name,
            mode_spec=replace(spec, mode_index=1),
        )
        for p in ports
        if p.name in {"near", "far"}
    ]
    waves = _extract_port_waves_dft(
        result, ports=[*ports, *secondary], frequencies=[f], return_power=True
    )
    positive, negative = ("a_plus", "a_minus") if axis == "z" else ("a_minus", "a_plus")
    forward, backward = (positive, negative) if sign == 1 else (negative, positive)
    metrics = {}
    for p in ports:
        w = waves[p.name]
        metrics[p.name] = {
            "forward_amplitude": [float(w[forward][0].real), float(w[forward][0].imag)],
            "forward_power": float(abs(w[forward][0]) ** 2),
            "backward_power": float(abs(w[backward][0]) ** 2),
            "flux": float(result[p.name].flux[0]),
            "sampled_modal_flux": float(w["projected_signed_power"][0]),
            "reconstruction_residual": float(w["projection_residual_balanced"][0]),
            "neff": float(w["mode_neff"][0]),
        }
        if rectilinear and p.name in {"near", "far"}:
            metrics[p.name]["independent_flux"] = float(
                result[p.name + "_flux"].flux[0]
            )
    for p in secondary:
        metrics[p.monitor_name]["other_mode_power"] = float(
            abs(waves[p.name][forward][0]) ** 2
        )
    return (
        {
            "preview_neffs": []
            if preview is None
            else np.real(preview.neffs[0]).tolist(),
            "ports": metrics,
        },
        result,
        ports,
        waves,
    )


def test_absolute_modal_watts_and_reference_api_for_nonunit_source():
    report, result, _, _ = run(source_power=2.5)
    near, far = result.mode("near"), result.mode("far")
    assert result.launched_power() == pytest.approx(2.5, rel=1e-10)
    assert report["ports"]["near"]["forward_power"] == pytest.approx(2.5, rel=1e-3)
    transmission = far.normalized_to(near)
    np.testing.assert_allclose(
        transmission.power.sel(direction="+", mode_index=0), 1, rtol=1e-3
    )
    assert abs(near.flux.item()) == pytest.approx(2.5, rel=1e-3)
    # Reference normalization scales amplitudes and measured flux together.
    np.testing.assert_allclose(transmission.modal_flux, transmission.flux, rtol=5e-5)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_tensor_guide_launch_projection_cpml_and_archive(axis, direction, tmp_path):
    report, result, ports, waves = run(axis, direction)
    values = report["ports"]
    near = values["near"]
    assert abs(near["flux"]) == pytest.approx(result.launched_power(), rel=1e-3)
    assert near["forward_power"] > 0
    assert near["forward_power"] == pytest.approx(result.launched_power(), rel=1e-3)
    for name in ("near", "far", "shifted"):
        row = values[name]
        assert row["backward_power"] / row["forward_power"] < 1e-6
        # Off-grid sampling can change discrete flux; modal watts must track
        # the independently recorded flux at that same plane.
        assert row["forward_power"] == pytest.approx(abs(row["flux"]), rel=5e-5)
        if name != "shifted":
            assert row["forward_power"] == pytest.approx(
                near["forward_power"], rel=1e-3
            )
        assert row["sampled_modal_flux"] == pytest.approx(row["flux"], rel=5e-5)
        assert row["reconstruction_residual"] < 2e-3
    assert abs(values["back"]["flux"]) / abs(near["flux"]) < 1e-6
    assert near["other_mode_power"] / near["forward_power"] < 1e-6
    assert values["far"]["other_mode_power"] / near["forward_power"] < 1e-6
    positive, negative = ("a_plus", "a_minus") if axis == "z" else ("a_minus", "a_plus")
    forward = positive if direction == "+" else negative
    public = result.mode("near")
    assert public.monitor.mode_spec.polarization is None
    for index, name in enumerate(("near", "near_mode1")):
        np.testing.assert_allclose(
            public.amps.sel(direction=direction, mode_index=index),
            waves[name][forward],
            rtol=1e-10,
            atol=1e-10,
        )
    np.testing.assert_allclose(public.modal_flux, result["near"].flux, rtol=5e-5)
    np.testing.assert_allclose(
        public.power.sel(direction=direction, mode_index=0),
        near["forward_power"],
        rtol=1e-10,
    )
    assert np.max(abs(public.unresolved_flux)) < 5e-5
    transmission = waves["far"][forward][0] / waves["near"][forward][0]
    expected = np.exp(1j * 2 * np.pi / 1.55e-6 * near["neff"] * 0.8e-6)
    np.testing.assert_allclose(transmission, expected, rtol=0, atol=5e-5)
    archive = tmp_path / "results.npz"
    result.save(archive)
    loaded = SimulationResults.load(archive)
    np.testing.assert_allclose(
        loaded.mode("near").amps, public.amps, rtol=1e-12, atol=1e-12
    )
    restored = _extract_port_waves_dft(
        loaded, ports=ports, frequencies=[299792458 / 1.55e-6], return_power=True
    )
    # Repeat with the same modal grouping used above: archive must not affect any basis.
    original = _extract_port_waves_dft(
        result, ports=ports, frequencies=[299792458 / 1.55e-6], return_power=True
    )
    for port in ports:
        for key in ("a_plus", "a_minus", "mode_neff", "projected_signed_power"):
            np.testing.assert_allclose(
                restored[port.name][key],
                original[port.name][key],
                rtol=1e-12,
                atol=1e-12,
            )
        assert (
            not loaded[port.name]
            .material_region.yee_tensors["eps_node"]
            .flags.writeable
        )


def test_zero_cross_terms_recover_diagonal_workflow():
    _, full, ports, full_waves = run(cross=False)
    _, diagonal, _, diagonal_waves = run(cross=False, smoothing="farjadpour_diagonal")
    for port in ports:
        for key in ("a_plus", "a_minus", "projected_signed_power"):
            np.testing.assert_allclose(
                full_waves[port.name][key],
                diagonal_waves[port.name][key],
                rtol=1e-7,
                atol=1e-9,
            )
        np.testing.assert_allclose(
            full[port.name].flux, diagonal[port.name].flux, rtol=1e-7, atol=1e-9
        )
        for component in full[port.name].dft_fields:
            np.testing.assert_allclose(
                full[port.name].dft_fields[component],
                diagonal[port.name].dft_fields[component],
                rtol=1e-7,
                atol=1e-9,
            )


def test_reciprocal_tensor_guide_complex_transmission():
    transmissions = []
    for direction in ("+", "-"):
        _, _, _, waves = run(direction=direction)
        forward = "a_minus" if direction == "+" else "a_plus"
        transmissions.append(waves["far"][forward][0] / waves["near"][forward][0])
    np.testing.assert_allclose(*transmissions, rtol=0, atol=5e-5)


def test_oblique_isotropic_interface_uses_coupled_port_operator():
    report, result, _, _ = run(axis="z", smoothed=True)
    assert np.any(result["near"].material_region.yee_tensors["eps_node"][3:] != 0)
    for name in ("near", "far", "shifted"):
        row = report["ports"][name]
        assert row["backward_power"] / row["forward_power"] < 1e-6
        assert row["reconstruction_residual"] < 2e-3
        assert row["sampled_modal_flux"] == pytest.approx(row["flux"], rel=5e-5)
    assert (
        report["ports"]["near"]["other_mode_power"]
        / report["ports"]["near"]["forward_power"]
        < 1e-6
    )
