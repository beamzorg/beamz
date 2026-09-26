<div align="center">
  <picture style="padding-right: 0px; padding-bottom: 7px;">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/beamz_logo_black.png">
    <img alt="BeamZ logo" src="docs/assets/beamz_logo_white.png" width="130">
  </picture>

  <strong>BeamZ</strong> is a differentiable, <strong>GPU-accelerated</strong> <strong><a href="https://en.wikipedia.org/wiki/Electromagnetism">electromagnetic</a> simulation</strong> framework for photonic chip designers using the <strong><a href="https://en.wikipedia.org/wiki/Finite-difference_time-domain_method">FDTD</a> method</strong>. Its engine enables fast, large-scale simulations and offers a <strong>familiar, high-level API</strong> for fast prototyping with just a few lines of code as well as an <strong>inverse design module</strong> for gradient-based optimization.

  <h3>

  [Homepage](https://www.beamz.tech) / [Documentation](https://www.beamz.tech/docs/getting-started/) / [Example Library](https://www.beamz.tech/simulation-examples)
  
  </h3>

  [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/beamzorg/beamz/blob/main/LICENSE)
  [![Tests](https://github.com/beamzorg/beamz/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/beamzorg/beamz/actions/workflows/tests.yml)
  [![Coverage](https://raw.githubusercontent.com/beamzorg/beamz/main/.github/badges/coverage.svg)](https://github.com/beamzorg/beamz/actions/workflows/tests.yml)
</div>



## Core Features
- **Python-first** with an intuitive and **familiar API**.
- **Free and open-source** FDTD simulation in **true 2D and 3D**.
- **GPU-accelerated**, achieving high **GCUPS performance**.
- **Multi-GPU** runs, handling **large-scale simulations** with _billions of cells_.
- CPU-capable for **fast prototyping**, even on your laptop.
- Dedicated CUDA-backend for acceleration beyond default Jax.
- **CPML**, absorbing layers, PEC and zero-phase periodic boundaries (JAX).
- **Unidirectional mode sources** (single freq. and broadband, Huygens fields + TFSF, TE/TM).
- **3D Gaussian sources**, e.g. for grating coupler simulations.
- **Broadband dispersive materials** (pole-residue, Drude/Lorentz) and uniform plane-wave sources on single-device JAX.
- **DFT monitors** and S-parameter extraction workflow for compact modeling.
- Integrated **FDFD mode solver** and **rasterization module**.
- Regular and **rectilinear meshing**.
- Full-tensor and diagonal Farjadpour (polarized averaging) **subpixel-smoothing**.
- Simple **GDS**, **GMSH** and **STL** import and export.
- Streamlined, integrated **parametric design** module.
- Projection kernels and utilies for inverse design via **auto-diff and adjoint-method** workflow.


## Examples

Try out notebooks from our growing **[example library](https://beamz.tech/simulation-examples/)**. It includes:

- [1) Mode Sources and Monitors (3D)](https://beamz.tech/examples/modal_sources_monitors)
- [2) Waveguide Crossing with Cosine Tapers (3D)](https://beamz.tech/examples/cosine_waveguide_crossing)
- [3) Grating Coupler (3D)](https://www.beamz.tech/examples/grating_coupler)
- [4) Microring Resooator (3D)](https://www.beamz.tech/examples/ring_resonator_add_drop)
- [5) 1x4 MMI Powersplitter (3D)](https://www.beamz.tech/examples/mmi1x4_power_splitter)
- [6) Topology Optimized 90° Bend (2D)](https://beamz.tech/examples/ceviche_bend)
- [7) GDSFactory PDK 1x2 MMI (3D)](https://www.beamz.tech/examples/gdsfactory_component_sparameters)


- [CMOS RGB Image Sensor](https://beamz.tech/examples/cmos_rgb_sensor)
- [High-Q Silicon Resonator](https://beamz.tech/examples/high_q_silicon_resonator)

## Integration 

BeamZ is used by several other OSS packages as an FDTD engine and integrates with workflows across the open-source photonics ecosystem:
+ [SiEPIC's GDS FDTD](https://github.com/SiEPIC/gds_fdtd), an EDA- and solver-agnostic 3D FDTD compact modeling framework using BeamZ as the default engine.
+ [Lumix](https://github.com/amiskandarmuda/lumix), a research codebase for optical neural networks and matrix inverse design, using BeamZ as the main solver.
+ [GDSFactory](https://github.com/gdsfactory/gdsfactory), the most popular open-source layout package for chip design for which BeamZ provides a simple workflow from PDK import to compact modeling.


## Installation

Install BeamZ using pip:

```bash
pip install beamz
```

Development uses [uv](https://docs.astral.sh/uv/). Clone the repository and sync
the package with its contributor dependencies:

```bash
git clone https://github.com/beamzorg/beamz
cd beamz
uv sync --extra dev --extra test
```

For a ready-to-use CUDA and Jupyter development environment, see the
[Docker and RunPod guide](docker/runpod/README.md).


## Mission

BeamZ aims to become the FDTD engine of choice for **photonic simulations** in industry and research, including photonic circuits, inverse design, metamaterials, fiber optics, nanophotonics, sensors (planned), active photonic devices (planned), and RF (planned), focusing on **streamlined workflows** and bringing multi-physics GPU-acceleration for **maximum performance in large-scale simulations** to everyone. - [_Detailed roadmap coming soon_.](www.beamz.tech/roadmap)


## Contributing

**We appreciate all contributions.** If you are planning to contribute bug-fixes, please do so without any further discussion. If you would like to add new features, please first open an issue and discuss the feature with us. There may be ongoing work that could conflict with your changes, or we may be heading in a different direction and we don't want to waste your time working on something that might be rejected. - You can find [more information here](CONTRIBUTING.md).

The simplest way to support the project of course is by **giving this repo a star.** Thank you!

---

Copyright © 2026 Quentin Wach — [Apache-2.0](https://github.com/beamzorg/beamz/blob/HEAD/LICENSE)
