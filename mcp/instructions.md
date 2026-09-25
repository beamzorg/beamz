# BeamZ FDTD MCP Server Instructions

## Overview
BeamZ MCP provides electromagnetic Finite-Difference Time-Domain (FDTD) simulation and eigenmode solving for integrated photonic circuits.

---

## 1. Units & Coordinate Conventions (CRITICAL)

- **Units**: All dimensions, lengths, widths, heights, radii, spans, and coordinates are strictly in **micrometers ($\mu\text{m}$)**. Wavelengths are also in **$\mu\text{m}$** (e.g., $1.55$ for $1550\,\text{nm}$, $0.65$ for $650\,\text{nm}$).
- **Rectangle (`Rect` / `add_structure`)**:
  - `(x_um, y_um, z_um)` represents the **LOWER-LEFT-FRONT corner**.
  - To center a rectangle of width $W$ and height $H$ at $(x_c, y_c)$, place it at:
    $$x = x_c - W/2, \quad y = y_c - H/2$$
- **Circle & Ring (`add_circle`, `add_ring`)**:
  - `(x_um, y_um)` represents the **xy-CENTER**.
- **Polygon (`add_polygon`)**:
  - `vertices` is a list of 2D coordinates `[(x1, y1), (x2, y2), ...]`.
- **Sources & Monitors (`add_source`, `add_monitor`, `add_mode_monitor`)**:
  - `(x_um, y_um, z_um)` represents the **CENTER** of the modal/flux plane.

---

## 2. Boundary & PML Rules

- **Waveguide Continuity**: Continuous input and output waveguides **MUST** extend from the outer domain boundary ($x = 0$) completely through the opposite boundary ($x = \text{domain.width\_um}$) into the PML. Do NOT terminate waveguides right at the PML boundary, as this causes artificial facet reflections.
- **Source and Monitor Placement**: Always place sources and monitors inside the active simulation domain, away from the PML absorbing boundaries by at least $0.5\,\mu\text{m}$.

---

## 3. Workflows

### Option A: All-In-One Execution (`simulate_device`) — RECOMMENDED FOR CHATBOTS
Allows defining and executing an entire simulation in a single tool call without multi-turn polling:
```json
{
  "domain": {
    "width_um": 10.0,
    "height_um": 4.0,
    "resolution_um": 0.04,
    "cladding_index": 1.44
  },
  "structures": [
    {
      "name": "wg",
      "x_um": 0.0,
      "y_um": 1.75,
      "width_um": 10.0,
      "height_um": 0.50,
      "index": 3.48
    }
  ],
  "sources": [
    {
      "name": "src",
      "x_um": 1.5,
      "y_um": 2.0,
      "span_um": 2.0
    }
  ],
  "monitors": [
    {
      "name": "ref",
      "x_um": 2.2,
      "y_um": 2.0,
      "span_um": 1.8
    },
    {
      "name": "out",
      "x_um": 8.0,
      "y_um": 2.0,
      "span_um": 1.8
    }
  ]
}
```

### Option B: Granular Step-by-Step
1. `create_scene`: Create 2D or 3D scene (returns `scene_id`).
2. Add geometries: `add_structure`, `add_circle`, `add_ring`, `add_polygon`.
3. Add sources and monitors: `add_source`, `add_gaussian_source`, `add_monitor`, `add_mode_monitor`, `add_time_monitor`.
4. Validate & pre-solve: `validate(scene_id)` or `solve_modes(scene_id)`.
5. Run simulation: `run_simulation(scene_id, wait=True)`.
6. Inspect results: `get_transmission(job_id)`, `get_mode_data(job_id)`, `plot_field_snapshot(job_id)`.
