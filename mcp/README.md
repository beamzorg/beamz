# BeamZ MCP Server
A Model Context Protocol (MCP) server for the BeamZ FDTD solver. It lets AI assistants and MCP-compatible clients create photonic scenes, validate them, run simulations, and inspect results using structured tool calls instead of hand-written scripts.

## Why this project exists
BeamZ provides the numerical engine for electromagnetic simulation, but using it directly from a chatbot or AI agent is awkward. This MCP server exposes that functionality as tools with clear inputs, validation, and outputs so a model can:

- describe a photonic device in natural language
- create a scene and geometry programmatically
- validate the design before running a full solve
- run FDTD or mode analysis
- review transmission, mode data, and field snapshots

## Features
- 2D and 3D photonic scene creation
- Geometry primitives: rectangles, circles, rings, and polygons
- Mode source and Gaussian source support
- Flux, mode, and time monitors
- Scene validation and compute-cost estimation
- One-shot simulation helper via `simulate_device`
- Structured outputs for transmission and field visualization
- Prompt templates for common photonic design tasks

## Requirements
- Python 3.12+
- BeamZ 0.5.0+
- A compatible MCP client such as Claude Desktop, Antigravity, or another MCP host

## Quick start
Start the server locally:

```bash
python server.py
```

This starts the MCP server over stdio. It does not print a normal CLI banner, because it is designed to be used by an MCP client rather than by a human terminal session.

## Configure an MCP client
Use absolute paths in your client configuration. The MCP server is launched with Python and points to `server.py`.

### Claude Desktop
Open the config file for Claude Desktop and add:

```json
{
  "mcpServers": {
    "beamz": {
      "command": "C:/absolute/path/to/python.exe",
      "args": [
        "D:/absolute/path/to/beamz-mcp/server.py"
      ]
    }
  }
}
```

### Antigravity
Open the MCP config for Antigravity and add the same server entry:

```json
{
  "mcpServers": {
    "beamz": {
      "command": "C:/absolute/path/to/python.exe",
      "args": [
        "D:/absolute/path/to/beamz-mcp/server.py"
      ]
    }
  }
}
```

### Generic MCP client
Any MCP-compatible client should work with this pattern:

```json
{
  "mcpServers": {
    "beamz": {
      "command": "/absolute/path/to/python",
      "args": [
        "/absolute/path/to/beamz-mcp/server.py"
      ]
    }
  }
}
```

## Typical workflow
Once connected, an AI assistant can use tools like these:

- `create_scene(...)`
- `add_structure(...)`
- `add_circle(...)`
- `add_ring(...)`
- `add_source(...)`
- `add_monitor(...)`
- `run_simulation(...)`
- `get_transmission(...)`
- `plot_transmission(...)`
- `plot_field_snapshot(...)`

### Single-shot workflow
For the easiest experience, use the all-in-one helper `simulate_device`:

```json
{
  "domain": {
    "width_um": 8.0,
    "height_um": 4.0,
    "resolution_um": 0.04,
    "cladding_index": 1.44
  },
  "structures": [
    {
      "name": "wg",
      "x_um": 0.0,
      "y_um": 1.75,
      "width_um": 8.0,
      "height_um": 0.5,
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
      "x_um": 2.0,
      "y_um": 2.0,
      "span_um": 1.8
    },
    {
      "name": "out",
      "x_um": 6.5,
      "y_um": 2.0,
      "span_um": 1.8
    }
  ],
  "timeout_s": 60.0
}
```

This creates a scene, validates it, runs the simulation, and returns a transmission summary in one call.

### Step-by-step workflow
A more explicit workflow is:

1. `create_scene(...)`
2. Add geometry with `add_structure`, `add_circle`, `add_ring`, `add_polygon`
3. Add sources and monitors with `add_source`, `add_gaussian_source`, `add_monitor`, `add_mode_monitor`
4. Run `validate(scene_id)` and `solve_modes(scene_id)`
5. Call `run_simulation(scene_id, wait=True)`
6. Inspect results with `get_transmission(...)`, `get_mode_data(...)`, or `plot_field_snapshot(...)`

## Example prompts for AI clients
These are useful prompt patterns when you want the model to drive the simulation:

- "Design and simulate a straight silicon waveguide at 1550 nm. Compute transmission and show the resulting spectrum."
- "Model a Y-branch splitter and compare the output powers from each branch."
- "Create a microring resonator and extract the resonance wavelength and coupling behavior."
- "Solve the guided modes of a 3D waveguide and return the modal effective index."

## Validate the installation
This repository includes an end-to-end MCP test suite:

```bash
python test_e2e.py
```

It exercises the real stdio protocol and verifies that core tools are available and functioning.

## AI usage notice
This project was developed with the assistance of AI tool (Gemini 3.8 Flash) for validation, and debugging. The author remains responsible for the correctness, safety, and integrity of the code and outputs in this repository.