"""Use the opt-in region raster workflow with Simulation's material-grid input.

This example keeps its prepared scene explicit. Gmsh mesh simplification is a
separate geometry choice and should be checked against the original mesh.
"""

from beamz import Simulation
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import Box, Grid, Material, Object, RasterOptions, Scene
from beamz.design.raster._regions import rasterize_regions


def main() -> None:
    substrate = Object(Box((0, 0, 0), (1e-6, 1e-6, 0.2e-6)), material_id=2, id=1)
    core = Object(
        Box((0.2e-6, 0.4e-6, 0.6e-6), (0.8e-6, 0.6e-6, 0.8e-6)),
        material_id=1,
        id=2,
    )
    scene = Scene(
        (Material(), Material(permittivity=12.0), Material(permittivity=2.1)),
        (substrate, core),
    )
    grid = Grid.uniform((0, 0, 0), (1e-6, 1e-6, 1e-6), (20, 20, 20))
    result = rasterize_regions(
        scene,
        grid,
        options=RasterOptions(),
        tile_x_cells=5,
        substrate_object=substrate,
    )
    material_grid = MaterialGrid.from_raster_result(result, dimensions=3)
    simulation = Simulation(material_grid=material_grid, run_time=1e-15)
    print(simulation.grid.shape)


if __name__ == "__main__":
    main()
