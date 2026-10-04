use serde::{Deserialize, Serialize};

use crate::{Aabb, RasterError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AxisLocation {
    Center,
    Edge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SupportSpec {
    axes: [AxisLocation; 3],
}

impl SupportSpec {
    pub(crate) const CELL: Self = Self::new([
        AxisLocation::Center,
        AxisLocation::Center,
        AxisLocation::Center,
    ]);
    pub(crate) const NODE: Self =
        Self::new([AxisLocation::Edge, AxisLocation::Edge, AxisLocation::Edge]);
    pub(crate) const EX: Self =
        Self::new([AxisLocation::Center, AxisLocation::Edge, AxisLocation::Edge]);
    pub(crate) const EY: Self =
        Self::new([AxisLocation::Edge, AxisLocation::Center, AxisLocation::Edge]);
    pub(crate) const EZ: Self =
        Self::new([AxisLocation::Edge, AxisLocation::Edge, AxisLocation::Center]);
    pub(crate) const HX: Self = Self::new([
        AxisLocation::Edge,
        AxisLocation::Center,
        AxisLocation::Center,
    ]);
    pub(crate) const HY: Self = Self::new([
        AxisLocation::Center,
        AxisLocation::Edge,
        AxisLocation::Center,
    ]);
    pub(crate) const HZ: Self = Self::new([
        AxisLocation::Center,
        AxisLocation::Center,
        AxisLocation::Edge,
    ]);
    const fn new(axes: [AxisLocation; 3]) -> Self {
        Self { axes }
    }

    pub(crate) fn logical_shape(self, grid: &Grid) -> [usize; 3] {
        let cells = grid.shape();
        std::array::from_fn(|axis| cells[axis] + usize::from(self.axes[axis] == AxisLocation::Edge))
    }

    /// Actual staggered sample location, not the midpoint of a clipped dual volume.
    pub(crate) fn point(self, grid: &Grid, index: [usize; 3]) -> [f64; 3] {
        let edges = [&grid.x_edges, &grid.y_edges, &grid.z_edges];
        std::array::from_fn(|axis| match self.axes[axis] {
            AxisLocation::Center => 0.5 * (edges[axis][index[axis]] + edges[axis][index[axis] + 1]),
            AxisLocation::Edge => {
                let value = edges[axis][index[axis]];
                if index[axis] + 1 == edges[axis].len() {
                    // The domain's final face owns its interior limit. Otherwise
                    // a box clipped to the domain would acquire an exterior-air
                    // sample and contaminate the joined periodic half supports.
                    if value == 0.0 {
                        -f64::from_bits(1)
                    } else if value > 0.0 {
                        f64::from_bits(value.to_bits() - 1)
                    } else {
                        f64::from_bits(value.to_bits() + 1)
                    }
                } else {
                    value
                }
            }
        })
    }

    pub(crate) fn volume(self, grid: &Grid, index: [usize; 3]) -> Aabb {
        let edges = [&grid.x_edges, &grid.y_edges, &grid.z_edges];
        let bounds: [[f64; 2]; 3] = std::array::from_fn(|axis| match self.axes[axis] {
            AxisLocation::Center => [edges[axis][index[axis]], edges[axis][index[axis] + 1]],
            AxisLocation::Edge => edge_dual_bounds(edges[axis], index[axis]),
        });
        Aabb {
            min: [bounds[0][0], bounds[1][0], bounds[2][0]],
            max: [bounds[0][1], bounds[1][1], bounds[2][1]],
        }
    }

    /// Complete dual volumes at periodic faces, expressed across the low seam.
    pub(crate) fn periodic_volume(
        self,
        grid: &Grid,
        index: [usize; 3],
        periodic: [bool; 3],
    ) -> Aabb {
        let mut volume = self.volume(grid, index);
        let edges = [&grid.x_edges, &grid.y_edges, &grid.z_edges];
        for axis in 0..3 {
            let e = edges[axis];
            let last = e.len() - 1;
            if periodic[axis]
                && self.axes[axis] == AxisLocation::Edge
                && (index[axis] == 0 || index[axis] == last)
            {
                volume.min[axis] = e[0] - 0.5 * (e[last] - e[last - 1]);
                volume.max[axis] = e[0] + 0.5 * (e[1] - e[0]);
            }
        }
        volume
    }

    pub(crate) fn periodic_point(
        self,
        grid: &Grid,
        index: [usize; 3],
        periodic: [bool; 3],
    ) -> [f64; 3] {
        let mut point = self.point(grid, index);
        let edges = [&grid.x_edges, &grid.y_edges, &grid.z_edges];
        for axis in 0..3 {
            if periodic[axis]
                && self.axes[axis] == AxisLocation::Edge
                && index[axis] + 1 == edges[axis].len()
            {
                point[axis] = edges[axis][0];
            }
        }
        point
    }
}

fn edge_dual_bounds(edges: &[f64], index: usize) -> [f64; 2] {
    let cells = edges.len() - 1;
    let lower = if index == 0 {
        edges[0]
    } else {
        edges[index - 1] + 0.5 * (edges[index] - edges[index - 1])
    };
    let upper = if index == cells {
        edges[cells]
    } else {
        edges[index] + 0.5 * (edges[index + 1] - edges[index])
    };
    [lower, upper]
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Grid {
    pub x_edges: Vec<f64>,
    pub y_edges: Vec<f64>,
    pub z_edges: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct UniformGrid {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub shape: [usize; 3],
}

impl UniformGrid {
    pub fn build(self) -> Result<Grid> {
        if self.shape.contains(&0) {
            return Err(RasterError::InvalidGrid(
                "all uniform-grid dimensions must be nonzero".into(),
            ));
        }
        let mut axes = [Vec::new(), Vec::new(), Vec::new()];
        for (axis, axis_edges) in axes.iter_mut().enumerate() {
            let lo = self.min[axis];
            let hi = self.max[axis];
            if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                return Err(RasterError::InvalidGrid(format!(
                    "axis {axis} bounds must be finite and increasing"
                )));
            }
            let n = self.shape[axis];
            *axis_edges = (0..=n)
                .map(|i| lo + (hi - lo) * (i as f64) / (n as f64))
                .collect();
        }
        Grid::new(axes[0].clone(), axes[1].clone(), axes[2].clone())
    }
}

impl Grid {
    pub fn new(x_edges: Vec<f64>, y_edges: Vec<f64>, z_edges: Vec<f64>) -> Result<Self> {
        let result = Self {
            x_edges,
            y_edges,
            z_edges,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<()> {
        validate_axis("x", &self.x_edges)?;
        validate_axis("y", &self.y_edges)?;
        validate_axis("z", &self.z_edges)?;
        Ok(())
    }

    pub fn shape(&self) -> [usize; 3] {
        [
            self.x_edges.len() - 1,
            self.y_edges.len() - 1,
            self.z_edges.len() - 1,
        ]
    }

    pub fn centers(edges: &[f64]) -> Vec<f64> {
        edges
            .windows(2)
            .map(|w| w[0] + 0.5 * (w[1] - w[0]))
            .collect()
    }

    pub fn domain(&self) -> [[f64; 2]; 3] {
        [
            [self.x_edges[0], *self.x_edges.last().unwrap()],
            [self.y_edges[0], *self.y_edges.last().unwrap()],
            [self.z_edges[0], *self.z_edges.last().unwrap()],
        ]
    }
}

fn validate_axis(name: &str, edges: &[f64]) -> Result<()> {
    if edges.len() < 2 {
        return Err(RasterError::InvalidGrid(format!(
            "{name} axis requires at least two edges"
        )));
    }
    if edges.iter().any(|v| !v.is_finite()) {
        return Err(RasterError::InvalidGrid(format!(
            "{name} edges must be finite"
        )));
    }
    if edges.windows(2).any(|w| w[1] <= w[0]) {
        return Err(RasterError::InvalidGrid(format!(
            "{name} edges must be strictly increasing"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn periodic_dual_volume_joins_nonuniform_end_cells_and_canonicalizes_point() {
        let grid = super::Grid::new(vec![0.0, 1.0, 3.0], vec![0.0, 2.0], vec![0.0, 1.0]).unwrap();
        let support = super::SupportSpec::EY;
        let periodic = [true, false, false];
        let low = support.periodic_volume(&grid, [0, 0, 0], periodic);
        let high = support.periodic_volume(&grid, [2, 0, 0], periodic);
        assert_eq!(low, high);
        assert_eq!(low.min[0], -1.0);
        assert_eq!(low.max[0], 0.5);
        assert_eq!(support.periodic_point(&grid, [0, 0, 0], periodic)[0], 0.0);
        assert_eq!(support.periodic_point(&grid, [2, 0, 0], periodic)[0], 0.0);
    }

    use super::*;

    #[test]
    fn uniform_grid_has_exact_requested_bounds() {
        let grid = UniformGrid {
            min: [-1.0, 0.0, 2.0],
            max: [1.0, 3.0, 6.0],
            shape: [4, 3, 2],
        }
        .build()
        .unwrap();
        assert_eq!(grid.shape(), [4, 3, 2]);
        assert_eq!(grid.domain(), [[-1.0, 1.0], [0.0, 3.0], [2.0, 6.0]]);
    }

    #[test]
    fn rejects_non_monotonic_edges() {
        assert!(Grid::new(vec![0.0, 1.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]).is_err());
    }

    #[test]
    fn validates_deserialized_grid_values() {
        let grid = Grid {
            x_edges: vec![],
            y_edges: vec![0.0, 1.0],
            z_edges: vec![0.0, 1.0],
        };
        assert!(grid.validate().is_err());
    }

    #[test]
    fn electric_and_magnetic_supports_match_the_beamz_yee_lattice() {
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [4.0, 3.0, 2.0],
            shape: [4, 3, 2],
        }
        .build()
        .unwrap();
        assert_eq!(SupportSpec::EX.logical_shape(&grid), [4, 4, 3]);
        assert_eq!(SupportSpec::EY.logical_shape(&grid), [5, 3, 3]);
        assert_eq!(SupportSpec::EZ.logical_shape(&grid), [5, 4, 2]);
        assert_eq!(SupportSpec::NODE.logical_shape(&grid), [5, 4, 3]);
        let hx = SupportSpec::new([
            AxisLocation::Edge,
            AxisLocation::Center,
            AxisLocation::Center,
        ]);
        let hy = SupportSpec::new([
            AxisLocation::Center,
            AxisLocation::Edge,
            AxisLocation::Center,
        ]);
        let hz = SupportSpec::new([
            AxisLocation::Center,
            AxisLocation::Center,
            AxisLocation::Edge,
        ]);
        assert_eq!(hx.logical_shape(&grid), [5, 3, 2]);
        assert_eq!(hy.logical_shape(&grid), [4, 4, 2]);
        assert_eq!(hz.logical_shape(&grid), [4, 3, 3]);
    }

    #[test]
    fn nonuniform_edge_supports_are_clipped_between_neighboring_centers() {
        let grid = Grid::new(vec![0.0, 1.0, 3.0], vec![0.0, 2.0], vec![0.0, 4.0]).unwrap();
        let hx = SupportSpec::new([
            AxisLocation::Edge,
            AxisLocation::Center,
            AxisLocation::Center,
        ]);
        assert_eq!(
            hx.volume(&grid, [0, 0, 0]),
            Aabb::new([0.0, 0.0, 0.0], [0.5, 2.0, 4.0]).unwrap()
        );
        assert_eq!(
            hx.volume(&grid, [1, 0, 0]),
            Aabb::new([0.5, 0.0, 0.0], [2.0, 2.0, 4.0]).unwrap()
        );
        assert_eq!(
            hx.volume(&grid, [2, 0, 0]),
            Aabb::new([2.0, 0.0, 0.0], [3.0, 2.0, 4.0]).unwrap()
        );
    }
}
