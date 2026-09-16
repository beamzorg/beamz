//! Exact material ownership for scenes made from rectilinear extrusions.
//!
//! Resolve priority in XY between consecutive object heights, then union by
//! material. The resulting prisms have disjoint interiors. Caps only retain
//! the area not occupied by the same material on the other side of the plane.
use geo::{BooleanOps, CoordsIter, MapCoords, MultiPolygon, Polygon as GeoPolygon, Rect};

use crate::geometry::{
    InterfaceAssessment, InterfaceEvidence, add_extrusion_side_evidence, strictly_inside,
};
use crate::{Aabb, ExtrudedPolygon, Geometry, Object, Polygon2, Scene};

pub(crate) struct ResolvedExtrusions {
    pub(crate) scene: Scene,
    caps: Vec<[Vec<Polygon2>; 2]>,
}

impl ResolvedExtrusions {
    pub(crate) fn build(scene: &Scene) -> Option<Self> {
        let mut inputs = Vec::new();
        let mut heights = Vec::new();
        for object in &scene.objects {
            let (polygon, z_min, z_max) = match &object.geometry {
                Geometry::Box { bounds } => (
                    Rect::new(
                        (bounds.min[0], bounds.min[1]),
                        (bounds.max[0], bounds.max[1]),
                    )
                    .to_polygon(),
                    bounds.min[2],
                    bounds.max[2],
                ),
                Geometry::ExtrudedPolygon(extrusion) => {
                    (extrusion.polygon.to_geo(), extrusion.z_min, extrusion.z_max)
                }
                _ => return None,
            };
            if z_max <= z_min || object.geometry.bounds().volume() == 0.0 {
                continue;
            }
            // Material table aliases do not introduce physical boundaries.
            let material_id = scene
                .materials
                .iter()
                .position(|material| *material == scene.materials[object.material_id])
                .unwrap();
            inputs.push((
                object,
                MultiPolygon(vec![polygon]),
                z_min,
                z_max,
                material_id,
            ));
            heights.extend([z_min, z_max]);
        }
        let coordinates = CoordinateRestore::new(
            inputs
                .iter()
                .flat_map(|(_, polygon, ..)| polygon.coords_iter()),
        );
        inputs.sort_by_key(|(object, ..)| (object.priority, object.id));
        heights.sort_by(f64::total_cmp);
        heights.dedup();
        let background = scene
            .materials
            .iter()
            .position(|material| *material == scene.materials[scene.background_material])
            .unwrap();
        let empty = MultiPolygon::<f64>(vec![]);
        let mut layers = Vec::new();
        for height in heights.windows(2) {
            let mut covered = empty.clone();
            let mut materials = vec![empty.clone(); scene.materials.len()];
            // The highest (priority, ID) owns overlaps, even when it explicitly
            // paints background material. Only discard background after masking.
            for (_, polygon, z_min, z_max, material_id) in inputs.iter().rev() {
                if *z_min >= height[1] || *z_max <= height[0] {
                    continue;
                }
                let visible = coordinates.restore(polygon.difference(&covered));
                if *material_id != background {
                    materials[*material_id] =
                        coordinates.restore(materials[*material_id].union(&visible));
                }
                covered = coordinates.restore(covered.union(polygon));
            }
            layers.push(materials);
        }
        let mut result = Self {
            scene: Scene {
                materials: scene.materials.clone(),
                objects: vec![],
                background_material: background,
            },
            caps: vec![],
        };
        for (layer_index, layer) in layers.iter().enumerate() {
            for (material_id, region) in layer.iter().enumerate() {
                for polygon in &region.0 {
                    let caps =
                        [layer_index.checked_sub(1), Some(layer_index + 1)].map(|neighbor| {
                            let adjacent = neighbor
                                .and_then(|i| layers.get(i))
                                .map_or(&empty, |layer| &layer[material_id]);
                            coordinates
                                .restore(MultiPolygon(vec![polygon.clone()]).difference(adjacent))
                                .0
                                .iter()
                                .map(from_geo)
                                .collect()
                        });
                    result.caps.push(caps);
                    result.scene.objects.push(Object {
                        id: result.scene.objects.len() as u64,
                        material_id,
                        priority: 0,
                        geometry: Geometry::ExtrudedPolygon(ExtrudedPolygon {
                            polygon: from_geo(polygon),
                            z_min: heights[layer_index],
                            z_max: heights[layer_index + 1],
                        }),
                    });
                }
            }
        }
        Some(result)
    }

    /// Only physical caps break local z invariance. Artificial height cuts in
    /// the resolved partition must not change the contour-path closure.
    pub(crate) fn has_exposed_cap(&self, volume: &Aabb, candidates: &[usize]) -> bool {
        candidates.iter().any(|&index| {
            let Geometry::ExtrudedPolygon(extrusion) = &self.scene.objects[index].geometry else {
                unreachable!("resolved ownership only contains extrusions");
            };
            [extrusion.z_min, extrusion.z_max]
                .into_iter()
                .zip(&self.caps[index])
                .any(|(height, polygons)| {
                    strictly_inside(height, volume.min[2], volume.max[2])
                        && polygons
                            .iter()
                            .any(|polygon| polygon.intersection_area(volume) > 0.0)
                })
        })
    }

    pub(crate) fn interface(
        &self,
        volume: &Aabb,
        candidates: &[usize],
        minimum_alignment: f64,
    ) -> InterfaceAssessment {
        let mut evidence = InterfaceEvidence::new(minimum_alignment);
        for &index in candidates {
            let Geometry::ExtrudedPolygon(extrusion) = &self.scene.objects[index].geometry else {
                unreachable!("resolved ownership only contains extrusions");
            };
            add_extrusion_side_evidence(
                &mut evidence,
                &extrusion.polygon,
                extrusion.z_min,
                extrusion.z_max,
                0.0,
                0.0,
                volume,
            );
            for (height, polygons) in [extrusion.z_min, extrusion.z_max]
                .into_iter()
                .zip(&self.caps[index])
            {
                if strictly_inside(height, volume.min[2], volume.max[2]) {
                    let area = polygons
                        .iter()
                        .map(|polygon| polygon.intersection_area(volume))
                        .sum();
                    evidence.add([0.0, 0.0, 1.0], area);
                }
            }
        }
        evidence.assess()
    }
}

fn from_geo(polygon: &GeoPolygon<f64>) -> Polygon2 {
    let ring = |line: &geo::LineString<f64>| {
        line.0[..line.0.len() - 1]
            .iter()
            .map(|point| [point.x, point.y])
            .collect()
    };
    Polygon2 {
        exterior: ring(polygon.exterior()),
        holes: polygon.interiors().iter().map(ring).collect(),
    }
}

/// `geo` overlays quantize XY to an integer lattice with roughly 29 bits per
/// half-extent. Restore original axis coordinates within a rounding tolerance
/// derived from the scene extent after each operation, so quantization cannot
/// move an input-aligned face
/// just inside a support and turn a laminar layer into a spurious corner.
struct CoordinateRestore {
    axes: [Vec<f64>; 2],
    tolerance: f64,
}

impl CoordinateRestore {
    fn new(points: impl Iterator<Item = geo::Coord<f64>>) -> Self {
        let mut axes = [vec![], vec![]];
        for point in points {
            axes[0].push(point.x);
            axes[1].push(point.y);
        }
        for axis in &mut axes {
            axis.sort_by(f64::total_cmp);
            axis.dedup();
        }
        let extent = axes
            .iter()
            .map(|axis| match (axis.first(), axis.last()) {
                (Some(first), Some(last)) => last - first,
                _ => 0.0,
            })
            .fold(0.0_f64, f64::max);
        let lattice_step = if extent > 0.0 {
            2.0_f64.powf((extent * 0.5).log2().trunc() - 29.0)
        } else {
            0.0
        };
        let magnitude = axes
            .iter()
            .flatten()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        Self {
            axes,
            tolerance: 2.0 * lattice_step + 16.0 * f64::EPSILON * magnitude,
        }
    }

    fn restore(&self, polygons: MultiPolygon<f64>) -> MultiPolygon<f64> {
        polygons.map_coords(|point| {
            let snap = |value: f64, axis: usize| {
                let values = &self.axes[axis];
                let i = values.partition_point(|v| *v < value);
                [i.checked_sub(1), Some(i)]
                    .into_iter()
                    .flatten()
                    .filter_map(|i| values.get(i))
                    .copied()
                    .filter(|v| (*v - value).abs() <= self.tolerance)
                    .min_by(|a, b| (a - value).abs().total_cmp(&(b - value).abs()))
                    .unwrap_or(value)
            };
            geo::Coord {
                x: snap(point.x, 0),
                y: snap(point.y, 1),
            }
        })
    }
}
