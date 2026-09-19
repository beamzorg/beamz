//! Contour-path line/surface coefficients for parallel dielectric interfaces.
//!
//! See beamz/design/raster/contour_path.md for the derivation and scope.
use crate::{Aabb, Geometry, Polygon2, RasterError, Result, Scene, SymmetricTensor};

pub(crate) fn validate(scene: &Scene) -> Result<()> {
    for material in &scene.materials {
        let epsilon = material.epsilon_r.0[0];
        if epsilon <= 0.0
            || !epsilon.is_finite()
            || material.epsilon_r != SymmetricTensor::isotropic(epsilon)
            || material.mu_r != SymmetricTensor::isotropic(1.0)
            || material.conductivity != SymmetricTensor::isotropic(0.0)
        {
            return Err(RasterError::InvalidMaterial(
                "contour_path requires positive isotropic, lossless, nonmagnetic dielectrics"
                    .into(),
            ));
        }
    }
    if scene.objects.iter().any(|object| {
        !matches!(
            object.geometry,
            Geometry::Box { .. } | Geometry::ExtrudedPolygon(_)
        )
    }) {
        return Err(RasterError::InvalidScene(
            "contour_path supports boxes and extruded polygons only".into(),
        ));
    }
    Ok(())
}

/// Length of a polygon section, including holes. Half-open crossings avoid
/// double counting vertices and give the same ownership convention as volumes.
fn section_length(polygon: &Polygon2, axis: usize, fixed: f64, lo: f64, hi: f64) -> f64 {
    let other = 1 - axis;
    let mut cuts = vec![lo, hi];
    for ring in std::iter::once(&polygon.exterior).chain(polygon.holes.iter()) {
        for i in 0..ring.len() {
            let a = ring[i];
            let b = ring[(i + 1) % ring.len()];
            if (a[other] <= fixed && fixed < b[other]) || (b[other] <= fixed && fixed < a[other]) {
                let value =
                    a[axis] + (fixed - a[other]) * (b[axis] - a[axis]) / (b[other] - a[other]);
                if lo < value && value < hi {
                    cuts.push(value);
                }
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    cuts.windows(2)
        .filter_map(|pair| {
            let mut point = [0.0; 2];
            point[other] = fixed;
            point[axis] = 0.5 * (pair[0] + pair[1]);
            polygon
                .contains_half_open(point)
                .then_some(pair[1] - pair[0])
        })
        .sum()
}

fn line_fraction(geometry: &Geometry, volume: &Aabb, point: [f64; 3], axis: usize) -> f64 {
    let Geometry::ExtrudedPolygon(extrusion) = geometry else {
        unreachable!("resolved extrusion")
    };
    let length = volume.max[axis] - volume.min[axis];
    if axis == 2 {
        if !extrusion.polygon.contains_half_open([point[0], point[1]]) {
            return 0.0;
        }
        (volume.max[2].min(extrusion.z_max) - volume.min[2].max(extrusion.z_min)).max(0.0) / length
    } else {
        if point[2] < extrusion.z_min || point[2] >= extrusion.z_max {
            return 0.0;
        }
        section_length(
            &extrusion.polygon,
            axis,
            point[1 - axis],
            volume.min[axis],
            volume.max[axis],
        ) / length
    }
}

fn surface_fraction(geometry: &Geometry, volume: &Aabb, point: [f64; 3], axis: usize) -> f64 {
    let Geometry::ExtrudedPolygon(extrusion) = geometry else {
        unreachable!("resolved extrusion")
    };
    if axis == 2 {
        if point[2] < extrusion.z_min || point[2] >= extrusion.z_max {
            return 0.0;
        }
        extrusion.polygon.intersection_area(volume)
            / ((volume.max[0] - volume.min[0]) * (volume.max[1] - volume.min[1]))
    } else {
        let other = 1 - axis;
        let width = section_length(
            &extrusion.polygon,
            other,
            point[axis],
            volume.min[other],
            volume.max[other],
        );
        let height =
            (volume.max[2].min(extrusion.z_max) - volume.min[2].max(extrusion.z_min)).max(0.0);
        width * height / ((volume.max[other] - volume.min[other]) * (volume.max[2] - volume.min[2]))
    }
}

pub(crate) fn coefficient(
    scene: &Scene,
    candidates: &[usize],
    volume: &Aabb,
    point: [f64; 3],
    axis: usize,
    normal: [f64; 3],
) -> f64 {
    let background = scene.materials[scene.background_material].epsilon_r.0[0];
    let center = candidates
        .iter()
        .map(|&i| &scene.objects[i])
        .find(|object| object.geometry.contains_half_open(point))
        .map_or(background, |object| {
            scene.materials[object.material_id].epsilon_r.0[0]
        });
    let mut surface_epsilon = background;
    let mut line_inverse = 1.0 / background;
    // Resolved objects have disjoint interiors and same-material seams have
    // already been removed from the interface evidence (#242).
    for &index in candidates {
        let object = &scene.objects[index];
        let epsilon = scene.materials[object.material_id].epsilon_r.0[0];
        surface_epsilon +=
            surface_fraction(&object.geometry, volume, point, axis) * (epsilon - background);
        line_inverse += line_fraction(&object.geometry, volume, point, axis)
            * (1.0 / epsilon - 1.0 / background);
    }
    let q = normal[axis] * normal[axis];
    ((1.0 - q) * surface_epsilon + q * center) / ((1.0 - q) + q * center * line_inverse)
}

/// Exact 2-D contour path in a locally z-invariant extrusion.
/// The caller checks exposed (not artificial partition) caps before calling. Separate paths
/// retain separate intersection normals, including polygonal curved interfaces.
/// Multiple parallel crossings are supported. Nonparallel sequential crossings
/// on one half-path have no single scalar boundary mapping and fall back.
pub(crate) fn extrusion_coefficient(
    scene: &Scene,
    candidates: &[usize],
    volume: &Aabb,
    point: [f64; 3],
    axis: usize,
) -> Option<f64> {
    if axis == 2 {
        // E_z is everywhere tangential to the sidewalls.
        return Some(coefficient(
            scene,
            candidates,
            volume,
            point,
            axis,
            [1., 0., 0.],
        ));
    }
    let flux = path_integral(scene, candidates, volume, point, 1 - axis, axis, true)?;
    let circulation = path_integral(scene, candidates, volume, point, axis, axis, false)?;
    Some(flux / circulation)
}

fn path_integral(
    scene: &Scene,
    candidates: &[usize],
    volume: &Aabb,
    point: [f64; 3],
    path_axis: usize,
    field_axis: usize,
    flux: bool,
) -> Option<f64> {
    let transverse = 1 - path_axis;
    let owner = |coordinate: f64| {
        let mut p = point;
        p[path_axis] = coordinate;
        candidates
            .iter()
            .map(|&i| &scene.objects[i])
            .find(|object| object.geometry.contains_half_open(p))
            .map_or(scene.background_material, |object| object.material_id)
    };
    let epsilon = |id: usize| scene.materials[id].epsilon_r.0[0];
    let center_epsilon = epsilon(owner(point[path_axis]));
    let mut crossings = Vec::new();
    for &index in candidates {
        let Geometry::ExtrudedPolygon(e) = &scene.objects[index].geometry else {
            return None;
        };
        if point[2] < e.z_min || point[2] >= e.z_max {
            continue;
        }
        for ring in std::iter::once(&e.polygon.exterior).chain(e.polygon.holes.iter()) {
            for i in 0..ring.len() {
                let a = ring[i];
                let b = ring[(i + 1) % ring.len()];
                if (a[transverse] <= point[transverse] && point[transverse] < b[transverse])
                    || (b[transverse] <= point[transverse] && point[transverse] < a[transverse])
                {
                    let c = a[path_axis]
                        + (point[transverse] - a[transverse]) * (b[path_axis] - a[path_axis])
                            / (b[transverse] - a[transverse]);
                    if volume.min[path_axis] < c && c < volume.max[path_axis] {
                        let dx = b[0] - a[0];
                        let dy = b[1] - a[1];
                        let length = dx.hypot(dy);
                        crossings.push((c, [-dy / length, dx / length]));
                    }
                }
            }
        }
    }
    crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut cuts = vec![
        volume.min[path_axis],
        point[path_axis],
        volume.max[path_axis],
    ];
    cuts.extend(crossings.iter().map(|c| c.0));
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let mut integral = 0.0;
    for pair in cuts.windows(2) {
        let middle = 0.5 * (pair[0] + pair[1]);
        let value = epsilon(owner(middle));
        let mut normal: Option<[f64; 2]> = None;
        for &(c, n) in &crossings {
            if c < middle.min(point[path_axis]) || c > middle.max(point[path_axis]) {
                continue;
            }
            // Ignore hidden seams: inspect the open intervals next to a cut.
            let k = cuts.binary_search_by(|v| v.total_cmp(&c)).ok()?;
            if k == 0 || k + 1 == cuts.len() {
                continue;
            }
            if epsilon(owner(0.5 * (cuts[k - 1] + c))) == epsilon(owner(0.5 * (cuts[k + 1] + c))) {
                continue;
            }
            if let Some(previous) = normal {
                if (previous[0] * n[0] + previous[1] * n[1]).abs() < 1.0 - 1e-12 {
                    return None;
                }
            } else {
                normal = Some(n);
            }
        }
        let q = normal.map_or(0.0, |n| n[field_axis].powi(2));
        let field_ratio = (1.0 - q) + q * center_epsilon / value;
        integral += (pair[1] - pair[0]) * field_ratio * if flux { value } else { 1.0 };
    }
    Some(integral / (volume.max[path_axis] - volume.min[path_axis]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sections_respect_holes_and_vertices() {
        let p = Polygon2::new(
            vec![[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]],
            vec![vec![[-0.1, -1.], [-0.1, 1.], [0.1, 1.], [0.1, -1.]]],
        )
        .unwrap();
        assert!((section_length(&p, 0, 0., -1., 1.) - 1.8).abs() < 1e-12);
        assert_eq!(section_length(&p, 0, 2., -1., 1.), 0.0);
        assert_eq!(section_length(&p, 0, -2., -1., 1.), 2.0);
    }
}
