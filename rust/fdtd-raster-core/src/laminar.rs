//! Exact local partitions for planar interfaces, including bounded junctions.
//!
//! Geometry certifies that every surface patch crossing the support lies on
//! one of these parallel planes. Each open slab therefore has one owner. This
//! permits exact clipping plus one ownership query per slab, including painter
//! order, without assuming that separate mesh regions are disjoint.

use crate::{Aabb, Scene, Vec3};

/// Certify one normal using only patches crossing the evidence support.
/// This query never changes the planes used for material-volume integration.
pub(crate) fn interface_normal(
    scene: &Scene,
    support: &Aabb,
    candidates: &[usize],
) -> Option<crate::geometry::InterfaceAssessment> {
    let mut normal = None;
    for &index in candidates {
        for patch in scene.objects[index].geometry.planar_patches(support)? {
            let other = unit(patch.normal)?;
            let first = *normal.get_or_insert(other);
            if length(cross(first, other)) > 256.0 * f64::EPSILON {
                return Some(crate::geometry::InterfaceAssessment::MultipleOrientations);
            }
        }
    }
    Some(match normal {
        Some(normal) => crate::geometry::InterfaceAssessment::Laminar(normal),
        None => crate::geometry::InterfaceAssessment::Missing,
    })
}

pub(crate) fn integrate(
    scene: &Scene,
    support: &Aabb,
    candidates: &[usize],
) -> Option<(Vec<f64>, Vec3, u64)> {
    let mut patches: Vec<crate::geometry::PlanarPatch> = Vec::new();
    let mut normal = None;
    for &index in candidates {
        for patch in scene.objects[index].geometry.planar_patches(support)? {
            let other = unit(patch.normal)?;
            let first = *normal.get_or_insert(other);
            if length(cross(first, other)) > 256.0 * f64::EPSILON {
                return None;
            }
            if !patches
                .iter()
                .any(|existing| existing.coplanar_with(&patch))
            {
                patches.push(patch);
            }
        }
    }
    let normal = normal?;
    let size = sub(support.max, support.min);
    let lower: f64 = (0..3)
        .map(|axis| (normal[axis] * size[axis]).min(0.0))
        .sum();
    let upper: f64 = (0..3)
        .map(|axis| (normal[axis] * size[axis]).max(0.0))
        .sum();
    let mut cuts = vec![lower, upper];
    cuts.extend(
        patches
            .iter()
            .map(|patch| dot(normal, sub(patch.point, support.min)))
            .filter(|offset| *offset > lower && *offset < upper),
    );
    cuts.sort_by(f64::total_cmp);
    // Every distinct cut can bound a real layer, even below a relative
    // tolerance. Redundant near-coplanar cuts are harmless extra partitions.
    cuts.dedup_by(|a, b| *a == *b);

    let mut fractions = vec![0.0; scene.materials.len()];
    let mut tests = 0;
    for pair in cuts.windows(2) {
        let mut faces = box_faces(size);
        faces = clip(faces, normal, pair[1]);
        faces = clip(faces, normal.map(|value| -value), -pair[0]);
        let Some((volume, center)) = measure(&faces) else {
            continue;
        };
        if volume <= 0.0 {
            continue;
        }
        let point = std::array::from_fn(|axis| support.min[axis] + center[axis]);
        let owner = candidates
            .iter()
            .map(|index| &scene.objects[*index])
            .filter(|object| object.geometry.contains_half_open(point))
            .max_by_key(|object| (object.priority, object.id))
            .map_or(scene.background_material, |object| object.material_id);
        fractions[owner] += volume / support.volume();
        tests += candidates.len() as u64;
    }
    let total: f64 = fractions.iter().sum();
    if !total.is_finite() || (total - 1.0).abs() > 1e-10 {
        return None;
    }
    for fraction in &mut fractions {
        *fraction /= total;
    }
    Some((fractions, normal, tests))
}

/// Exact volumes at planar junctions. Extending finite patches to whole planes
/// only over-partitions the support; ownership is constant in every open piece.
/// Bound arrangement complexity so curved or dense meshes retain adaptive work.
pub(crate) fn integrate_partition(
    scene: &Scene,
    support: &Aabb,
    candidates: &[usize],
) -> Option<(Vec<f64>, u64)> {
    let size = sub(support.max, support.min);
    let mut planes: Vec<(Vec3, f64)> = Vec::new();
    let mut patches: Vec<crate::geometry::PlanarPatch> = Vec::new();
    for &index in candidates {
        for patch in scene.objects[index].geometry.planar_patches(support)? {
            let mut normal = unit(patch.normal)?;
            let dominant = (0..3).max_by(|a, b| normal[*a].abs().total_cmp(&normal[*b].abs()))?;
            if normal[dominant] < 0.0 {
                normal = normal.map(|v| -v);
            }
            let offset = dot(normal, sub(patch.point, support.min));
            // The robust orientation predicate recognizes truly coplanar
            // triangles even when their rounded equations differ. No distance
            // tolerance is used, so distinct thin interfaces stay separate.
            if patches
                .iter()
                .any(|existing| existing.coplanar_with(&patch))
            {
                continue;
            }
            patches.push(patch);
            planes.push((normal, offset));
            if planes.len() > 12 {
                return None;
            }
        }
    }
    let mut pieces = vec![box_faces(size)];
    for (normal, offset) in planes {
        let mut next = Vec::new();
        for piece in pieces {
            // A plane touching a face must not duplicate the whole piece.
            let mut below = false;
            let mut above = false;
            for vertex in piece.iter().flatten() {
                let distance = dot(normal, *vertex) - offset;
                below |= distance < 0.0;
                above |= distance > 0.0;
            }
            if below && above {
                next.push(clip(piece.clone(), normal, offset));
                next.push(clip(piece, normal.map(|v| -v), -offset));
            } else {
                next.push(piece);
            }
            if next.len() > 256 {
                return None;
            }
        }
        pieces = next;
    }
    let mut fractions = vec![0.0; scene.materials.len()];
    let mut tests = 0;
    for piece in pieces {
        let Some((volume, center)) = measure(&piece) else {
            continue;
        };
        let point = std::array::from_fn(|axis| support.min[axis] + center[axis]);
        let owner = candidates
            .iter()
            .map(|index| &scene.objects[*index])
            .filter(|object| object.geometry.contains_half_open(point))
            .max_by_key(|object| (object.priority, object.id))
            .map_or(scene.background_material, |object| object.material_id);
        fractions[owner] += volume / support.volume();
        tests += candidates.len() as u64;
    }
    let total: f64 = fractions.iter().sum();
    if !total.is_finite() || (total - 1.0).abs() > 1e-10 {
        return None;
    }
    for fraction in &mut fractions {
        *fraction /= total;
    }
    Some((fractions, tests))
}

fn box_faces(size: Vec3) -> Vec<Vec<Vec3>> {
    let [x, y, z] = size;
    let vertices = [
        [0.0, 0.0, 0.0],
        [x, 0.0, 0.0],
        [x, y, 0.0],
        [0.0, y, 0.0],
        [0.0, 0.0, z],
        [x, 0.0, z],
        [x, y, z],
        [0.0, y, z],
    ];
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    .iter()
    .map(|face| face.iter().map(|index| vertices[*index]).collect())
    .collect()
}

/// Clip a convex polyhedron to dot(normal, point) <= offset. Face winding
/// is immaterial: measurement uses tetrahedra about an interior point.
fn clip(faces: Vec<Vec<Vec3>>, normal: Vec3, offset: f64) -> Vec<Vec<Vec3>> {
    let mut output = Vec::new();
    let mut cap = Vec::new();
    for face in faces {
        let mut polygon = Vec::new();
        let mut previous = *face.last().unwrap();
        let mut previous_distance = dot(normal, previous) - offset;
        for current in face {
            let distance = dot(normal, current) - offset;
            if (distance <= 0.0) != (previous_distance <= 0.0) {
                let fraction = previous_distance / (previous_distance - distance);
                let intersection = std::array::from_fn(|axis| {
                    previous[axis] + fraction * (current[axis] - previous[axis])
                });
                polygon.push(intersection);
                // The same edge is visited from both of its incident faces.
                if !cap.contains(&intersection) {
                    cap.push(intersection);
                }
            }
            if distance <= 0.0 {
                polygon.push(current);
            }
            previous = current;
            previous_distance = distance;
        }
        if polygon.len() >= 3 {
            output.push(polygon);
        }
    }
    if cap.len() >= 3 {
        let center = mean(cap.iter().copied());
        let axis = (0..3)
            .min_by(|a, b| normal[*a].abs().total_cmp(&normal[*b].abs()))
            .unwrap();
        let mut basis = [0.0; 3];
        basis[axis] = 1.0;
        let u = unit(cross(normal, basis)).unwrap();
        let v = cross(normal, u);
        let angle = |point| {
            let relative = sub(point, center);
            dot(relative, v).atan2(dot(relative, u))
        };
        cap.sort_by(|a, b| angle(*a).total_cmp(&angle(*b)));
        output.push(cap);
    }
    output
}

fn measure(faces: &[Vec<Vec3>]) -> Option<(f64, Vec3)> {
    if faces.is_empty() {
        return None;
    }
    // A positive convex combination of the vertices lies inside every slab.
    let center = mean(faces.iter().flatten().copied());
    let mut volume = 0.0;
    for face in faces {
        let a = sub(face[0], center);
        for index in 1..face.len() - 1 {
            volume += dot(
                a,
                cross(sub(face[index], center), sub(face[index + 1], center)),
            )
            .abs()
                / 6.0;
        }
    }
    Some((volume, center))
}

fn mean(points: impl Iterator<Item = Vec3>) -> Vec3 {
    let mut sum = [0.0; 3];
    let mut count = 0;
    for point in points {
        for axis in 0..3 {
            sum[axis] += point[axis];
        }
        count += 1;
    }
    sum.map(|value| value / count as f64)
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|axis| a[axis] - b[axis])
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|axis| a[axis] * b[axis]).sum()
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(value: Vec3) -> f64 {
    dot(value, value).sqrt()
}
fn unit(value: Vec3) -> Option<Vec3> {
    let norm = length(value);
    (norm.is_finite() && norm > 0.0).then(|| value.map(|v| v / norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Geometry, Material, Object};

    #[test]
    fn scene_coordinates_round_trip_without_moving_grid_aligned_faces() {
        let coordinates: Vec3 = [
            -3.0000000000000004e-9,
            1.2000000000000002e-8,
            7.000000000000001e-9,
        ];
        let decoded: Vec3 =
            serde_json::from_str(&serde_json::to_string(&coordinates).unwrap()).unwrap();
        assert_eq!(decoded.map(f64::to_bits), coordinates.map(f64::to_bits));
    }

    #[test]
    fn clipped_oblique_slabs_match_independent_cube_integral() {
        // Volume of x+y+z <= t in the unit cube, by inclusion/exclusion.
        let cdf = |t: f64| -> f64 {
            (0u32..8)
                .map(|corner| {
                    let count = corner.count_ones();
                    let sign = if count % 2 == 0 { 1.0 } else { -1.0 };
                    sign * (t - f64::from(count)).max(0.0).powi(3) / 6.0
                })
                .sum()
        };
        for scale in [1e-9, 1.0, 1e9] {
            for (low, high) in [(0.0, 0.37), (0.37, 1.37), (1.0, 2.0), (2.0, 2.8)] {
                let normal = unit([1.0; 3]).unwrap();
                let faces = clip(box_faces([scale; 3]), normal, high * scale / 3f64.sqrt());
                let faces = clip(faces, normal.map(|v| -v), -low * scale / 3f64.sqrt());
                let (volume, center) = measure(&faces).unwrap();
                assert!((volume / scale.powi(3) - (cdf(high) - cdf(low))).abs() < 1e-12);
                let projection = center.iter().sum::<f64>() / scale;
                assert!(projection > low && projection < high);
            }
        }
    }

    #[test]
    fn nonparallel_material_boundaries_have_exact_partition_without_laminar_smoothing() {
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![
                Object {
                    id: 1,
                    material_id: 1,
                    priority: 0,
                    geometry: Geometry::Box {
                        bounds: Aabb::new([-1.0; 3], [0.4, 2.0, 2.0]).unwrap(),
                    },
                },
                Object {
                    id: 2,
                    material_id: 0,
                    priority: 1,
                    geometry: Geometry::Box {
                        bounds: Aabb::new([-1.0; 3], [2.0, 0.6, 2.0]).unwrap(),
                    },
                },
            ],
            0,
        )
        .unwrap();
        let support = Aabb::new([0.0; 3], [1.0; 3]).unwrap();
        assert!(integrate(&scene, &support, &[0, 1]).is_none());
        let (fractions, _) = integrate_partition(&scene, &support, &[0, 1]).unwrap();
        assert!((fractions[1] - 0.16).abs() < 1e-12);
        assert!((fractions[0] - 0.84).abs() < 1e-12);
    }
    #[test]
    fn plane_partition_budget_is_bounded() {
        let objects = (1..=13)
            .map(|index| Object {
                id: index,
                material_id: 1,
                priority: index as i32,
                geometry: Geometry::Box {
                    bounds: Aabb::new([-1.0; 3], [index as f64 / 14.0, 2.0, 2.0]).unwrap(),
                },
            })
            .collect();
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            objects,
            0,
        )
        .unwrap();
        let support = Aabb::new([0.0; 3], [1.0; 3]).unwrap();
        assert!(integrate_partition(&scene, &support, &(0..12).collect::<Vec<_>>()).is_some());
        assert!(integrate_partition(&scene, &support, &(0..13).collect::<Vec<_>>()).is_none());
    }
}
