use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use parry3d_f64::{
    math::{Pose, Vector},
    query::intersection_test,
    shape::Triangle,
};
use serde::{Deserialize, Serialize};

use crate::geometry::{InterfaceEvidence, PlanarPatch};
use crate::{Aabb, RasterError, Result, Vec3};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct MeshReport {
    pub vertices: usize,
    pub triangles: usize,
    pub connected_components: usize,
    pub boundary_edges: usize,
    pub nonmanifold_edges: usize,
    pub inconsistent_edges: usize,
    pub degenerate_triangles: usize,
    pub self_intersections: usize,
    pub signed_volume: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriangleMesh {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    bounds: Aabb,
    #[serde(skip, default = "default_acceleration")]
    acceleration: Arc<OnceLock<MeshBvh>>,
}

fn default_acceleration() -> Arc<OnceLock<MeshBvh>> {
    Arc::new(OnceLock::new())
}

impl PartialEq for TriangleMesh {
    fn eq(&self, other: &Self) -> bool {
        self.vertices == other.vertices
            && self.triangles == other.triangles
            && self.bounds == other.bounds
    }
}

impl TriangleMesh {
    pub fn new(vertices: Vec<Vec3>, triangles: Vec<[u32; 3]>) -> Result<Self> {
        validate_closed_size(&vertices, &triangles)?;
        let (report, bounds) = validate_mesh(&vertices, &triangles)?;
        if report.boundary_edges > 0
            || report.nonmanifold_edges > 0
            || report.inconsistent_edges > 0
            || report.degenerate_triangles > 0
            || report.self_intersections > 0
        {
            return Err(RasterError::InvalidMesh(format!(
                "mesh is not a consistently oriented watertight manifold: {report:?}"
            )));
        }
        if !resolved_component_volumes(&vertices, &triangles, &report, bounds) {
            return Err(RasterError::InvalidMesh(
                "mesh has zero enclosed volume".into(),
            ));
        }
        let triangles = if report.signed_volume < 0.0 {
            triangles.into_iter().map(|[a, b, c]| [a, c, b]).collect()
        } else {
            triangles
        };
        Ok(Self {
            vertices,
            triangles,
            bounds,
            acceleration: default_acceleration(),
        })
    }

    pub fn inspect(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Result<MeshReport> {
        validate_mesh(vertices, triangles).map(|(report, _)| report)
    }

    pub fn validate(&self) -> Result<()> {
        validate_closed_size(&self.vertices, &self.triangles)?;
        let (report, actual_bounds) = validate_mesh(&self.vertices, &self.triangles)?;
        if actual_bounds != self.bounds {
            return Err(RasterError::InvalidMesh(
                "serialized mesh bounds do not match its vertices".into(),
            ));
        }
        if report.boundary_edges > 0
            || report.nonmanifold_edges > 0
            || report.inconsistent_edges > 0
            || report.degenerate_triangles > 0
            || report.self_intersections > 0
        {
            return Err(RasterError::InvalidMesh(format!(
                "mesh is not a consistently oriented watertight manifold: {report:?}"
            )));
        }
        if !resolved_component_volumes(&self.vertices, &self.triangles, &report, actual_bounds) {
            return Err(RasterError::InvalidMesh(
                "mesh has zero enclosed volume".into(),
            ));
        }
        Ok(())
    }

    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    pub fn contains(&self, point: Vec3) -> bool {
        if !self.bounds.contains_half_open(point) {
            return false;
        }
        // A nearly axis-aligned ray keeps BVH traversal narrow while the two
        // irrational perturbations avoid systematic vertex/edge coincidences.
        let direction = normalize([1.0, 1.234_567_89e-7, 9.876_543_21e-8]);
        let mesh_scale = (0..3)
            .map(|axis| (self.bounds.max[axis] - self.bounds.min[axis]).powi(2))
            .sum::<f64>()
            .sqrt();
        let minimum_distance = 256.0 * f64::EPSILON * mesh_scale;
        let mut hits = 0usize;
        let acceleration = self.acceleration();
        for index in acceleration.ray_candidates(point, direction) {
            let triangle = &self.triangles[index];
            let a = self.vertices[triangle[0] as usize];
            let b = self.vertices[triangle[1] as usize];
            let c = self.vertices[triangle[2] as usize];
            if ray_triangle_distance(point, direction, a, b, c, minimum_distance).is_some() {
                hits += 1;
            }
        }
        hits % 2 == 1
    }

    pub fn surface_may_intersect(&self, volume: &Aabb) -> bool {
        if !self.bounds.intersects(volume) {
            return false;
        }
        self.acceleration().aabb_candidates(volume).any(|index| {
            let triangle = &self.triangles[index];
            let a = self.vertices[triangle[0] as usize];
            let b = self.vertices[triangle[1] as usize];
            let c = self.vertices[triangle[2] as usize];
            triangle_crosses_interior(a, b, c, volume)
        })
    }

    /// Integrate a closed, connected polyhedral surface exactly (up to
    /// floating-point clipping error), without sampling mesh occupancy.
    pub fn exact_overlap_volume(&self, volume: &Aabb) -> Option<f64> {
        let acceleration = self.acceleration();
        // Disconnected/nested shells currently use parity containment. Their
        // independently chosen windings need not define the same signed solid,
        // so retain adaptive integration for those meshes.
        if !acceleration.single_component {
            return None;
        }
        if !self.bounds.intersects(volume) {
            return Some(0.0);
        }
        // Divergence theorem with F=(0,0,clamp(z-z_min,0,height)) inside the
        // support's XY rectangle. Side faces contribute zero flux; original
        // mesh faces above z_max contribute their projected area times height.
        // Query the whole upward column, including faces outside the support.
        let column = Aabb {
            min: volume.min,
            max: [volume.max[0], volume.max[1], self.bounds.max[2]],
        };
        let height = volume.max[2] - volume.min[2];
        let mut total = 0.0;
        let mut correction = 0.0;
        for index in acceleration.aabb_candidates(&column) {
            let triangle = self.triangles[index];
            // Work relative to the support to avoid subtracting large world
            // coordinates in the area/volume sum.
            let mut polygon: Vec<Vec3> = triangle
                .map(|vertex| sub(self.vertices[vertex as usize], volume.min))
                .to_vec();
            for axis in 0..2 {
                polygon = clip_polygon(&polygon, axis, 0.0, true);
                polygon = clip_polygon(&polygon, axis, volume.max[axis] - volume.min[axis], false);
            }
            polygon = clip_polygon(&polygon, 2, 0.0, true);
            let within = clip_polygon(&polygon, 2, height, false);
            let above = clip_polygon(&polygon, 2, height, true);
            // A triangle lying exactly on the top plane belongs to only one
            // piece. Otherwise the inclusive clipping would count it twice.
            let upper_flux = if polygon.iter().any(|point| point[2] > height) {
                projected_flux(&above, Some(height))
            } else {
                0.0
            };
            let flux = projected_flux(&within, None) + upper_flux;
            let adjusted = flux - correction;
            let next = total + adjusted;
            correction = (next - total) - adjusted;
            total = next;
        }
        // Both globally inward and outward orientations are accepted by the
        // mesh API. A connected closed surface has a consistent global sign.
        Some(total.abs().min(volume.volume()))
    }

    pub(crate) fn planar_patches(&self, volume: &Aabb) -> Vec<PlanarPatch> {
        self.acceleration()
            .aabb_candidates(volume)
            .filter_map(|index| {
                let face = self.triangles[index];
                let [a, b, c] = face.map(|vertex| self.vertices[vertex as usize]);
                (triangle_crosses_interior(a, b, c, volume)
                    && clipped_triangle_area(a, b, c, volume) > 0.0)
                    .then(|| PlanarPatch {
                        point: a,
                        normal: cross(sub(b, a), sub(c, a)),
                        plane_points: [a, b, c],
                    })
            })
            .collect()
    }

    pub(crate) fn interface_evidence(
        &self,
        volume: &Aabb,
        minimum_alignment: f64,
    ) -> InterfaceEvidence {
        let mut evidence = InterfaceEvidence::new(minimum_alignment);
        for index in self.acceleration().aabb_candidates(volume) {
            let triangle = self.triangles[index];
            let a = self.vertices[triangle[0] as usize];
            let b = self.vertices[triangle[1] as usize];
            let c = self.vertices[triangle[2] as usize];
            if !triangle_crosses_interior(a, b, c, volume) {
                continue;
            }
            let normal = cross(sub(b, a), sub(c, a));
            let clipped_area = clipped_triangle_area(a, b, c, volume);
            if clipped_area > 0.0 {
                evidence.add(normal, clipped_area);
            }
        }
        evidence
    }

    fn acceleration(&self) -> &MeshBvh {
        self.acceleration
            .get_or_init(|| MeshBvh::build(&self.vertices, &self.triangles))
    }
}

#[derive(Debug)]
struct MeshBvh {
    nodes: Vec<BvhNode>,
    single_component: bool,
}

#[derive(Debug)]
struct BvhNode {
    bounds: Aabb,
    kind: BvhKind,
}

#[derive(Debug)]
enum BvhKind {
    Leaf(Vec<usize>),
    Branch(usize, usize),
}

impl MeshBvh {
    fn build(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Self {
        let mut result = Self {
            nodes: Vec::new(),
            single_component: triangle_components(triangles) == 1,
        };
        let mut indices: Vec<usize> = (0..triangles.len()).collect();
        result.build_node(vertices, triangles, &mut indices);
        result
    }

    fn build_node(
        &mut self,
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
        indices: &mut [usize],
    ) -> usize {
        let bounds = indices
            .iter()
            .map(|index| {
                let triangle = triangles[*index];
                triangle_bounds(
                    vertices[triangle[0] as usize],
                    vertices[triangle[1] as usize],
                    vertices[triangle[2] as usize],
                )
            })
            .reduce(union_bounds)
            .expect("BVH nodes are never empty");
        let node_index = self.nodes.len();
        self.nodes.push(BvhNode {
            bounds,
            kind: BvhKind::Leaf(Vec::new()),
        });
        if indices.len() <= 8 {
            self.nodes[node_index].kind = BvhKind::Leaf(indices.to_vec());
            return node_index;
        }
        let extents = [
            bounds.max[0] - bounds.min[0],
            bounds.max[1] - bounds.min[1],
            bounds.max[2] - bounds.min[2],
        ];
        let axis = (0..3)
            .max_by(|left, right| extents[*left].total_cmp(&extents[*right]))
            .unwrap();
        indices.sort_unstable_by(|left, right| {
            triangle_centroid(vertices, triangles[*left])[axis]
                .total_cmp(&triangle_centroid(vertices, triangles[*right])[axis])
        });
        let middle = indices.len() / 2;
        let (left_indices, right_indices) = indices.split_at_mut(middle);
        let left = self.build_node(vertices, triangles, left_indices);
        let right = self.build_node(vertices, triangles, right_indices);
        self.nodes[node_index].kind = BvhKind::Branch(left, right);
        node_index
    }

    fn aabb_candidates<'a>(&'a self, query: &'a Aabb) -> impl Iterator<Item = usize> + 'a {
        let mut result = Vec::new();
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            // BVH bounds can be flat (e.g. a leaf containing only a mesh cap).
            // Use closed intersection here; callers decide whether touching
            // a support boundary contributes area, flux, or neither.
            if (0..3).any(|axis| {
                node.bounds.max[axis] < query.min[axis] || node.bounds.min[axis] > query.max[axis]
            }) {
                continue;
            }
            match &node.kind {
                BvhKind::Leaf(indices) => result.extend(indices),
                BvhKind::Branch(left, right) => {
                    stack.push(*left);
                    stack.push(*right);
                }
            }
        }
        result.into_iter()
    }

    fn ray_candidates(&self, origin: Vec3, direction: Vec3) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            if !ray_intersects_aabb(origin, direction, &node.bounds) {
                continue;
            }
            match &node.kind {
                BvhKind::Leaf(indices) => result.extend(indices),
                BvhKind::Branch(left, right) => {
                    stack.push(*left);
                    stack.push(*right);
                }
            }
        }
        result
    }
}

fn validate_closed_size(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Result<()> {
    if vertices.len() < 4 || triangles.len() < 4 {
        return Err(RasterError::InvalidMesh(
            "a closed triangle mesh requires at least four vertices and triangles".into(),
        ));
    }
    Ok(())
}

fn validate_mesh(vertices: &[Vec3], triangles: &[[u32; 3]]) -> Result<(MeshReport, Aabb)> {
    if vertices.iter().flatten().any(|value| !value.is_finite()) {
        return Err(RasterError::InvalidMesh(
            "mesh vertices must be finite".into(),
        ));
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let bounds = Aabb::new(min, max)?;
    let mut report = MeshReport {
        vertices: vertices.len(),
        triangles: triangles.len(),
        ..MeshReport::default()
    };
    let mut edges: HashMap<(u32, u32), (usize, i32)> = HashMap::new();
    let mut volume6 = 0.0;
    let volume_origin = bounds.min;
    for triangle in triangles {
        if triangle
            .iter()
            .any(|index| *index as usize >= vertices.len())
        {
            return Err(RasterError::InvalidMesh(
                "triangle index is out of range".into(),
            ));
        }
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0] {
            report.degenerate_triangles += 1;
            continue;
        }
        let a = vertices[triangle[0] as usize];
        let b = vertices[triangle[1] as usize];
        let c = vertices[triangle[2] as usize];
        let ab = sub(b, a);
        let ac = sub(c, a);
        let bc = sub(c, b);
        let normal = cross(ab, ac);
        let edge_scale2 = dot(ab, ab).max(dot(ac, ac)).max(dot(bc, bc));
        let relative_tolerance = 128.0 * f64::EPSILON;
        if dot(normal, normal) <= relative_tolerance.powi(2) * edge_scale2.powi(2) {
            report.degenerate_triangles += 1;
        }
        volume6 += dot(
            sub(a, volume_origin),
            cross(sub(b, volume_origin), sub(c, volume_origin)),
        );
        for [from, to] in [
            [triangle[0], triangle[1]],
            [triangle[1], triangle[2]],
            [triangle[2], triangle[0]],
        ] {
            let key = if from < to { (from, to) } else { (to, from) };
            let orientation = if from < to { 1 } else { -1 };
            let entry = edges.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += orientation;
        }
    }
    for (count, orientation) in edges.values() {
        if *count == 1 {
            report.boundary_edges += 1;
        } else if *count != 2 {
            report.nonmanifold_edges += 1;
        } else if *orientation != 0 {
            report.inconsistent_edges += 1;
        }
    }
    report.signed_volume = volume6 / 6.0;
    report.connected_components = triangle_components(triangles);
    if report.boundary_edges == 0
        && report.nonmanifold_edges == 0
        && report.inconsistent_edges == 0
        && report.degenerate_triangles == 0
    {
        report.self_intersections = count_self_intersections(vertices, triangles);
    }
    Ok((report, bounds))
}

fn count_self_intersections(vertices: &[Vec3], triangles: &[[u32; 3]]) -> usize {
    let acceleration = MeshBvh::build(vertices, triangles);
    let pose = Pose::IDENTITY;
    let mut intersections = 0;
    for (left_index, left) in triangles.iter().enumerate() {
        let left_bounds = triangle_bounds(
            vertices[left[0] as usize],
            vertices[left[1] as usize],
            vertices[left[2] as usize],
        );
        let left_shape = parry_triangle(vertices, *left);
        for right_index in acceleration.aabb_candidates(&left_bounds) {
            if right_index <= left_index {
                continue;
            }
            let right = triangles[right_index];
            if left.iter().any(|vertex| right.contains(vertex)) {
                continue;
            }
            let right_shape = parry_triangle(vertices, right);
            if intersection_test(&pose, &left_shape, &pose, &right_shape).unwrap_or(false) {
                intersections += 1;
            }
        }
    }
    intersections
}

fn parry_triangle(vertices: &[Vec3], triangle: [u32; 3]) -> Triangle {
    let point = |index: u32| Vector::from_array(vertices[index as usize]);
    Triangle::new(point(triangle[0]), point(triangle[1]), point(triangle[2]))
}

fn triangle_components(triangles: &[[u32; 3]]) -> usize {
    triangle_component_ids(triangles)
        .into_iter()
        .max()
        .map_or(0, |value| value + 1)
}

fn resolved_component_volumes(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    report: &MeshReport,
    bounds: Aabb,
) -> bool {
    if report.connected_components == 1 {
        return report.signed_volume.is_finite()
            && report.signed_volume.abs() > 128.0 * f64::EPSILON * bounds.volume();
    }
    // Independent shell orientations may cancel in the global signed sum.
    // Validate each shell relative to its own origin and bounds instead.
    let labels = triangle_component_ids(triangles);
    let count = report.connected_components;
    let mut origins = vec![None; count];
    let mut minima = vec![[f64::INFINITY; 3]; count];
    let mut maxima = vec![[f64::NEG_INFINITY; 3]; count];
    let mut sums = vec![0.0; count];
    let mut corrections = vec![0.0; count];
    for (triangle, label) in triangles.iter().zip(labels) {
        let [a, b, c] = triangle.map(|i| vertices[i as usize]);
        let origin = *origins[label].get_or_insert(a);
        for vertex in [a, b, c] {
            for axis in 0..3 {
                minima[label][axis] = minima[label][axis].min(vertex[axis]);
                maxima[label][axis] = maxima[label][axis].max(vertex[axis]);
            }
        }
        let term = dot(sub(a, origin), cross(sub(b, origin), sub(c, origin)));
        let adjusted = term - corrections[label];
        let next = sums[label] + adjusted;
        corrections[label] = (next - sums[label]) - adjusted;
        sums[label] = next;
    }
    (0..count).all(|index| {
        let volume = (0..3)
            .map(|axis| maxima[index][axis] - minima[index][axis])
            .product::<f64>();
        sums[index].is_finite() && sums[index].abs() > 6.0 * 128.0 * f64::EPSILON * volume
    })
}

fn triangle_component_ids(triangles: &[[u32; 3]]) -> Vec<usize> {
    // Surface shells are connected through edges. Two solids meeting at just
    // one vertex may have independent winding and must not share a signed
    // volume integral.
    let edges = |[a, b, c]: [u32; 3]| [[a, b], [b, c], [c, a]].map(|[u, v]| (u.min(v), u.max(v)));
    let mut edge_to_triangles: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (index, triangle) in triangles.iter().enumerate() {
        for edge in edges(*triangle) {
            edge_to_triangles.entry(edge).or_default().push(index);
        }
    }
    let mut seen = vec![usize::MAX; triangles.len()];
    let mut components = 0;
    for start in 0..triangles.len() {
        if seen[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = components;
        while let Some(index) = stack.pop() {
            for edge in edges(triangles[index]) {
                for &neighbor in &edge_to_triangles[&edge] {
                    if seen[neighbor] == usize::MAX {
                        seen[neighbor] = components;
                        stack.push(neighbor);
                    }
                }
            }
        }
        components += 1;
    }
    seen
}

fn triangle_bounds(a: Vec3, b: Vec3, c: Vec3) -> Aabb {
    Aabb {
        min: [
            a[0].min(b[0]).min(c[0]),
            a[1].min(b[1]).min(c[1]),
            a[2].min(b[2]).min(c[2]),
        ],
        max: [
            a[0].max(b[0]).max(c[0]),
            a[1].max(b[1]).max(c[1]),
            a[2].max(b[2]).max(c[2]),
        ],
    }
}

fn triangle_centroid(vertices: &[Vec3], triangle: [u32; 3]) -> Vec3 {
    let a = vertices[triangle[0] as usize];
    let b = vertices[triangle[1] as usize];
    let c = vertices[triangle[2] as usize];
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

fn union_bounds(left: Aabb, right: Aabb) -> Aabb {
    Aabb {
        min: [
            left.min[0].min(right.min[0]),
            left.min[1].min(right.min[1]),
            left.min[2].min(right.min[2]),
        ],
        max: [
            left.max[0].max(right.max[0]),
            left.max[1].max(right.max[1]),
            left.max[2].max(right.max[2]),
        ],
    }
}

fn triangle_box_intersects(a: Vec3, b: Vec3, c: Vec3, bounds: &Aabb) -> bool {
    let center = bounds.center();
    let extent = [
        0.5 * (bounds.max[0] - bounds.min[0]),
        0.5 * (bounds.max[1] - bounds.min[1]),
        0.5 * (bounds.max[2] - bounds.min[2]),
    ];
    let vertices = [sub(a, center), sub(b, center), sub(c, center)];
    let edges = [
        sub(vertices[1], vertices[0]),
        sub(vertices[2], vertices[1]),
        sub(vertices[0], vertices[2]),
    ];
    let mut axes = Vec::with_capacity(13);
    axes.extend([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    axes.push(cross(edges[0], edges[1]));
    for edge in edges {
        axes.push(cross(edge, [1.0, 0.0, 0.0]));
        axes.push(cross(edge, [0.0, 1.0, 0.0]));
        axes.push(cross(edge, [0.0, 0.0, 1.0]));
    }
    axes.into_iter().all(|axis| {
        let length = dot(axis, axis).sqrt();
        if !length.is_finite() || length == 0.0 {
            return true;
        }
        let axis = axis.map(|value| value / length);
        let projections = [
            dot(vertices[0], axis),
            dot(vertices[1], axis),
            dot(vertices[2], axis),
        ];
        let min = projections[0].min(projections[1]).min(projections[2]);
        let max = projections[0].max(projections[1]).max(projections[2]);
        let radius =
            extent[0] * axis[0].abs() + extent[1] * axis[1].abs() + extent[2] * axis[2].abs();
        max >= -radius && min <= radius
    })
}

fn triangle_crosses_interior(a: Vec3, b: Vec3, c: Vec3, bounds: &Aabb) -> bool {
    (0..3).all(|axis| {
        a[axis].max(b[axis]).max(c[axis]) > bounds.min[axis]
            && a[axis].min(b[axis]).min(c[axis]) < bounds.max[axis]
    }) && triangle_box_intersects(a, b, c, bounds)
}

// Signed XY area times the mean (linear) flux height on each fan triangle.
fn projected_flux(polygon: &[Vec3], constant_height: Option<f64>) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let a = polygon[0];
    (1..polygon.len() - 1)
        .map(|index| {
            let b = polygon[index];
            let c = polygon[index + 1];
            let area = 0.5 * cross(sub(b, a), sub(c, a))[2];
            area * constant_height.unwrap_or((a[2] + b[2] + c[2]) / 3.0)
        })
        .sum()
}

fn ray_intersects_aabb(origin: Vec3, direction: Vec3, bounds: &Aabb) -> bool {
    let mut near = 0.0f64;
    let mut far = f64::INFINITY;
    for axis in 0..3 {
        let inverse = 1.0 / direction[axis];
        let mut first = (bounds.min[axis] - origin[axis]) * inverse;
        let mut second = (bounds.max[axis] - origin[axis]) * inverse;
        if first > second {
            std::mem::swap(&mut first, &mut second);
        }
        near = near.max(first);
        far = far.min(second);
        if far < near {
            return false;
        }
    }
    far > 0.0
}

fn ray_triangle_distance(
    origin: Vec3,
    direction: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    minimum_distance: f64,
) -> Option<f64> {
    let edge1 = sub(b, a);
    let edge2 = sub(c, a);
    let p = cross(direction, edge2);
    let determinant = dot(edge1, p);
    let determinant_scale = dot(edge1, edge1).sqrt() * dot(edge2, edge2).sqrt();
    if determinant.abs() <= 64.0 * f64::EPSILON * determinant_scale {
        return None;
    }
    let inv = 1.0 / determinant;
    let tvec = sub(origin, a);
    let u = dot(tvec, p) * inv;
    if !(0.0..1.0).contains(&u) {
        return None;
    }
    let q = cross(tvec, edge1);
    let v = dot(direction, q) * inv;
    if v < 0.0 || u + v >= 1.0 {
        return None;
    }
    let distance = dot(edge2, q) * inv;
    (distance > minimum_distance).then_some(distance)
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(value: Vec3) -> Vec3 {
    let length = dot(value, value).sqrt();
    [value[0] / length, value[1] / length, value[2] / length]
}

fn clipped_triangle_area(a: Vec3, b: Vec3, c: Vec3, bounds: &Aabb) -> f64 {
    let mut polygon = vec![a, b, c];
    for axis in 0..3 {
        polygon = clip_polygon(&polygon, axis, bounds.min[axis], true);
        polygon = clip_polygon(&polygon, axis, bounds.max[axis], false);
        if polygon.len() < 3 {
            return 0.0;
        }
    }
    let origin = polygon[0];
    (1..polygon.len() - 1)
        .map(|index| {
            let area_vector = cross(sub(polygon[index], origin), sub(polygon[index + 1], origin));
            0.5 * dot(area_vector, area_vector).sqrt()
        })
        .sum()
}

fn clip_polygon(points: &[Vec3], axis: usize, bound: f64, keep_above: bool) -> Vec<Vec3> {
    if points.is_empty() {
        return Vec::new();
    }
    let inside = |point: Vec3| {
        if keep_above {
            point[axis] >= bound
        } else {
            point[axis] <= bound
        }
    };
    let mut output = Vec::with_capacity(points.len() + 1);
    let mut previous = points[points.len() - 1];
    let mut previous_inside = inside(previous);
    for &current in points {
        let current_inside = inside(current);
        if current_inside != previous_inside {
            let denominator = current[axis] - previous[axis];
            if denominator != 0.0 {
                let fraction = (bound - previous[axis]) / denominator;
                output.push(std::array::from_fn(|component| {
                    previous[component] + fraction * (current[component] - previous[component])
                }));
            }
        }
        if current_inside {
            output.push(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
    output
}

#[cfg(test)]
mod tests {
    use crate::geometry::InterfaceAssessment;

    use super::*;

    fn tetrahedron() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )
        .unwrap()
    }

    #[test]
    fn validates_and_classifies_tetrahedron() {
        let mesh = tetrahedron();
        assert!(mesh.contains([0.1, 0.1, 0.1]));
        assert!(!mesh.contains([0.8, 0.8, 0.8]));
    }

    #[test]
    fn exact_tetrahedron_clipping_matches_analytic_integral() {
        // Inclusion/exclusion of the simplex x+y+z <= 1, independently of
        // the surface-flux implementation. Exercise caps above the support,
        // partial XY clipping, exterior cells, translations, scales, windings.
        let edges: [f64; 8] = [-0.2, 0.0, 0.13, 0.37, 0.5, 0.79, 1.0, 1.2];
        for scale in [1e-9, 1.0, 1e9] {
            for reverse in [false, true] {
                let offset = [7.0 * scale, -3.0 * scale, 11.0 * scale];
                let mut mesh = tetrahedron();
                mesh.vertices = mesh
                    .vertices
                    .iter()
                    .map(|point| std::array::from_fn(|axis| offset[axis] + point[axis] * scale))
                    .collect();
                mesh.bounds = Aabb::new(offset, offset.map(|value| value + scale)).unwrap();
                if reverse {
                    mesh.triangles.iter_mut().for_each(|face| face.swap(1, 2));
                }
                for x in edges.windows(2) {
                    for y in edges.windows(2) {
                        for z in edges.windows(2) {
                            let intervals = [x, y, z];
                            let support = Aabb::new(
                                std::array::from_fn(|axis| {
                                    offset[axis] + scale * intervals[axis][0]
                                }),
                                std::array::from_fn(|axis| {
                                    offset[axis] + scale * intervals[axis][1]
                                }),
                            )
                            .unwrap();
                            let mut expected = 0.0;
                            if intervals.iter().all(|interval| interval[1] > 0.0) {
                                for corner in 0u32..8 {
                                    let remaining = 1.0
                                        - (0..3)
                                            .map(|axis| {
                                                intervals[axis][((corner >> axis) & 1) as usize]
                                                    .max(0.0)
                                            })
                                            .sum::<f64>();
                                    expected += if corner.count_ones() % 2 == 0 {
                                        1.0
                                    } else {
                                        -1.0
                                    } * remaining.max(0.0).powi(3)
                                        / 6.0;
                                }
                            }
                            let actual =
                                mesh.exact_overlap_volume(&support).unwrap() / scale.powi(3);
                            assert!(
                                (actual - expected).abs() < 2e-14,
                                "{intervals:?}: {actual} != {expected} at scale {scale}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn disconnected_shells_retain_parity_integration() {
        let base = tetrahedron();
        let mut vertices = base.vertices.clone();
        vertices.extend(base.vertices.iter().map(|point| point.map(|v| v + 2.0)));
        let mut triangles = base.triangles.clone();
        triangles.extend(base.triangles.iter().map(|face| face.map(|v| v + 4)));
        let mesh = TriangleMesh::new(vertices, triangles).unwrap();
        assert!(
            mesh.exact_overlap_volume(&Aabb::new([0.0; 3], [3.0; 3]).unwrap())
                .is_none()
        );
        assert!(mesh.contains([0.1; 3]));
        assert!(mesh.contains([2.1; 3]));
        assert!(!mesh.contains([1.5; 3]));
    }

    #[test]
    fn vertex_touching_shells_do_not_combine_independent_windings() {
        let mut vertices = tetrahedron().vertices;
        vertices.extend([[-0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]]);
        let mut triangles = tetrahedron().triangles;
        triangles.extend([[0, 5, 4], [0, 4, 6], [4, 5, 6], [5, 0, 6]]);
        let mesh = TriangleMesh::new(vertices, triangles).unwrap();
        assert!(
            mesh.exact_overlap_volume(&Aabb::new([-1.0; 3], [1.0; 3]).unwrap())
                .is_none()
        );
    }

    #[test]
    fn nested_shells_retain_parity_independent_of_winding() {
        for reverse in [false, true] {
            let base = tetrahedron();
            let mut vertices = base.vertices.clone();
            vertices.extend(base.vertices.iter().map(|p| p.map(|v| 0.1 + 0.2 * v)));
            let mut triangles = base.triangles.clone();
            triangles.extend(base.triangles.iter().map(|face| {
                let mut face = face.map(|v| v + 4);
                if reverse {
                    face.swap(1, 2);
                }
                face
            }));
            let mesh = TriangleMesh::new(vertices, triangles).unwrap();
            assert!(
                mesh.exact_overlap_volume(&Aabb::new([0.0; 3], [1.0; 3]).unwrap())
                    .is_none()
            );
            assert!(mesh.contains([0.05; 3]));
            assert!(!mesh.contains([0.12; 3]));
        }
    }

    #[test]
    fn containment_is_scale_invariant() {
        for scale in [1e-9, 1.0, 1e9] {
            let mesh = TriangleMesh::new(
                vec![
                    [0.0, 0.0, 0.0],
                    [scale, 0.0, 0.0],
                    [0.0, scale, 0.0],
                    [0.0, 0.0, scale],
                ],
                vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            )
            .unwrap();
            assert!(mesh.contains([0.1 * scale; 3]));
            assert!(!mesh.contains([0.8 * scale; 3]));
        }
    }

    #[test]
    fn noncoplanar_mesh_patches_are_not_averaged() {
        for scale in [1e-9, 1.0, 1e9] {
            let mesh = TriangleMesh::new(
                tetrahedron()
                    .vertices
                    .into_iter()
                    .map(|point| point.map(|value| value * scale))
                    .collect(),
                tetrahedron().triangles,
            )
            .unwrap();
            let support = Aabb::new([-0.1 * scale; 3], [0.6 * scale; 3]).unwrap();
            assert_eq!(
                mesh.interface_evidence(&support, 0.995).assess(),
                InterfaceAssessment::MultipleOrientations
            );
        }
    }

    #[test]
    fn rejects_open_mesh() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert!(TriangleMesh::new(vertices, vec![[0, 2, 1], [0, 1, 3], [1, 2, 3]]).is_err());
    }

    #[test]
    fn triangle_box_sat_rejects_bbox_only_overlap() {
        for scale in [1e-9, 1.0, 1e9] {
            let bounds =
                Aabb::new([0.8 * scale, 0.8 * scale, 0.0], [scale, scale, 0.2 * scale]).unwrap();
            assert!(!triangle_box_intersects(
                [0.0, 0.0, 0.1 * scale],
                [scale, 0.0, 0.1 * scale],
                [0.0, scale, 0.1 * scale],
                &bounds
            ));
        }
    }

    #[test]
    fn interface_weights_use_triangle_area_inside_the_support() {
        let bounds = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).unwrap();
        let area = clipped_triangle_area(
            [-10.0, -10.0, 0.5],
            [10.0, -10.0, 0.5],
            [0.0, 10.0, 0.5],
            &bounds,
        );
        assert!((area - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_intersecting_closed_components() {
        let mut vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.2, 0.2],
            [1.2, 0.2, 0.2],
            [0.2, 1.2, 0.2],
            [0.2, 0.2, 1.2],
        ];
        let mut triangles = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        triangles.extend([[4, 6, 5], [4, 5, 7], [5, 6, 7], [6, 4, 7]]);
        assert!(TriangleMesh::new(std::mem::take(&mut vertices), triangles).is_err());
    }
}
