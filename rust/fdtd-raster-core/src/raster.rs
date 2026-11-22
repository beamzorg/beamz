use std::time::Instant;

#[path = "raster_planes.rs"]
mod planes;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::geometry::InterfaceAssessment;
use crate::grid::SupportSpec;
use crate::ownership::ResolvedExtrusions;
use crate::{Aabb, Grid, Material, Result, Scene, SymmetricTensor};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    Fast,
    #[default]
    Balanced,
    Reference,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SmoothingMode {
    #[default]
    Volume,
    FarjadpourDiagonal,
    FarjadpourFull,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputComponents {
    #[default]
    All,
    TwoDimensionalTm,
    TwoDimensionalTe,
}

impl OutputComponents {
    fn includes(self, component: Component) -> bool {
        match self {
            Self::All => true,
            Self::TwoDimensionalTm => {
                matches!(component, Component::Ez | Component::Hx | Component::Hy)
            }
            Self::TwoDimensionalTe => {
                matches!(component, Component::Ex | Component::Ey | Component::Hz)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntegrationOptions {
    pub smoothing: SmoothingMode,
    pub output_components: OutputComponents,
    /// Keep an exactly uniform scalar tensor as one value with its full logical shape.
    pub broadcast_uniform: bool,
    /// Emit the diagonal consumed by each Yee component for diagonal solver material grids.
    pub project_yee_diagonal: bool,
    /// Allow an immutable plane dictionary for large solver rasters.
    pub compact_planes: bool,
    /// Material IDs whose mixed supports use point ownership.
    pub staircase_materials: Vec<usize>,
    /// Complete electric/magnetic dual supports across these physical axes.
    pub periodic_axes: [bool; 3],
    minimum_depth: u8,
    max_depth: u8,
    fraction_error_tolerance: f64,
    minimum_normal_alignment: f64,
}

impl Default for IntegrationOptions {
    fn default() -> Self {
        Self::for_quality(Quality::Balanced)
    }
}

impl IntegrationOptions {
    pub fn for_quality(quality: Quality) -> Self {
        let (minimum_depth, max_depth, fraction_error_tolerance, minimum_normal_alignment) =
            match quality {
                Quality::Fast => (0, 0, 1.25e-1, 0.98),
                Quality::Balanced => (1, 2, 1e-2, 0.995),
                Quality::Reference => (3, 5, 1e-4, 0.999),
            };
        Self {
            smoothing: SmoothingMode::Volume,
            output_components: OutputComponents::All,
            broadcast_uniform: false,
            project_yee_diagonal: false,
            compact_planes: false,
            staircase_materials: Vec::new(),
            periodic_axes: [false; 3],
            minimum_depth,
            max_depth,
            fraction_error_tolerance,
            minimum_normal_alignment,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TensorArray {
    /// Compact constitutive components followed by storage order `(c, z, y, x)`.
    /// `c=1` is isotropic, `c=3` diagonal, and `c=6` symmetric packed.
    pub shape: [usize; 4],
    /// Component-major dense data, one broadcast scalar, or unique xy planes when `plane_ids` is populated.
    pub values: Vec<f32>,
    /// When present, index one physical xy plane per logical (component, z).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plane_ids: Vec<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DiagnosticSummary {
    pub samples: usize,
    pub background_samples: usize,
    pub uniform_samples: usize,
    pub exact_samples: usize,
    pub adaptive_samples: usize,
    pub unresolved_samples: usize,
    pub estimated_fraction_error_sum: f64,
    pub maximum_estimated_fraction_error: f64,
    pub smoothed_samples: usize,
    pub ambiguous_interface_samples: usize,
    pub fallback_multiple_orientations: usize,
    pub fallback_multiple_objects: usize,
    pub fallback_unresolved_geometry: usize,
    pub fallback_missing_surface_evidence: usize,
    pub candidate_object_tests: u64,
    pub validation_seconds: f64,
    pub indexing_seconds: f64,
    pub integration_seconds: f64,
    pub hashing_seconds: f64,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RasterResult {
    /// Compact epsilon tensors at Ex, Ey, and Ez supports.
    pub epsilon: [TensorArray; 3],
    /// Compact epsilon tensor on the dual-cell centers shared by cross terms.
    pub node_epsilon: TensorArray,
    /// Compact mu tensors at Hx, Hy, and Hz supports.
    pub mu: [TensorArray; 3],
    /// Compact electric-conductivity tensors at Ex, Ey, and Ez supports.
    pub conductivity: [TensorArray; 3],
    /// Authoritative compact cell tensors with 1, 3, or 6 leading components.
    pub cell_epsilon: TensorArray,
    pub cell_mu: TensorArray,
    pub cell_conductivity: TensorArray,
    pub diagnostics: DiagnosticSummary,
    pub scene_hash: String,
}

#[derive(Clone, Copy)]
enum Component {
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
    Node,
    Cell,
}

impl Component {
    fn support(self) -> SupportSpec {
        match self {
            Self::Ex => SupportSpec::EX,
            Self::Ey => SupportSpec::EY,
            Self::Ez => SupportSpec::EZ,
            Self::Hx => SupportSpec::HX,
            Self::Hy => SupportSpec::HY,
            Self::Hz => SupportSpec::HZ,
            Self::Node => SupportSpec::NODE,
            Self::Cell => SupportSpec::CELL,
        }
    }
}

#[derive(Clone, Copy)]
struct RasterDomain<'a> {
    scene: &'a Scene,
    index: &'a ObjectIndex,
    grid: &'a Grid,
    options: &'a IntegrationOptions,
}

struct ObjectIndex {
    domain: Aabb,
    dimensions: [usize; 3],
    bins: Vec<Vec<usize>>,
    object_bounds: Vec<Aabb>,
    resolved: Option<ResolvedExtrusions>,
}

impl ObjectIndex {
    fn build(scene: &Scene, grid: &Grid) -> Self {
        let [nx, ny, nz] = grid.shape();
        let dimensions = [
            nx.div_ceil(8).clamp(1, 64),
            ny.div_ceil(8).clamp(1, 64),
            nz.div_ceil(8).clamp(1, 64),
        ];
        let domain_values = grid.domain();
        let domain = Aabb {
            min: [
                domain_values[0][0],
                domain_values[1][0],
                domain_values[2][0],
            ],
            max: [
                domain_values[0][1],
                domain_values[1][1],
                domain_values[2][1],
            ],
        };
        let mut result = Self {
            domain,
            resolved: None,
            dimensions,
            bins: vec![Vec::new(); dimensions.iter().product()],
            object_bounds: scene
                .objects
                .iter()
                .map(|object| object.geometry.bounds())
                .collect(),
        };
        for object_index in 0..result.object_bounds.len() {
            if let Some(bounds) = result.object_bounds[object_index].intersection(&domain) {
                let ranges = result.bin_ranges(&bounds);
                for z in ranges[2].0..=ranges[2].1 {
                    for y in ranges[1].0..=ranges[1].1 {
                        for x in ranges[0].0..=ranges[0].1 {
                            let flat = result.flat(x, y, z);
                            result.bins[flat].push(object_index);
                        }
                    }
                }
            }
        }
        for bin in &mut result.bins {
            bin.sort_unstable();
            bin.dedup();
        }
        result
    }

    fn query(&self, bounds: &Aabb) -> Vec<usize> {
        let Some(bounds) = bounds.intersection(&self.domain) else {
            return Vec::new();
        };
        let ranges = self.bin_ranges(&bounds);
        let mut result = Vec::new();
        for z in ranges[2].0..=ranges[2].1 {
            for y in ranges[1].0..=ranges[1].1 {
                for x in ranges[0].0..=ranges[0].1 {
                    result.extend_from_slice(&self.bins[self.flat(x, y, z)]);
                }
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    fn bin_ranges(&self, bounds: &Aabb) -> [(usize, usize); 3] {
        std::array::from_fn(|axis| {
            let extent = self.domain.max[axis] - self.domain.min[axis];
            let count = self.dimensions[axis];
            let coordinate = |value: f64| {
                (((value - self.domain.min[axis]) / extent) * count as f64)
                    .floor()
                    .clamp(0.0, (count - 1) as f64) as usize
            };
            (coordinate(bounds.min[axis]), coordinate(bounds.max[axis]))
        })
    }

    fn flat(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.dimensions[0] * (y + self.dimensions[1] * z)
    }
}

#[derive(Clone, Copy)]
struct Sample<T> {
    values: T,
    meta: SampleMeta,
}
type CellSample = Sample<Material>;
// Volume samples use (epsilon, mu, conductivity). Projected Yee samples use
// the requested property order, so each batch carries only three scalars.
type ScalarSample = Sample<[f32; 3]>;

#[derive(Clone, Copy)]
struct SampleMeta {
    error: f32,
    path: SamplePath,
    candidate_tests: u64,
    smoothed: bool,
    fallback: Option<FallbackReason>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FallbackReason {
    MultipleOrientations,
    MultipleObjects,
    UnresolvedGeometry,
    MissingSurfaceEvidence,
}

#[derive(Clone, Copy, Debug)]
enum InterfaceClass {
    None,
    Laminar([f64; 3]),
    Ambiguous(FallbackReason),
}

#[derive(Clone, Copy)]
enum SamplePath {
    Background,
    Uniform,
    Exact,
    Adaptive,
}

pub fn rasterize(scene: &Scene, grid: &Grid, options: &IntegrationOptions) -> Result<RasterResult> {
    let start = Instant::now();
    scene.validate()?;
    grid.validate()?;
    let validation_seconds = start.elapsed().as_secs_f64();
    rasterize_impl(scene, grid, options, start, validation_seconds)
}

/// Rasterize a scene whose caller has already run [`Scene::validate`].
///
/// This entry point is intended for compiled language bindings that retain a
/// validated scene between grid sweeps.
pub fn rasterize_prevalidated(
    scene: &Scene,
    grid: &Grid,
    options: &IntegrationOptions,
) -> Result<RasterResult> {
    let start = Instant::now();
    grid.validate()?;
    rasterize_impl(scene, grid, options, start, 0.0)
}

fn rasterize_impl(
    scene: &Scene,
    grid: &Grid,
    options: &IntegrationOptions,
    start: Instant,
    validation_seconds: f64,
) -> Result<RasterResult> {
    if matches!(
        options.output_components,
        OutputComponents::TwoDimensionalTm | OutputComponents::TwoDimensionalTe
    ) && grid.shape()[2] != 1
    {
        return Err(crate::RasterError::InvalidGrid(
            "two-dimensional output requires exactly one z cell".into(),
        ));
    }
    let index_start = Instant::now();
    let original_scene = scene;
    let resolved =
        ResolvedExtrusions::build_with_identity(scene, !options.staircase_materials.is_empty());
    let mut index = ObjectIndex::build(resolved.as_ref().map_or(scene, |value| &value.scene), grid);
    index.resolved = resolved;
    let scene = index.resolved.as_ref().map_or(scene, |value| &value.scene);
    let indexing_seconds = index_start.elapsed().as_secs_f64();
    let integration_start = Instant::now();
    let mut diagnostics = DiagnosticSummary::default();
    let epsilon_components = material_component_count(scene, |material| material.epsilon_r);
    let mu_components = material_component_count(scene, |material| material.mu_r);
    let conductivity_components = material_component_count(scene, |material| material.conductivity);
    if options.smoothing == SmoothingMode::FarjadpourDiagonal
        && [epsilon_components, mu_components, conductivity_components].contains(&6)
    {
        return Err(crate::RasterError::InvalidMaterial(
            "farjadpour_diagonal cannot preserve intrinsic off-diagonal material coefficients; \
             use farjadpour_full or volume smoothing"
                .into(),
        ));
    }
    let scalar_volume = options.smoothing == SmoothingMode::Volume
        && epsilon_components == 1
        && mu_components == 1
        && conductivity_components == 1;
    let constant_unit_mu = !scene.materials.is_empty()
        && scene
            .materials
            .iter()
            .all(|material| material.mu_r == SymmetricTensor::isotropic(1.0));
    let maximum_components = if options.smoothing == SmoothingMode::FarjadpourDiagonal {
        3
    } else {
        6
    };
    let domain = RasterDomain {
        scene,
        index: &index,
        grid,
        options,
    };
    let mut epsilon = Vec::with_capacity(3);
    let mut conductivity = Vec::with_capacity(3);
    for component in [Component::Ex, Component::Ey, Component::Ez] {
        if options.output_components.includes(component) {
            if scalar_volume {
                let mut fields = raster_scalar_tensors(
                    domain,
                    component,
                    &mut diagnostics,
                    &[|s| s.values[0], |s| s.values[2]],
                );
                conductivity.push(fields.pop().unwrap());
                epsilon.push(fields.pop().unwrap());
            } else {
                let mut fields = raster_material_tensors(
                    domain,
                    component,
                    &mut diagnostics,
                    maximum_components,
                    &[|s| s.values.epsilon_r, |s| s.values.conductivity],
                );
                conductivity.push(fields.pop().unwrap());
                epsilon.push(fields.pop().unwrap());
            }
        } else {
            epsilon.push(omitted_tensor(component, grid));
            conductivity.push(omitted_tensor(component, grid));
        }
    }
    let mut mu = Vec::with_capacity(3);
    for component in [Component::Hx, Component::Hy, Component::Hz] {
        if options.output_components.includes(component) {
            if constant_unit_mu {
                mu.push(constant_unit_tensor(
                    component,
                    grid,
                    options.broadcast_uniform,
                ));
            } else if scalar_volume {
                mu.push(
                    raster_scalar_tensors(domain, component, &mut diagnostics, &[|s| s.values[1]])
                        .pop()
                        .unwrap(),
                );
            } else {
                mu.push(
                    raster_material_tensors(
                        domain,
                        component,
                        &mut diagnostics,
                        maximum_components,
                        &[|s| s.values.mu_r],
                    )
                    .pop()
                    .unwrap(),
                );
            }
        } else {
            mu.push(omitted_tensor(component, grid));
        }
    }
    let node_epsilon = if options.smoothing == SmoothingMode::FarjadpourFull
        && options.output_components != OutputComponents::TwoDimensionalTm
    {
        raster_material_tensors(
            domain,
            Component::Node,
            &mut diagnostics,
            maximum_components,
            &[|s| s.values.epsilon_r],
        )
        .pop()
        .unwrap()
    } else {
        omitted_tensor(Component::Node, grid)
    };
    let (cell_epsilon, cell_mu, cell_conductivity) = if scalar_volume {
        let mut fields = raster_scalar_tensors(
            domain,
            Component::Cell,
            &mut diagnostics,
            &[|s| s.values[0], |s| s.values[1], |s| s.values[2]],
        );
        let conductivity = fields.pop().unwrap();
        let mu = fields.pop().unwrap();
        (fields.pop().unwrap(), mu, conductivity)
    } else {
        let mut fields = raster_material_tensors(
            domain,
            Component::Cell,
            &mut diagnostics,
            maximum_components,
            &[
                |s| s.values.epsilon_r,
                |s| s.values.mu_r,
                |s| s.values.conductivity,
            ],
        );
        let conductivity = fields.pop().unwrap();
        let mu = fields.pop().unwrap();
        (fields.pop().unwrap(), mu, conductivity)
    };
    diagnostics.validation_seconds = validation_seconds;
    diagnostics.indexing_seconds = indexing_seconds;
    diagnostics.integration_seconds = integration_start.elapsed().as_secs_f64();
    let hash_start = Instant::now();
    let scene_hash = original_scene.stable_hash()?;
    diagnostics.hashing_seconds = hash_start.elapsed().as_secs_f64();
    diagnostics.elapsed_seconds = start.elapsed().as_secs_f64();
    Ok(RasterResult {
        epsilon: epsilon.try_into().unwrap(),
        node_epsilon,
        mu: mu.try_into().unwrap(),
        conductivity: conductivity.try_into().unwrap(),
        cell_epsilon,
        cell_mu,
        cell_conductivity,
        diagnostics,
        scene_hash,
    })
}

fn record_diagnostics<T>(diagnostics: &mut DiagnosticSummary, samples: &[Sample<T>], sites: usize) {
    diagnostics.samples += sites;
    // A certified background batch stores only its first representative.
    diagnostics.background_samples += sites - samples.len();
    for sample in samples {
        let meta = sample.meta;
        diagnostics.candidate_object_tests += meta.candidate_tests;
        match meta.path {
            SamplePath::Background => diagnostics.background_samples += 1,
            SamplePath::Uniform => diagnostics.uniform_samples += 1,
            SamplePath::Exact => diagnostics.exact_samples += 1,
            SamplePath::Adaptive => diagnostics.adaptive_samples += 1,
        }
        let estimated_error = f64::from(meta.error);
        diagnostics.unresolved_samples += usize::from(estimated_error > 0.0);
        diagnostics.estimated_fraction_error_sum += estimated_error;
        diagnostics.maximum_estimated_fraction_error = diagnostics
            .maximum_estimated_fraction_error
            .max(estimated_error);
        diagnostics.smoothed_samples += usize::from(meta.smoothed);
        if let Some(reason) = meta.fallback {
            diagnostics.ambiguous_interface_samples += 1;
            match reason {
                FallbackReason::MultipleOrientations => {
                    diagnostics.fallback_multiple_orientations += 1;
                }
                FallbackReason::MultipleObjects => {
                    diagnostics.fallback_multiple_objects += 1;
                }
                FallbackReason::UnresolvedGeometry => {
                    diagnostics.fallback_unresolved_geometry += 1;
                }
                FallbackReason::MissingSurfaceEvidence => {
                    diagnostics.fallback_missing_surface_evidence += 1;
                }
            }
        }
    }
}

fn omitted_tensor(component: Component, grid: &Grid) -> TensorArray {
    let logical_shape = component.support().logical_shape(grid);
    TensorArray {
        plane_ids: Vec::new(),
        shape: [0, logical_shape[2], logical_shape[1], logical_shape[0]],
        values: Vec::new(),
    }
}

fn constant_unit_tensor(component: Component, grid: &Grid, broadcast: bool) -> TensorArray {
    let [nx, ny, nz] = component.support().logical_shape(grid);
    TensorArray {
        plane_ids: Vec::new(),
        shape: [1, nz, ny, nx],
        values: vec![1.0; if broadcast { 1 } else { nx * ny * nz }],
    }
}

fn compact_component_count(
    samples: &[CellSample],
    property: impl Fn(&CellSample) -> SymmetricTensor,
    maximum_components: usize,
) -> usize {
    let mut components = 1;
    for sample in samples {
        let tensor = property(sample).0;
        let scale = tensor
            .iter()
            .fold(1.0_f64, |current, value| current.max(value.abs()));
        let tolerance = 256.0 * f64::EPSILON * scale;
        if maximum_components == 6 && tensor[3..].iter().any(|value| value.abs() > tolerance) {
            components = 6;
            break;
        }
        if (tensor[0] - tensor[1]).abs() > tolerance || (tensor[0] - tensor[2]).abs() > tolerance {
            components = 3;
        }
    }
    components
}

#[cfg(test)]
fn tensor_from_samples(
    shape: [usize; 3],
    samples: &[CellSample],
    property: impl Fn(&CellSample) -> SymmetricTensor,
    maximum_components: usize,
) -> TensorArray {
    let components = compact_component_count(samples, &property, maximum_components);
    let mut values = Vec::with_capacity(components * samples.len());
    for component in 0..components {
        values.extend(
            samples
                .iter()
                .map(|sample| property(sample).0[component] as f32),
        );
    }
    TensorArray {
        plane_ids: Vec::new(),
        shape: [components, shape[0], shape[1], shape[2]],
        values,
    }
}

fn material_component_count(
    scene: &Scene,
    property: impl Fn(&Material) -> SymmetricTensor,
) -> usize {
    let mut components = 1;
    for material in &scene.materials {
        let tensor = property(material).0;
        if tensor[3..].iter().any(|value| *value != 0.0) {
            return 6;
        }
        if tensor[0] != tensor[1] || tensor[0] != tensor[2] {
            components = 3;
        }
    }
    components
}

/// Sparse laminar mixtures on electric supports, using the same geometry
/// integration and ambiguity classification as the static Farjadpour path.
pub struct InterfaceSamples {
    pub component: &'static str,
    pub indices: Vec<usize>,
    pub fractions: Vec<f64>,
    pub normal_squared: Vec<f64>,
}

pub fn interface_samples(
    scene: &Scene,
    grid: &Grid,
    options: &IntegrationOptions,
) -> Vec<InterfaceSamples> {
    // Geometry ownership must distinguish media with identical static epsilon
    // but different dispersion. Prevent static-material coalescing by assigning
    // unique scalar labels before the geometry-only integration.
    let mut labelled = scene.clone();
    for (id, material) in labelled.materials.iter_mut().enumerate() {
        material.epsilon_r = SymmetricTensor::isotropic((id + 1) as f64);
    }
    let scene = &labelled;
    let resolved =
        ResolvedExtrusions::build_with_identity(scene, !options.staircase_materials.is_empty());
    let mut index = ObjectIndex::build(resolved.as_ref().map_or(scene, |v| &v.scene), grid);
    index.resolved = resolved;
    let scene = index.resolved.as_ref().map_or(scene, |v| &v.scene);
    let mut output = Vec::new();
    for (axis, (name, component)) in [
        ("Ex", Component::Ex),
        ("Ey", Component::Ey),
        ("Ez", Component::Ez),
    ]
    .into_iter()
    .enumerate()
    {
        if !options.output_components.includes(component) {
            continue;
        }
        let (_, samples) = raster_support(
            &index,
            grid,
            component,
            options,
            |volume, candidates, point| {
                let m = integrate_support_mixture(
                    scene, &index, grid, volume, candidates, point, options,
                );
                match (m.uniform_owner, m.interface, m.fractions) {
                    (None, InterfaceClass::Laminar(n), Some(f)) => {
                        let length2: f64 = n.iter().map(|x| x * x).sum();
                        let weight = n[axis] * n[axis] / length2;
                        if weight > 1e-12 {
                            Some((f, weight))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            },
        );
        let mut result = InterfaceSamples {
            component: name,
            indices: Vec::new(),
            fractions: Vec::new(),
            normal_squared: Vec::new(),
        };
        for (flat, sample) in samples.into_iter().enumerate() {
            if let Some((fractions, weight)) = sample {
                result.indices.push(flat);
                result.fractions.extend(fractions);
                result.normal_squared.push(weight);
            }
        }
        output.push(result);
    }
    output
}

fn raster_scalar_tensors(
    domain: RasterDomain<'_>,
    component: Component,
    diagnostics: &mut DiagnosticSummary,
    properties: &[fn(&ScalarSample) -> f32],
) -> Vec<TensorArray> {
    let logical = component.support().logical_shape(domain.grid);
    raster_scalar_layer(
        domain,
        component,
        diagnostics,
        properties,
        logical,
        0,
        |volume, candidates, point| {
            integrate_scalar(
                domain.scene,
                volume,
                candidates,
                point,
                domain.options,
                domain.index,
                domain.grid,
            )
        },
    )
}

fn raster_scalar_layer(
    domain: RasterDomain<'_>,
    component: Component,
    diagnostics: &mut DiagnosticSummary,
    properties: &[fn(&ScalarSample) -> f32],
    logical: [usize; 3],
    offset: usize,
    sample: impl Fn(&Aabb, &[usize], [f64; 3]) -> ScalarSample + Sync,
) -> Vec<TensorArray> {
    let length: usize = logical.iter().product();
    let mut fields: Vec<Vec<f32>> = properties
        .iter()
        .map(|_| {
            Vec::with_capacity(if domain.options.broadcast_uniform {
                1
            } else {
                length
            })
        })
        .collect();
    for begin in (0..length).step_by(65536) {
        let end = (begin + 65536).min(length);
        let (_, samples) = raster_support_range(
            domain.index,
            domain.grid,
            component,
            domain.options,
            &sample,
            offset + begin,
            offset + end,
        );
        let repeated = end - begin - samples.len();
        record_diagnostics(diagnostics, &samples, end - begin);
        for (values, property) in fields.iter_mut().zip(properties) {
            if domain.options.broadcast_uniform && (values.is_empty() || values.len() < begin) {
                let first = *values.first().unwrap_or(&property(&samples[0]));
                if samples
                    .iter()
                    .all(|sample| property(sample).to_bits() == first.to_bits())
                {
                    if values.is_empty() {
                        values.push(first);
                    }
                    continue;
                }
                values.resize(begin, first);
                values.reserve_exact(length - begin);
            }
            if repeated > 0 {
                values.resize(end, property(&samples[0]));
            } else {
                values.extend(samples.iter().map(property));
            }
        }
    }
    fields
        .into_iter()
        .map(|values| TensorArray {
            plane_ids: Vec::new(),
            shape: [1, logical[2], logical[1], logical[0]],
            values,
        })
        .collect()
}

// Integrate in bounded batches, retaining only compact constitutive values.
// Promotion is lossless: previous isotropic diagonals are duplicated and previous
// absent off-diagonal components are zero-filled, exactly as the full-grid pass.
fn raster_material_tensors(
    domain: RasterDomain<'_>,
    component: Component,
    diagnostics: &mut DiagnosticSummary,
    maximum_components: usize,
    properties: &[fn(&CellSample) -> SymmetricTensor],
) -> Vec<TensorArray> {
    let support = component.support();
    let [nx, ny, nz] = support.logical_shape(domain.grid);
    let schedule = (domain.options.compact_planes
        && domain.options.project_yee_diagonal
        && domain.options.smoothing == SmoothingMode::FarjadpourDiagonal
        && nz >= 8)
        .then(|| planes::schedule(domain.scene, domain.grid, support, domain.options))
        .flatten();
    if let Some((layers, ids, counts)) = schedule {
        let mut banks: Vec<planes::Bank> = (0..properties.len())
            .map(|_| planes::Bank::default())
            .collect();
        let mut summary = DiagnosticSummary::default();
        let mut discarded_bits = false;
        for (representative, times) in layers.into_iter().zip(counts) {
            let mut layer_diagnostics = DiagnosticSummary::default();
            let tensors = raster_material_layer(
                domain,
                component,
                &mut layer_diagnostics,
                maximum_components,
                properties,
                Some(representative),
                &mut discarded_bits,
            );
            if discarded_bits {
                break;
            }
            planes::accumulate(&mut summary, &layer_diagnostics, times);
            for (bank, tensor) in banks.iter_mut().zip(tensors) {
                bank.add(tensor);
            }
        }
        // A globally promoted tensor can expose bits locally below the
        // isotropy tolerance. Evaluate the original pass in that case.
        if !discarded_bits {
            planes::accumulate(diagnostics, &summary, 1);
            return banks
                .into_iter()
                .map(|bank| bank.finish([nz, ny, nx], &ids))
                .collect();
        }
    }
    raster_material_layer(
        domain,
        component,
        diagnostics,
        maximum_components,
        properties,
        None,
        &mut false,
    )
}

fn raster_material_layer(
    domain: RasterDomain<'_>,
    component: Component,
    diagnostics: &mut DiagnosticSummary,
    maximum_components: usize,
    properties: &[fn(&CellSample) -> SymmetricTensor],
    z: Option<usize>,
    discarded_bits: &mut bool,
) -> Vec<TensorArray> {
    let RasterDomain {
        scene,
        index,
        grid,
        options,
    } = domain;
    let projection = match component {
        Component::Ex | Component::Hx => Some(0),
        Component::Ey | Component::Hy => Some(1),
        Component::Ez | Component::Hz => Some(2),
        _ => None,
    }
    .filter(|_| {
        options.project_yee_diagonal && options.smoothing == SmoothingMode::FarjadpourDiagonal
    });
    let mut logical = component.support().logical_shape(grid);
    let plane_size = logical[0] * logical[1];
    let offset = z.unwrap_or(0) * plane_size;
    if z.is_some() {
        logical[2] = 1;
    }
    if let Some(axis) = projection {
        let output: [fn(&ScalarSample) -> f32; 3] =
            [|s| s.values[0], |s| s.values[1], |s| s.values[2]];
        return raster_scalar_layer(
            domain,
            component,
            diagnostics,
            &output[..properties.len()],
            logical,
            offset,
            |volume, candidates, point| {
                let (material, meta) =
                    integrate_material(scene, volume, candidates, point, options, index, grid);
                let cell = CellSample {
                    values: material,
                    meta,
                };
                let mut values = [0f32; 3];
                for (output, property) in values.iter_mut().zip(properties) {
                    *output = property(&cell).0[axis] as f32;
                }
                ScalarSample { values, meta }
            },
        );
    }
    let length: usize = logical.iter().product();
    let mut fields: Vec<Vec<f32>> = properties
        .iter()
        .map(|_| {
            if options.broadcast_uniform {
                Vec::new()
            } else {
                vec![0.0; length]
            }
        })
        .collect();
    let mut deferred: Vec<Vec<Vec<(usize, f32)>>> = properties
        .iter()
        .map(|_| (0..maximum_components).map(|_| Vec::new()).collect())
        .collect();
    for begin in (0..length).step_by(65536) {
        let end = (begin + 65536).min(length);
        let (_, samples) = raster_support_range(
            index,
            grid,
            component,
            options,
            |volume, candidates, point| {
                let (material, meta) =
                    integrate_material(scene, volume, candidates, point, options, index, grid);
                CellSample {
                    values: material,
                    meta,
                }
            },
            begin + offset,
            end + offset,
        );
        let repeated = end - begin - samples.len();
        record_diagnostics(diagnostics, &samples, end - begin);
        for ((values, missing), property) in fields.iter_mut().zip(&mut deferred).zip(properties) {
            // Defer allocating a volume while this property remains spatially constant.
            // Compare discarded components too: a later heterogeneous batch may promote
            // this compact tensor to three or six planes, including signed zero bits.
            if values.len() < length {
                let first = *values
                    .first()
                    .unwrap_or(&(property(&samples[0]).0[0] as f32));
                if samples.iter().all(|sample| {
                    let t = property(sample).0;
                    (t[0] as f32).to_bits() == first.to_bits()
                        && t[0].to_bits() == t[1].to_bits()
                        && t[0].to_bits() == t[2].to_bits()
                        && t[3..maximum_components].iter().all(|v| v.to_bits() == 0)
                }) {
                    if values.is_empty() {
                        values.push(first);
                    }
                    continue;
                }
                values.resize(length, first);
            }
            let previous = values.len() / length;
            let components = previous.max(compact_component_count(
                &samples,
                property,
                maximum_components,
            ));
            values.resize(components * length, 0.0);
            for (axis, exceptions) in missing
                .iter_mut()
                .enumerate()
                .take(components)
                .skip(previous)
            {
                let offset = axis * length;
                if axis < 3 {
                    values.copy_within(..begin, offset);
                }
                for (i, value) in exceptions.drain(..) {
                    values[offset + i] = value;
                }
            }
            // Keep actual rounded values for already-promoted components, even
            // when this batch alone would pass the compact/isotropic tolerance.
            for (axis, plane) in values.chunks_exact_mut(length).enumerate() {
                if repeated > 0 {
                    plane[begin..end].fill(property(&samples[0]).0[axis] as f32);
                } else {
                    for (value, sample) in plane[begin..end].iter_mut().zip(&samples) {
                        *value = property(sample).0[axis] as f32;
                    }
                }
            }
            // Preserve exceptional discarded bits for a possible later promotion.
            // Usually these lists are empty, including signed-zero-free isotropy.
            for (c, exceptions) in missing.iter_mut().enumerate().skip(components) {
                for (i, sample) in samples.iter().enumerate() {
                    let tensor = property(sample).0;
                    let value = tensor[c] as f32;
                    let default = if c < 3 { tensor[0] as f32 } else { 0.0 };
                    if value.to_bits() != default.to_bits() {
                        exceptions.extend((begin + i..=begin + i + repeated).map(|j| (j, value)));
                    }
                }
            }
        }
    }
    if z.is_some() {
        *discarded_bits |= deferred.iter().flatten().any(|entries| !entries.is_empty());
    }
    fields
        .into_iter()
        .map(|values| TensorArray {
            plane_ids: Vec::new(),
            shape: [
                (values.len() / length).max(1),
                logical[2],
                logical[1],
                logical[0],
            ],
            values,
        })
        .collect()
}

fn raster_support<T: Send + Clone>(
    index: &ObjectIndex,
    grid: &Grid,
    component: Component,
    options: &IntegrationOptions,
    integrate: impl Fn(&Aabb, &[usize], [f64; 3]) -> T + Sync,
) -> ([usize; 3], Vec<T>) {
    let length = component.support().logical_shape(grid).iter().product();
    let (shape, mut samples) =
        raster_support_range(index, grid, component, options, integrate, 0, length);
    if samples.len() < length {
        samples.resize(length, samples[0].clone());
    }
    (shape, samples)
}

fn raster_support_range<T: Send + Clone>(
    index: &ObjectIndex,
    grid: &Grid,
    component: Component,
    options: &IntegrationOptions,
    integrate: impl Fn(&Aabb, &[usize], [f64; 3]) -> T + Sync,
    begin: usize,
    end: usize,
) -> ([usize; 3], Vec<T>) {
    let logical_shape = component.support().logical_shape(grid);
    let shape_zyx = [logical_shape[2], logical_shape[1], logical_shape[0]];
    let support = component.support();
    let periodic = options.periodic_axes.iter().any(|&axis| axis);
    if !periodic && begin < end {
        let [nx, ny, _] = logical_shape;
        let xyz = |flat: usize| [flat % nx, flat / nx % ny, flat / (nx * ny)];
        let first = xyz(begin);
        let last = xyz(end - 1);
        let mut lo = first;
        let mut hi = last;
        // When a flat interval crosses a row or a plane, the enclosing box
        // spans the entire faster axis.
        if first[1..] != last[1..] {
            lo[0] = 0;
            hi[0] = nx - 1;
        }
        if first[2] != last[2] {
            lo[1] = 0;
            hi[1] = ny - 1;
        }
        let extent = Aabb {
            min: support.periodic_volume(grid, lo, options.periodic_axes).min,
            max: support.periodic_volume(grid, hi, options.periodic_axes).max,
        };
        if !index
            .query(&extent)
            .into_iter()
            .any(|i| index.object_bounds[i].intersects(&extent))
        {
            let volume = support.periodic_volume(grid, first, options.periodic_axes);
            let point = support.periodic_point(grid, first, options.periodic_axes);
            return (shape_zyx, vec![integrate(&volume, &[], point)]);
        }
    }
    // A wrapped high-face dual volume is expressed across the opposite seam.
    // Single-sample spans keep the search valid there as well.
    let span = if periodic { 1 } else { 64 };
    let samples = (begin..end)
        .into_par_iter()
        .with_min_len(64)
        .map_init(
            || (usize::MAX, Vec::new(), Vec::new(), false, [None::<T>, None]),
            |(previous, possible, candidates, uniform, cache), flat| {
                let x = flat % logical_shape[0];
                let row = flat / logical_shape[0];
                let y = row % logical_shape[1];
                let z = row / logical_shape[1];
                let offset = x % span;
                let first = flat - offset;
                if *previous != first {
                    let lo = x - offset;
                    let hi = (lo + span).min(logical_shape[0]) - 1;
                    let extent = Aabb {
                        min: support
                            .periodic_volume(grid, [lo, y, z], options.periodic_axes)
                            .min,
                        max: support
                            .periodic_volume(grid, [hi, y, z], options.periodic_axes)
                            .max,
                    };
                    *possible = index.query(&extent);
                    possible.retain(|&i| index.object_bounds[i].intersects(&extent));
                    // Every Yee support sees identical ownership and diagnostics when the
                    // entire span misses every surface and retains the same candidates.
                    *uniform = !periodic
                        && (possible.is_empty()
                            || index.resolved.as_ref().is_some_and(|r| {
                                possible.iter().all(|&i| {
                                    let bounds = index.object_bounds[i];
                                    (0..3).all(|a| {
                                        bounds.min[a] <= extent.min[a]
                                            && extent.max[a] <= bounds.max[a]
                                    }) && !r.scene.objects[i]
                                        .geometry
                                        .surface_may_intersect(&extent)
                                })
                            }));
                    cache[1] = None;
                    *previous = first;
                }
                // Keep the background across spans. Occupied spans can have a
                // different material even when each one is homogeneous.
                let cached = &mut cache[usize::from(!possible.is_empty())];
                if let Some(value) = cached.as_ref() {
                    return value.clone();
                }
                let volume = support.periodic_volume(grid, [x, y, z], options.periodic_axes);
                candidates.clear();
                candidates.extend(
                    possible
                        .iter()
                        .copied()
                        .filter(|&i| index.object_bounds[i].intersects(&volume)),
                );
                let sample = integrate(
                    &volume,
                    candidates,
                    support.periodic_point(grid, [x, y, z], options.periodic_axes),
                );
                if *uniform {
                    *cached = Some(sample.clone());
                }
                sample
            },
        )
        .collect();
    (shape_zyx, samples)
}

fn integrate_scalar(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    point: [f64; 3],
    options: &IntegrationOptions,
    index: &ObjectIndex,
    grid: &Grid,
) -> ScalarSample {
    let mixture = integrate_support_mixture(scene, index, grid, volume, candidates, point, options);
    let [epsilon, mu, conductivity] = if let Some(owner) = mixture.uniform_owner {
        let material = scene.materials[owner];
        [
            material.epsilon_r.0[0],
            material.mu_r.0[0],
            material.conductivity.0[0],
        ]
    } else {
        let mut values = [0.0; 3];
        for (material, fraction) in scene
            .materials
            .iter()
            .zip(mixture.fractions.as_ref().unwrap())
        {
            values[0] += material.epsilon_r.0[0] * fraction;
            values[1] += material.mu_r.0[0] * fraction;
            values[2] += material.conductivity.0[0] * fraction;
        }
        values
    };
    ScalarSample {
        values: [epsilon as f32, mu as f32, conductivity as f32],
        meta: SampleMeta {
            error: mixture.error as f32,
            path: mixture.path,
            candidate_tests: mixture.candidate_tests,
            smoothed: false,
            fallback: None,
        },
    }
}

fn integrate_material(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    point: [f64; 3],
    options: &IntegrationOptions,
    index: &ObjectIndex,
    grid: &Grid,
) -> (Material, SampleMeta) {
    let mixture = integrate_support_mixture(scene, index, grid, volume, candidates, point, options);
    let (material, smoothed, fallback) = if let Some(owner) = mixture.uniform_owner {
        (scene.materials[owner], false, None)
    } else {
        effective_material(
            scene,
            mixture.fractions.as_ref().unwrap(),
            options.smoothing,
            mixture.interface,
        )
    };
    (
        material,
        SampleMeta {
            error: mixture.error as f32,
            path: mixture.path,
            candidate_tests: mixture.candidate_tests,
            smoothed,
            fallback,
        },
    )
}

/// Geometry-only integration result consumed by every constitutive rule.
struct Mixture {
    fractions: Option<Vec<f64>>,
    interface: InterfaceClass,
    error: f64,
    path: SamplePath,
    candidate_tests: u64,
    uniform_owner: Option<usize>,
}

fn uniform_mixture(owner: usize, path: SamplePath, candidate_tests: u64) -> Mixture {
    Mixture {
        fractions: None,
        interface: InterfaceClass::None,
        error: 0.0,
        path,
        candidate_tests,
        uniform_owner: Some(owner),
    }
}

/// Assemble material fractions before applying a nonlinear interface law.
/// Averaging already-smoothed half supports is not harmonic averaging of the
/// complete periodic support. Static and dispersive paths share this assembly.
fn integrate_support_mixture(
    scene: &Scene,
    index: &ObjectIndex,
    grid: &Grid,
    volume: &Aabb,
    candidates: &[usize],
    point: [f64; 3],
    options: &IntegrationOptions,
) -> Mixture {
    let edges = [&grid.x_edges, &grid.y_edges, &grid.z_edges];
    let crosses =
        (0..3).any(|axis| options.periodic_axes[axis] && volume.min[axis] < edges[axis][0]);
    if !crosses {
        let mixture =
            integrate_mixture(scene, volume, candidates, options, index.resolved.as_ref());
        return staircase_mixture(scene, candidates, point, options, mixture);
    }
    let mut pieces = vec![(*volume, [0.0; 3])];
    for (axis, e) in edges.iter().enumerate() {
        if !options.periodic_axes[axis] || volume.min[axis] >= e[0] {
            continue;
        }
        let mut split = Vec::new();
        for (mut piece, shift) in pieces {
            let mut wrapped = piece;
            let mut wrapped_shift = shift;
            wrapped_shift[axis] += e[e.len() - 1] - e[0];
            wrapped.min[axis] += e[e.len() - 1] - e[0];
            wrapped.max[axis] = e[e.len() - 1];
            piece.min[axis] = e[0];
            split.extend([(wrapped, wrapped_shift), (piece, shift)]);
        }
        pieces = split;
    }
    let mut fractions = vec![0.0; scene.materials.len()];
    let mut error = 0.0;
    let mut candidate_tests = 0;
    let mut normal: Option<[f64; 3]> = None;
    let mut ambiguous = None;
    let mut adaptive = false;
    for (piece, shift) in &pieces {
        let mut candidates = index.query(piece);
        candidates
            .retain(|&candidate| scene.objects[candidate].geometry.bounds().intersects(piece));
        let mixture =
            integrate_mixture(scene, piece, &candidates, options, index.resolved.as_ref());
        let weight = piece.volume() / volume.volume();
        if let Some(owner) = mixture.uniform_owner {
            fractions[owner] += weight;
        } else if let Some(f) = mixture.fractions {
            for (target, value) in fractions.iter_mut().zip(f) {
                *target += weight * value;
            }
        }
        error += weight * mixture.error;
        candidate_tests += mixture.candidate_tests;
        adaptive |= matches!(mixture.path, SamplePath::Adaptive);
        // Curvature gates and representative normals must use the complete
        // support, not a narrower half whose artificial cut looks flatter.
        let assessment = if candidates.len() == 1
            && matches!(
                scene.objects[candidates[0]].geometry,
                crate::Geometry::Sphere { .. } | crate::Geometry::Cylinder { .. }
            )
            && matches!(mixture.interface, InterfaceClass::Laminar(_))
        {
            let whole = Aabb {
                min: std::array::from_fn(|axis| volume.min[axis] + shift[axis]),
                max: std::array::from_fn(|axis| volume.max[axis] + shift[axis]),
            };
            classify_interface(scene, &whole, &candidates, options, mixture.error)
        } else {
            mixture.interface
        };
        match assessment {
            InterfaceClass::Laminar(n) => {
                let length = n.iter().map(|x| x * x).sum::<f64>().sqrt();
                let n = n.map(|v| v / length);
                if let Some(previous) = normal {
                    let dot: f64 = previous.iter().zip(n).map(|(a, b)| a * b).sum();
                    if dot.abs() < options.minimum_normal_alignment {
                        ambiguous = Some(FallbackReason::MultipleOrientations);
                    }
                } else {
                    normal = Some(n);
                }
            }
            InterfaceClass::Ambiguous(reason) => ambiguous = Some(reason),
            InterfaceClass::None => {}
        }
    }
    let total: f64 = fractions.iter().sum();
    for fraction in &mut fractions {
        *fraction /= total;
    }
    let interface = if options.smoothing == SmoothingMode::Volume {
        InterfaceClass::None
    } else if let Some(reason) = ambiguous {
        InterfaceClass::Ambiguous(reason)
    } else if let Some(n) = normal {
        InterfaceClass::Laminar(n)
    } else {
        InterfaceClass::Ambiguous(FallbackReason::MissingSurfaceEvidence)
    };
    let mixture = Mixture {
        fractions: Some(fractions),
        interface,
        error,
        path: if adaptive {
            SamplePath::Adaptive
        } else {
            SamplePath::Exact
        },
        candidate_tests,
        uniform_owner: None,
    };
    // The seam point is canonicalized to the low face in every duplicate.
    // Query a finite interior neighborhood so boxes ending at the high face
    // cannot donate their exterior ownership to this point.
    let local = Aabb {
        min: point,
        max: std::array::from_fn(|axis| point[axis] + (edges[axis][1] - edges[axis][0]) * 1e-6),
    };
    staircase_mixture(scene, &index.query(&local), point, options, mixture)
}

/// A mixed support touching a designated metal uses one whole material at its
/// Yee location, for every coefficient and every constituent indicator raster.
fn staircase_mixture(
    scene: &Scene,
    candidates: &[usize],
    point: [f64; 3],
    options: &IntegrationOptions,
    mixture: Mixture,
) -> Mixture {
    if options.staircase_materials.is_empty() {
        return mixture;
    }
    // Always check the actual point, even when adaptive volume integration
    // misses a very small metal feature containing that Yee sample.
    let owner = owner_at(scene, candidates, point);
    let touches_metal = options.staircase_materials.iter().any(|&id| {
        owner == id
            || mixture.uniform_owner == Some(id)
            || mixture.fractions.as_ref().is_some_and(|f| f[id] > 1e-12)
    });
    if touches_metal {
        uniform_mixture(
            owner,
            mixture.path,
            mixture.candidate_tests + candidates.len() as u64,
        )
    } else {
        mixture
    }
}

fn integrate_mixture(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    options: &IntegrationOptions,
    resolved: Option<&ResolvedExtrusions>,
) -> Mixture {
    let mut candidate_tests = candidates.len() as u64;
    if candidates.is_empty() {
        return uniform_mixture(
            scene.background_material,
            SamplePath::Background,
            candidate_tests,
        );
    }
    if resolved.is_none() && candidates.len() > 1 {
        // Identical geometry is entirely owned by the highest (priority, ID),
        // including curved/mesh primitives on the adaptive path.
        let distinct: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&index| {
                let object = &scene.objects[index];
                !candidates.iter().any(|&other| {
                    let other = &scene.objects[other];
                    (other.priority, other.id) > (object.priority, object.id)
                        && other.geometry == object.geometry
                })
            })
            .collect();
        if distinct.len() != candidates.len() {
            return integrate_mixture(scene, volume, &distinct, options, None);
        }
        // An unrelated curved object elsewhere in the scene must not disable
        // exact ownership in supports containing only supported extrusions.
        if candidates.iter().all(|&index| {
            matches!(
                scene.objects[index].geometry,
                crate::Geometry::Box { .. } | crate::Geometry::ExtrudedPolygon(_)
            )
        }) {
            let local = Scene {
                materials: scene.materials.clone(),
                background_material: scene.background_material,
                objects: candidates
                    .iter()
                    .map(|&i| scene.objects[i].clone())
                    .collect(),
            };
            if let Some(local) = ResolvedExtrusions::build_with_identity(
                &local,
                !options.staircase_materials.is_empty(),
            ) {
                let indices: Vec<usize> = (0..local.scene.objects.len()).collect();
                return integrate_mixture(&local.scene, volume, &indices, options, Some(&local));
            }
        }
    }
    let surfaces = candidates
        .iter()
        .any(|index| scene.objects[*index].geometry.surface_may_intersect(volume));
    if !surfaces {
        let owner = owner_at(scene, candidates, volume.center());
        return uniform_mixture(owner, SamplePath::Uniform, candidate_tests);
    }
    if let Some(resolved) = resolved {
        let mut fractions = vec![0.0; scene.materials.len()];
        for &index in candidates {
            let object = &scene.objects[index];
            let overlap = object.geometry.exact_overlap_volume(volume).unwrap();
            fractions[object.material_id] += (overlap / volume.volume()).clamp(0.0, 1.0);
        }
        let occupied: f64 = fractions.iter().sum();
        // Boolean clipping has floating-point roundoff; keep a normalized
        // partition even at supports filled by several adjacent prisms.
        if occupied > 1.0 {
            for fraction in &mut fractions {
                *fraction /= occupied;
            }
        } else {
            fractions[scene.background_material] += 1.0 - occupied;
        }
        let interface = if options.smoothing == SmoothingMode::Volume {
            InterfaceClass::None
        } else {
            interface_class(resolved.interface(
                volume,
                candidates,
                options.minimum_normal_alignment,
            ))
        };
        return Mixture {
            fractions: Some(fractions),
            interface,
            error: 0.0,
            path: SamplePath::Exact,
            candidate_tests,
            uniform_owner: None,
        };
    }
    if candidates.len() == 1 {
        let object = &scene.objects[candidates[0]];
        if let Some(overlap) = object.geometry.exact_overlap_volume(volume) {
            let fraction = (overlap / volume.volume()).clamp(0.0, 1.0);
            let mut fractions = vec![0.0; scene.materials.len()];
            fractions[scene.background_material] = 1.0 - fraction;
            fractions[object.material_id] += fraction;
            return Mixture {
                fractions: Some(fractions),
                interface: classify_interface(scene, volume, candidates, options, 0.0),
                error: 0.0,
                path: SamplePath::Exact,
                candidate_tests,
                uniform_owner: None,
            };
        }
    }
    if let Some((fractions, _normal, tests)) = crate::laminar::integrate(scene, volume, candidates)
    {
        return Mixture {
            fractions: Some(fractions),
            interface: classify_interface(scene, volume, candidates, options, 0.0),
            error: 0.0,
            path: SamplePath::Exact,
            candidate_tests: candidate_tests + tests,
            uniform_owner: None,
        };
    }

    if let Some((fractions, tests)) = crate::laminar::integrate_partition(scene, volume, candidates)
    {
        return Mixture {
            fractions: Some(fractions),
            interface: classify_interface(scene, volume, candidates, options, 0.0),
            error: 0.0,
            path: SamplePath::Exact,
            candidate_tests: candidate_tests + tests,
            uniform_owner: None,
        };
    }
    let mut volumes = vec![0.0; scene.materials.len()];
    let mut estimated_error_volume = 0.0;
    let mut adaptive_output = AdaptiveOutput {
        accumulated: &mut volumes,
        estimated_error_volume: &mut estimated_error_volume,
        candidate_tests: &mut candidate_tests,
    };
    adaptive_integrate(
        scene,
        volume,
        candidates,
        volume.volume(),
        0,
        options,
        &mut adaptive_output,
    );
    let total = volume.volume();
    let fractions: Vec<f64> = volumes.iter().map(|occupied| occupied / total).collect();
    let error = (estimated_error_volume / total).min(1.0);
    Mixture {
        fractions: Some(fractions),
        interface: classify_interface(scene, volume, candidates, options, error),
        error,
        path: SamplePath::Adaptive,
        candidate_tests,
        uniform_owner: None,
    }
}

fn classify_interface(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    options: &IntegrationOptions,
    unresolved_fraction: f64,
) -> InterfaceClass {
    if options.smoothing == SmoothingMode::Volume {
        return InterfaceClass::None;
    }
    // The coarse/fine fraction disagreement is an a-posteriori estimate, not a
    // strict mathematical bound. Only reject smoothing when it says most of
    // the support remains unresolved.
    // The estimator is assembled from recursively scaled volumes.  Values at
    // exactly one half can land one ulp above or below the threshold when the
    // same geometry is expressed in another length unit.  Treat that boundary
    // consistently; only a materially larger unresolved share disables the
    // interface model.
    if unresolved_fraction > 0.5 + 256.0 * f64::EPSILON {
        return InterfaceClass::Ambiguous(FallbackReason::UnresolvedGeometry);
    }
    // Only the evidence query is inset: integration bounds and partition
    // planes remain exact. Include coordinate magnitude so the inset remains
    // representable after translation, and use each axis independently.
    let mut evidence_volume = *volume;
    for axis in 0..3 {
        let width = volume.max[axis] - volume.min[axis];
        let inset = 8.0
            * f64::EPSILON
            * width
                .max(volume.min[axis].abs())
                .max(volume.max[axis].abs());
        let lower = volume.min[axis] + inset;
        let upper = volume.max[axis] - inset;
        // Do not collapse a support whose width is itself only a few ULPs.
        if lower < upper {
            evidence_volume.min[axis] = lower;
            evidence_volume.max[axis] = upper;
        }
    }
    if candidates.len() != 1 {
        return match crate::laminar::interface_normal(scene, &evidence_volume, candidates) {
            Some(InterfaceAssessment::Laminar(normal)) => InterfaceClass::Laminar(normal),
            Some(_) => InterfaceClass::Ambiguous(FallbackReason::MultipleOrientations),
            None => InterfaceClass::Ambiguous(FallbackReason::MultipleObjects),
        };
    }
    interface_class(
        scene.objects[candidates[0]]
            .geometry
            .interface_evidence(&evidence_volume, options.minimum_normal_alignment)
            .assess(),
    )
}

fn interface_class(assessment: InterfaceAssessment) -> InterfaceClass {
    match assessment {
        InterfaceAssessment::Laminar(normal) => InterfaceClass::Laminar(normal),
        InterfaceAssessment::MultipleOrientations => {
            InterfaceClass::Ambiguous(FallbackReason::MultipleOrientations)
        }
        InterfaceAssessment::Missing => {
            InterfaceClass::Ambiguous(FallbackReason::MissingSurfaceEvidence)
        }
    }
}

struct AdaptiveOutput<'a> {
    accumulated: &'a mut [f64],
    estimated_error_volume: &'a mut f64,
    candidate_tests: &'a mut u64,
}

fn adaptive_integrate(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    root_volume: f64,
    depth: u8,
    options: &IntegrationOptions,
    output: &mut AdaptiveOutput<'_>,
) {
    let intersecting: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|index| scene.objects[*index].geometry.bounds().intersects(volume))
        .collect();
    let has_surface = intersecting
        .iter()
        .any(|index| scene.objects[*index].geometry.surface_may_intersect(volume));
    if !has_surface {
        output.accumulated[owner_at_counted(
            scene,
            &intersecting,
            volume.center(),
            output.candidate_tests,
        )] += volume.volume();
        return;
    }
    let (mut fine, coarse) =
        octant_material_volumes(scene, volume, &intersecting, output.candidate_tests);
    let mut estimated_error = mixture_difference(&coarse, &fine);
    let mut occupied = fine
        .iter()
        .filter(|occupied_volume| **occupied_volume > 32.0 * f64::EPSILON * volume.volume())
        .count();
    if occupied <= 1 {
        estimated_error = mixture_difference(&coarse, &fine);
        // A surface is known to cross this node. Agreement on one owner can be
        // aliasing rather than convergence, so force refinement. At the depth
        // limit only these aliased nodes pay for a denser estimate.
        if estimated_error <= 256.0 * f64::EPSILON * volume.volume() {
            estimated_error = volume.volume();
        }
    }
    let at_depth_limit = depth >= options.max_depth;
    if at_depth_limit && occupied <= 1 {
        fine = if options.max_depth == 0 {
            sparse_stratified_material_volumes(scene, volume, &intersecting, output.candidate_tests)
        } else {
            stratified_material_volumes(scene, volume, &intersecting, 3, output.candidate_tests)
        };
        estimated_error = mixture_difference(&coarse, &fine);
        occupied = fine
            .iter()
            .filter(|occupied_volume| **occupied_volume > 32.0 * f64::EPSILON * volume.volume())
            .count();
        if occupied <= 1 {
            estimated_error = volume.volume();
        }
    }
    if at_depth_limit
        || (depth >= options.minimum_depth
            && estimated_error / root_volume <= options.fraction_error_tolerance)
    {
        for (target, occupied_volume) in output.accumulated.iter_mut().zip(fine) {
            *target += occupied_volume;
        }
        *output.estimated_error_volume += estimated_error;
        return;
    }
    for child in 0..8 {
        adaptive_integrate(
            scene,
            &volume.child(child),
            &intersecting,
            root_volume,
            depth + 1,
            options,
            output,
        );
    }
}

fn octant_material_volumes(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    candidate_tests: &mut u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut all = vec![0.0; scene.materials.len()];
    // A checkerboard subset of the octants aliases any geometry extruded
    // along an axis: both histograms then agree identically, regardless of
    // accuracy. Compare with the independent parent-center estimate instead.
    let mut coarse = vec![0.0; scene.materials.len()];
    coarse[owner_at_counted(scene, candidates, volume.center(), candidate_tests)] = volume.volume();
    for child in 0_usize..8 {
        let owner = owner_at_counted(
            scene,
            candidates,
            volume.child(child).center(),
            candidate_tests,
        );
        all[owner] += volume.volume() / 8.0;
    }
    (all, coarse)
}

fn stratified_material_volumes(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    divisions: usize,
    candidate_tests: &mut u64,
) -> Vec<f64> {
    let mut result = vec![0.0; scene.materials.len()];
    let sample_count = divisions.pow(3) as f64;
    let sample_volume = volume.volume() / sample_count;
    let width: [f64; 3] = std::array::from_fn(|axis| volume.max[axis] - volume.min[axis]);
    for z in 0..divisions {
        for y in 0..divisions {
            for x in 0..divisions {
                let indices = [x, y, z];
                let point = std::array::from_fn(|axis| {
                    volume.min[axis] + (indices[axis] as f64 + 0.5) * width[axis] / divisions as f64
                });
                result[owner_at_counted(scene, candidates, point, candidate_tests)] +=
                    sample_volume;
            }
        }
    }
    result
}

fn sparse_stratified_material_volumes(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    candidate_tests: &mut u64,
) -> Vec<f64> {
    let mut result = vec![0.0; scene.materials.len()];
    let sample_volume = volume.volume() / 9.0;
    let width: [f64; 3] = std::array::from_fn(|axis| volume.max[axis] - volume.min[axis]);
    // A 3x3 Latin pattern covers every low/middle/high stratum on every axis
    // with nine points. It catches thin axis-aligned layers without making the
    // fast preset pay for the full 27-point reference fallback.
    for index in 0..9 {
        let strata = [index % 3, index / 3, (index % 3 + index / 3) % 3];
        let point = std::array::from_fn(|axis| {
            volume.min[axis] + (strata[axis] as f64 + 0.5) * width[axis] / 3.0
        });
        result[owner_at_counted(scene, candidates, point, candidate_tests)] += sample_volume;
    }
    result
}

fn mixture_difference(lhs: &[f64], rhs: &[f64]) -> f64 {
    0.5 * lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .sum::<f64>()
}

fn owner_at(scene: &Scene, candidates: &[usize], point: [f64; 3]) -> usize {
    candidates
        .iter()
        .map(|index| &scene.objects[*index])
        .filter(|object| object.geometry.contains_half_open(point))
        .max_by_key(|object| (object.priority, object.id))
        .map_or(scene.background_material, |object| object.material_id)
}

fn owner_at_counted(
    scene: &Scene,
    candidates: &[usize],
    point: [f64; 3],
    candidate_tests: &mut u64,
) -> usize {
    *candidate_tests += candidates.len() as u64;
    owner_at(scene, candidates, point)
}

fn effective_material(
    scene: &Scene,
    fractions: &[f64],
    smoothing: SmoothingMode,
    interface: InterfaceClass,
) -> (Material, bool, Option<FallbackReason>) {
    let occupied = fractions
        .iter()
        .filter(|fraction| **fraction > 32.0 * f64::EPSILON)
        .count();
    let normal = match interface {
        InterfaceClass::Laminar(normal) => Some(normal),
        InterfaceClass::None | InterfaceClass::Ambiguous(_) => None,
    };
    let can_smooth = smoothing != SmoothingMode::Volume && occupied > 1 && normal.is_some();
    let epsilon_r = if can_smooth {
        farjadpour_average(scene, fractions, normal.unwrap(), |material| {
            material.epsilon_r
        })
    } else {
        volume_average(scene, fractions, |material| material.epsilon_r)
    };
    let mu_r = if can_smooth {
        farjadpour_average(scene, fractions, normal.unwrap(), |material| material.mu_r)
    } else {
        volume_average(scene, fractions, |material| material.mu_r)
    };
    let conductivity = volume_average(scene, fractions, |material| material.conductivity);
    (
        Material {
            epsilon_r,
            mu_r,
            conductivity,
        },
        can_smooth,
        if smoothing == SmoothingMode::Volume || occupied <= 1 || can_smooth {
            None
        } else {
            Some(match interface {
                InterfaceClass::Ambiguous(reason) => reason,
                InterfaceClass::None | InterfaceClass::Laminar(_) => {
                    FallbackReason::MissingSurfaceEvidence
                }
            })
        },
    )
}

fn volume_average(
    scene: &Scene,
    fractions: &[f64],
    property: impl Fn(&Material) -> SymmetricTensor,
) -> SymmetricTensor {
    let mut result = SymmetricTensor([0.0; 6]);
    for (material, fraction) in scene.materials.iter().zip(fractions) {
        result.scaled_add(property(material), *fraction);
    }
    result
}

fn farjadpour_average(
    scene: &Scene,
    fractions: &[f64],
    normal: [f64; 3],
    property: impl Fn(&Material) -> SymmetricTensor,
) -> SymmetricTensor {
    let mut participating = scene
        .materials
        .iter()
        .zip(fractions)
        .filter(|(_, fraction)| **fraction > 32.0 * f64::EPSILON)
        .map(|(material, _)| property(material));
    if let Some(first) = participating.next() {
        if participating.all(|value| value == first) {
            return first;
        }
    }
    let basis = interface_basis(normal);
    let mut averaged_tau = [[0.0; 3]; 3];
    for (material, fraction) in scene.materials.iter().zip(fractions) {
        if *fraction == 0.0 {
            continue;
        }
        let local = rotate_to_local(property(material).matrix(), basis);
        let tau = tau_transform(local);
        for row in 0..3 {
            for col in 0..3 {
                averaged_tau[row][col] += fraction * tau[row][col];
            }
        }
    }
    let local = inverse_tau_transform(averaged_tau);
    SymmetricTensor::from_matrix(rotate_to_global(local, basis))
}

fn interface_basis(normal: [f64; 3]) -> [[f64; 3]; 3] {
    let length = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
    let n = normal.map(|value| value / length);
    let reference = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
        [1.0, 0.0, 0.0]
    } else if n[1].abs() <= n[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let tangent = normalize(cross(reference, n));
    let second = cross(n, tangent);
    [n, tangent, second]
}

fn normalize(value: [f64; 3]) -> [f64; 3] {
    let length = value.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
    value.map(|entry| entry / length)
}

fn cross(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn rotate_to_local(value: [[f64; 3]; 3], basis: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Basis vectors are rows, so local = B * global * B^T.
    multiply(multiply(basis, value), transpose(basis))
}

fn rotate_to_global(value: [[f64; 3]; 3], basis: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    multiply(multiply(transpose(basis), value), basis)
}

fn transpose(value: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|col| value[col][row]))
}

fn multiply(lhs: [[f64; 3]; 3], rhs: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|col| (0..3).map(|inner| lhs[row][inner] * rhs[inner][col]).sum())
    })
}

fn tau_transform(epsilon: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let pivot = epsilon[0][0];
    let mut tau = [[0.0; 3]; 3];
    tau[0][0] = -1.0 / pivot;
    for tangent in 1..3 {
        tau[0][tangent] = epsilon[0][tangent] / pivot;
        tau[tangent][0] = epsilon[tangent][0] / pivot;
    }
    for row in 1..3 {
        for col in 1..3 {
            tau[row][col] = epsilon[row][col] - epsilon[row][0] * epsilon[0][col] / pivot;
        }
    }
    tau
}

fn inverse_tau_transform(tau: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let pivot = tau[0][0];
    let mut epsilon = [[0.0; 3]; 3];
    epsilon[0][0] = -1.0 / pivot;
    for tangent in 1..3 {
        epsilon[0][tangent] = -tau[0][tangent] / pivot;
        epsilon[tangent][0] = -tau[tangent][0] / pivot;
    }
    for row in 1..3 {
        for col in 1..3 {
            epsilon[row][col] = tau[row][col] - tau[row][0] * tau[0][col] / pivot;
        }
    }
    epsilon
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{Geometry, Material, Object, Polygon2, UniformGrid};

    use super::*;

    fn box_scene(bounds: Aabb, epsilon: f64) -> Scene {
        Scene::new(
            vec![
                Material::default(),
                Material::new(epsilon, 1.0, 0.0).unwrap(),
            ],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::Box { bounds },
            }],
            0,
        )
        .unwrap()
    }

    #[test]
    fn plane_dictionary_preserves_rectilinear_material_and_mixed_geometry() {
        let mut scene = box_scene(
            Aabb::new([0.14, 0.19, 0.05], [0.93, 0.86, 0.91]).unwrap(),
            3.25,
        );
        scene.materials[1].mu_r = SymmetricTensor::isotropic(1.25);
        scene.objects.push(Object {
            id: 2,
            material_id: 1,
            priority: 2,
            geometry: Geometry::Sphere {
                center: [0.09, 0.11, 0.72],
                radius: 0.07,
            },
        });
        let grid = Grid::new(
            vec![0., 0.05, 0.1, 0.31, 0.56, 0.73, 0.87, 1.],
            vec![0., 0.1, 0.22, 0.43, 0.51, 0.77, 1.],
            (0..=64).map(|i| f64::from(i) / 64.).collect(),
        )
        .unwrap();
        let mut options = IntegrationOptions::for_quality(Quality::Balanced);
        options.project_yee_diagonal = true;
        options.smoothing = SmoothingMode::FarjadpourDiagonal;
        options.periodic_axes = [true, false, false];
        let reference = rasterize(&scene, &grid, &options).unwrap();
        options.compact_planes = true;
        let compact = rasterize(&scene, &grid, &options).unwrap();
        assert!(!compact.epsilon[0].plane_ids.is_empty());
        for (expected, got) in reference
            .epsilon
            .iter()
            .chain(&reference.mu)
            .chain(&reference.conductivity)
            .chain([
                &reference.cell_epsilon,
                &reference.cell_mu,
                &reference.cell_conductivity,
            ])
            .zip(
                compact
                    .epsilon
                    .iter()
                    .chain(&compact.mu)
                    .chain(&compact.conductivity)
                    .chain([
                        &compact.cell_epsilon,
                        &compact.cell_mu,
                        &compact.cell_conductivity,
                    ]),
            )
        {
            assert_eq!(got.shape, expected.shape);
            let plane = got.shape[2] * got.shape[3];
            for (i, &value) in expected.values.iter().enumerate() {
                let j = if got.values.len() == 1 {
                    0
                } else if got.plane_ids.is_empty() {
                    i
                } else {
                    got.plane_ids[i / plane] * plane + i % plane
                };
                assert_eq!(value.to_bits(), got.values[j].to_bits());
            }
        }
        assert_eq!(compact.diagnostics.samples, reference.diagnostics.samples);
        options.periodic_axes[2] = true;
        assert!(
            rasterize(&scene, &grid, &options)
                .unwrap()
                .epsilon
                .iter()
                .all(|t| t.plane_ids.is_empty())
        );
    }

    #[test]
    fn batched_tensor_promotion_matches_full_support_bits() {
        let mut scene = box_scene(Aabb::new([0.0, 0.0, 0.85], [1.0; 3]).unwrap(), 4.0);
        // First batch passes isotropic tolerance but has nonzero low-bit cross
        // terms. A later full tensor must restore those earlier values exactly.
        scene.materials[0].epsilon_r = SymmetricTensor([1.0, 1.0, 1.0, 1e-15, -0.0, 2e-15]);
        scene.materials[1].epsilon_r = SymmetricTensor([4.0, 5.0, 6.0, 0.1, 0.2, 0.3]);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [73, 41, 31],
        }
        .build()
        .unwrap();
        let index = ObjectIndex::build(&scene, &grid);
        let options = IntegrationOptions::default();
        let mut diagnostics = DiagnosticSummary::default();
        let actual = raster_material_tensors(
            RasterDomain {
                scene: &scene,
                index: &index,
                grid: &grid,
                options: &options,
            },
            Component::Cell,
            &mut diagnostics,
            6,
            &[|s| s.values.epsilon_r],
        )
        .pop()
        .unwrap();
        let (shape, samples) = raster_support(
            &index,
            &grid,
            Component::Cell,
            &options,
            |volume, candidates, point| {
                let (material, meta) =
                    integrate_material(&scene, volume, candidates, point, &options, &index, &grid);
                CellSample {
                    values: material,
                    meta,
                }
            },
        );
        let expected = tensor_from_samples(shape, &samples, |s| s.values.epsilon_r, 6);
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(
            actual
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut reference_diagnostics = DiagnosticSummary::default();
        record_diagnostics(&mut reference_diagnostics, &samples, samples.len());
        assert_eq!(diagnostics, reference_diagnostics);
    }

    #[test]
    fn yee_shapes_are_component_specific() {
        let scene = box_scene(Aabb::new([0.0; 3], [1.0; 3]).unwrap(), 4.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [2, 3, 4],
        }
        .build()
        .unwrap();
        let result = rasterize(&scene, &grid, &IntegrationOptions::default()).unwrap();
        assert_eq!(result.epsilon[0].shape, [1, 5, 4, 2]);
        assert_eq!(result.epsilon[1].shape, [1, 5, 3, 3]);
        assert_eq!(result.epsilon[2].shape, [1, 4, 4, 3]);
        assert_eq!(result.mu[0].shape, [1, 4, 3, 3]);
        assert_eq!(result.mu[1].shape, [1, 4, 4, 2]);
        assert_eq!(result.mu[2].shape, [1, 5, 3, 2]);
        assert!(result.epsilon[0].values.iter().all(|value| *value == 4.0));
        assert!(
            result
                .mu
                .iter()
                .all(|tensor| tensor.values.iter().all(|value| *value == 1.0))
        );
    }

    #[test]
    fn nonunit_mu_keeps_material_integration() {
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 2.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::Box {
                    bounds: Aabb::new([0.0; 3], [1.0; 3]).unwrap(),
                },
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [2, 3, 4],
        }
        .build()
        .unwrap();
        let result = rasterize(&scene, &grid, &IntegrationOptions::default()).unwrap();
        assert!(
            result
                .mu
                .iter()
                .all(|tensor| tensor.values.iter().all(|value| *value == 2.0))
        );
    }

    #[test]
    fn exact_box_fraction_is_volume_weighted() {
        let scene = box_scene(Aabb::new([0.25, 0.0, 0.0], [0.75, 1.0, 1.0]).unwrap(), 5.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1, 1, 1],
        }
        .build()
        .unwrap();
        let result = rasterize(&scene, &grid, &IntegrationOptions::default()).unwrap();
        assert_abs_diff_eq!(result.epsilon[1].values[0], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn adaptive_estimator_detects_axis_invariant_interfaces() {
        for axis in 0..3 {
            let mut upper = [2.0; 3];
            upper[axis] = 0.37;
            let scene = Scene::new(
                vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
                vec![Object {
                    id: 1,
                    material_id: 1,
                    priority: 0,
                    geometry: Geometry::Box {
                        bounds: Aabb::new([-1.0; 3], upper).unwrap(),
                    },
                }],
                0,
            )
            .unwrap();
            let mut tests = 0;
            let (fine, coarse) = octant_material_volumes(
                &scene,
                &Aabb::new([0.0; 3], [1.0; 3]).unwrap(),
                &[0],
                &mut tests,
            );
            assert_abs_diff_eq!(fine[1], 0.5, epsilon = 1e-15);
            assert_abs_diff_eq!(mixture_difference(&coarse, &fine), 0.5, epsilon = 1e-15);
            assert_eq!(tests, 9);
        }
    }

    #[test]
    fn curved_fraction_quality_is_estimate_driven_and_never_disappears() {
        let scene = Scene::new(
            vec![Material::default(), Material::new(2.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::Sphere {
                    center: [0.5; 3],
                    radius: 0.4,
                },
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let exact_fraction = 4.0 * std::f64::consts::PI * 0.4_f64.powi(3) / 3.0;
        let fast_options = IntegrationOptions::for_quality(Quality::Fast);
        let fast_mixture = integrate_mixture(
            &scene,
            &Aabb::new([0.0; 3], [1.0; 3]).unwrap(),
            &[0],
            &fast_options,
            None,
        );
        // One candidate lookup, eight octants, one alias check, and the
        // nine-point depth-limit fallback. This deterministic work budget guards
        // the fast preset against accidental sampling explosions in CI.
        assert_eq!(fast_mixture.candidate_tests, 19);
        let fast = rasterize(&scene, &grid, &fast_options).unwrap();
        let reference = rasterize(
            &scene,
            &grid,
            &IntegrationOptions::for_quality(Quality::Reference),
        )
        .unwrap();
        let fast_fraction = f64::from(fast.cell_epsilon.values[0] - 1.0);
        let reference_fraction = f64::from(reference.cell_epsilon.values[0] - 1.0);
        assert!(fast_fraction > 0.2);
        assert!(
            (reference_fraction - exact_fraction).abs() < (fast_fraction - exact_fraction).abs()
        );
        assert!(
            reference.diagnostics.maximum_estimated_fraction_error
                < fast.diagnostics.maximum_estimated_fraction_error
        );
    }

    #[test]
    fn tapered_mixture_is_scale_invariant() {
        fn scene(scale: f64) -> Scene {
            let polygon = Polygon2::new(
                vec![
                    [0.173 * scale, 0.193 * scale],
                    [0.817 * scale, 0.223 * scale],
                    [0.791 * scale, 0.843 * scale],
                    [0.201 * scale, 0.779 * scale],
                ],
                vec![],
            )
            .unwrap();
            Scene::new(
                vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
                vec![Object {
                    id: 1,
                    material_id: 1,
                    priority: 0,
                    geometry: Geometry::TaperedExtrudedPolygon(
                        crate::TaperedExtrudedPolygon::new(
                            polygon,
                            0.041 * scale,
                            0.937 * scale,
                            17.3,
                            0.37,
                        )
                        .unwrap(),
                    ),
                }],
                0,
            )
            .unwrap()
        }
        let volume = Aabb::new([0.4, 4.0 / 6.0, 4.0 / 7.0], [0.6, 5.0 / 6.0, 5.0 / 7.0]).unwrap();
        let scaled = Aabb::new(
            volume.min.map(|value| value * 1e-6),
            volume.max.map(|value| value * 1e-6),
        )
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::for_quality(Quality::Reference)
        };
        let unit_scene = scene(1.0);
        let micro_scene = scene(1e-6);
        let unit = integrate_mixture(&unit_scene, &volume, &[0], &options, None);
        let micro = integrate_mixture(&micro_scene, &scaled, &[0], &options, None);
        assert_abs_diff_eq!(unit.error, micro.error, epsilon = 1e-12);
        assert_abs_diff_eq!(
            unit.fractions.as_ref().unwrap()[1],
            micro.fractions.as_ref().unwrap()[1],
            epsilon = 1e-12
        );
        assert!(matches!(
            unit.interface,
            InterfaceClass::Ambiguous(FallbackReason::UnresolvedGeometry)
        ));
        assert!(matches!(
            micro.interface,
            InterfaceClass::Ambiguous(FallbackReason::UnresolvedGeometry)
        ));
    }

    #[test]
    fn fast_quality_does_not_erase_a_thin_tapered_layer() {
        let polygon =
            Polygon2::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], vec![]).unwrap();
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::TaperedExtrudedPolygon(
                    crate::TaperedExtrudedPolygon::new(polygon, 0.0, 0.22, 10.0, 0.0).unwrap(),
                ),
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let result = rasterize(
            &scene,
            &grid,
            &IntegrationOptions::for_quality(Quality::Fast),
        )
        .unwrap();

        assert!(result.cell_epsilon.values[0] > 1.0);
    }

    #[test]
    fn extruded_polygon_overlap_is_exact() {
        let polygon =
            Polygon2::new(vec![[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]], vec![]).unwrap();
        let scene = Scene::new(
            vec![Material::default(), Material::new(9.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::ExtrudedPolygon(
                    crate::ExtrudedPolygon::new(polygon, 0.0, 1.0).unwrap(),
                ),
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1, 1, 1],
        }
        .build()
        .unwrap();
        let result = rasterize(&scene, &grid, &IntegrationOptions::default()).unwrap();
        assert_abs_diff_eq!(result.cell_epsilon.values[0], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn two_dimensional_tm_only_emits_solver_components() {
        let scene = box_scene(Aabb::new([0.2, 0.2, 0.0], [0.8, 0.8, 1.0]).unwrap(), 12.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [12, 11, 1],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            output_components: OutputComponents::TwoDimensionalTm,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert!(result.epsilon[0].values.is_empty());
        assert!(result.epsilon[1].values.is_empty());
        assert!(!result.epsilon[2].values.is_empty());
        assert!(!result.mu[0].values.is_empty());
        assert!(!result.mu[1].values.is_empty());
        assert!(result.mu[2].values.is_empty());
        assert!(!result.cell_epsilon.values.is_empty());
    }

    #[test]
    fn two_dimensional_te_only_emits_solver_components() {
        let scene = box_scene(Aabb::new([0.2, 0.2, 0.0], [0.8, 0.8, 1.0]).unwrap(), 12.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [12, 11, 1],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            output_components: OutputComponents::TwoDimensionalTe,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert!(!result.epsilon[0].values.is_empty());
        assert!(!result.epsilon[1].values.is_empty());
        assert!(result.epsilon[2].values.is_empty());
        assert!(result.mu[0].values.is_empty());
        assert!(result.mu[1].values.is_empty());
        assert!(!result.mu[2].values.is_empty());
        assert!(result.node_epsilon.values.is_empty());
        assert!(!result.cell_epsilon.values.is_empty());
    }

    #[test]
    fn diagonal_mode_is_harmonic_normal_to_axis_aligned_interface() {
        let scene = box_scene(Aabb::new([0.0, 0.0, 0.0], [0.5, 1.0, 1.0]).unwrap(), 4.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1, 1, 1],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourDiagonal,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert_abs_diff_eq!(result.epsilon[0].values[0], 1.6, epsilon = 1e-6);
        let effective = farjadpour_average(&scene, &[0.5, 0.5], [1.0, 0.0, 0.0], |m| m.epsilon_r);
        assert_abs_diff_eq!(effective.0[0], 1.6, epsilon = 1e-12);
        assert_abs_diff_eq!(effective.0[1], 2.5, epsilon = 1e-12);
    }

    #[test]
    fn tau_transform_round_trips_symmetric_tensor() {
        let tensor = [[4.0, 0.2, -0.1], [0.2, 3.0, 0.3], [-0.1, 0.3, 2.0]];
        let recovered = inverse_tau_transform(tau_transform(tensor));
        for row in 0..3 {
            for col in 0..3 {
                assert_abs_diff_eq!(recovered[row][col], tensor[row][col], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn farjadpour_is_rotation_covariant_for_isotropic_media() {
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![],
            0,
        )
        .unwrap();
        let fractions = [0.5, 0.5];
        let effective = farjadpour_average(&scene, &fractions, [1.0, 1.0, 0.0], |m| m.epsilon_r);
        assert_abs_diff_eq!(effective.0[0], 2.05, epsilon = 1e-12);
        assert_abs_diff_eq!(effective.0[1], 2.05, epsilon = 1e-12);
        assert_abs_diff_eq!(effective.0[3], -0.45, epsilon = 1e-12);
        assert_abs_diff_eq!(effective.0[2], 2.5, epsilon = 1e-12);
    }

    #[test]
    fn full_tensors_are_integrated_independently_at_each_constitutive_support() {
        let polygon = Polygon2::new(vec![[-1.0, -1.0], [2.0, -1.0], [-1.0, 2.0]], vec![]).unwrap();
        let geometry =
            Geometry::ExtrudedPolygon(crate::ExtrudedPolygon::new(polygon, -1.0, 2.0).unwrap());
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: geometry.clone(),
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();

        for (component, tensor) in [Component::Ex, Component::Ey, Component::Ez, Component::Node]
            .into_iter()
            .zip(result.epsilon.iter().chain([&result.node_epsilon]))
        {
            assert_eq!(tensor.shape[0], 6);
            let logical_shape = component.support().logical_shape(&grid);
            let sample_count: usize = logical_shape.iter().product();
            for flat in 0..sample_count {
                let x = flat % logical_shape[0];
                let rest = flat / logical_shape[0];
                let y = rest % logical_shape[1];
                let z = rest / logical_shape[1];
                let support = component.support().volume(&grid, [x, y, z]);
                let fraction = geometry.exact_overlap_volume(&support).unwrap() / support.volume();
                let arithmetic = 1.0 + 3.0 * fraction;
                let harmonic = 1.0 / (1.0 - fraction + fraction / 4.0);
                let expected = [
                    0.5 * (arithmetic + harmonic),
                    0.5 * (arithmetic + harmonic),
                    arithmetic,
                    0.5 * (harmonic - arithmetic),
                    0.0,
                    0.0,
                ];
                for (axis, value) in expected.into_iter().enumerate() {
                    assert_abs_diff_eq!(
                        tensor.values[axis * sample_count + flat],
                        value as f32,
                        epsilon = 2e-6
                    );
                }
            }
        }
        assert!(result.mu.iter().all(|tensor| tensor.shape[0] == 1));
        assert!(
            result
                .conductivity
                .iter()
                .all(|tensor| tensor.shape[0] == 1)
        );
    }

    #[test]
    fn interface_geometry_preserves_equal_epsilon_material_identity() {
        let scene = box_scene(Aabb::new([0.0, -1.0, -1.0], [0.5, 3.0, 5.0]).unwrap(), 1.0);
        let grid = Grid::new(vec![0.0, 1.0, 3.0], vec![0.0, 2.0], vec![0.0, 4.0]).unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourDiagonal,
            ..IntegrationOptions::default()
        };
        let samples = interface_samples(&scene, &grid, &options);
        assert!(!samples[0].indices.is_empty());
        assert!(samples[1].indices.is_empty());
        assert!(samples[2].indices.is_empty());
        for fractions in samples[0].fractions.chunks_exact(2) {
            assert_abs_diff_eq!(fractions[0], 0.5, epsilon = 1e-12);
            assert_abs_diff_eq!(fractions[1], 0.5, epsilon = 1e-12);
        }
        assert!(samples[0].normal_squared.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn nonuniform_dual_volume_controls_support_tensor_fraction() {
        let scene = box_scene(Aabb::new([0.0, -1.0, -1.0], [1.0, 3.0, 5.0]).unwrap(), 4.0);
        let grid = Grid::new(vec![0.0, 1.0, 3.0], vec![0.0, 2.0], vec![0.0, 4.0]).unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();

        // Ey at x-edge 1 owns the nonuniform dual interval [0.5, 2.0]. The
        // high-index material occupies one third of it, so epsilon normal to
        // the interface is harmonic and the tangential value is arithmetic.
        let tensor = &result.epsilon[1];
        let samples = tensor.shape[1..].iter().product::<usize>();
        let flat = 1;
        assert_abs_diff_eq!(tensor.values[flat], 4.0 / 3.0, epsilon = 2e-6);
        assert_abs_diff_eq!(tensor.values[samples + flat], 2.0, epsilon = 2e-6);
    }

    #[test]
    fn planar_and_parallel_interfaces_are_smoothed() {
        for bounds in [
            Aabb::new([0.0, -1.0, -1.0], [0.5, 2.0, 2.0]).unwrap(),
            Aabb::new([0.25, -1.0, -1.0], [0.75, 2.0, 2.0]).unwrap(),
        ] {
            let scene = box_scene(bounds, 4.0);
            let grid = UniformGrid {
                min: [0.0; 3],
                max: [1.0; 3],
                shape: [1; 3],
            }
            .build()
            .unwrap();
            let options = IntegrationOptions {
                smoothing: SmoothingMode::FarjadpourFull,
                ..IntegrationOptions::default()
            };
            let result = rasterize(&scene, &grid, &options).unwrap();
            assert!(result.diagnostics.smoothed_samples > 0);
            assert_eq!(result.diagnostics.ambiguous_interface_samples, 0);
        }
    }

    #[test]
    fn multi_orientation_support_falls_back_to_volume_average() {
        let scene = box_scene(Aabb::new([0.0, 0.0, -1.0], [0.5, 0.5, 2.0]).unwrap(), 4.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert_abs_diff_eq!(result.cell_epsilon.values[0], 1.75, epsilon = 1e-6);
        assert!(result.diagnostics.fallback_multiple_orientations > 0);
    }

    #[test]
    fn overlapping_parallel_objects_respect_painter_order_and_smooth() {
        let scene = Scene::new(
            vec![
                Material::default(),
                Material::new(2.0, 1.0, 0.0).unwrap(),
                Material::new(4.0, 1.0, 0.0).unwrap(),
            ],
            vec![
                Object {
                    id: 1,
                    material_id: 1,
                    priority: 0,
                    geometry: Geometry::Box {
                        bounds: Aabb::new([0.0, -1.0, -1.0], [0.6, 2.0, 2.0]).unwrap(),
                    },
                },
                Object {
                    id: 2,
                    material_id: 2,
                    priority: 1,
                    geometry: Geometry::Box {
                        bounds: Aabb::new([0.4, -1.0, -1.0], [1.0, 2.0, 2.0]).unwrap(),
                    },
                },
            ],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert_eq!(result.diagnostics.fallback_multiple_objects, 0);
        assert!(result.diagnostics.smoothed_samples > 0);
        // The higher-priority object wins the overlap [0.4, 0.6].
        assert_abs_diff_eq!(
            result.cell_epsilon.values[0],
            1.0 / (0.4 / 2.0 + 0.6 / 4.0),
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            result.cell_epsilon.values[1],
            0.4 * 2.0 + 0.6 * 4.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn undefined_interface_normal_falls_back_without_nan() {
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::Sphere {
                    center: [0.5; 3],
                    radius: 0.4,
                },
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert!(
            result
                .cell_epsilon
                .values
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(result.diagnostics.ambiguous_interface_samples > 0);
    }

    fn diagonal_smoothing() -> IntegrationOptions {
        IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourDiagonal,
            ..IntegrationOptions::default()
        }
    }

    #[test]
    fn compact_material_support_counts_every_sample_and_keeps_interface_values() {
        // The x = 0.55 face crosses one column of cells, so the grid holds
        // background, fully filled and interface cells.
        let scene = box_scene(Aabb::new([0.0; 3], [0.55, 1.0, 1.0]).unwrap(), 4.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [10, 3, 2],
        }
        .build()
        .unwrap();
        let result = rasterize(&scene, &grid, &diagonal_smoothing()).unwrap();
        let summary = &result.diagnostics;
        let expected: usize = [Component::Ex, Component::Ey, Component::Ez, Component::Cell]
            .iter()
            .map(|component| {
                component
                    .support()
                    .logical_shape(&grid)
                    .iter()
                    .product::<usize>()
            })
            .sum();
        assert_eq!(summary.samples, expected);
        assert_eq!(
            summary.samples,
            summary.background_samples
                + summary.uniform_samples
                + summary.exact_samples
                + summary.adaptive_samples
        );
        let values = &result.epsilon[0].values;
        assert!(values.contains(&1.0) && values.contains(&4.0));
        assert!(values.iter().any(|value| *value > 1.0 && *value < 4.0));
    }

    #[test]
    fn tensor_components_follow_only_materials_present_in_the_grid() {
        let anisotropic = Material {
            epsilon_r: SymmetricTensor([2.0, 3.0, 4.0, 0.0, 0.0, 0.0]),
            mu_r: SymmetricTensor::isotropic(1.0),
            conductivity: SymmetricTensor::isotropic(0.0),
        };
        let scene = |bounds| {
            Scene::new(
                vec![Material::default(), anisotropic],
                vec![Object {
                    id: 1,
                    material_id: 1,
                    priority: 0,
                    geometry: Geometry::Box { bounds },
                }],
                0,
            )
            .unwrap()
        };
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [4, 4, 4],
        }
        .build()
        .unwrap();
        let inside = scene(Aabb::new([0.25; 3], [0.75; 3]).unwrap());
        let outside = scene(Aabb::new([2.0; 3], [3.0; 3]).unwrap());
        let options = diagonal_smoothing();
        assert_eq!(
            rasterize(&inside, &grid, &options).unwrap().epsilon[0].shape[0],
            3
        );
        assert_eq!(
            rasterize(&outside, &grid, &options).unwrap().epsilon[0].shape[0],
            1
        );
    }
}

#[cfg(test)]
mod diagnostics;
