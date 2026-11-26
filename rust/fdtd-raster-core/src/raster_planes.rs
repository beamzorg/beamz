use std::collections::HashMap;

use super::{DiagnosticSummary, IntegrationOptions, TensorArray};
use crate::{Geometry, Grid, Scene, grid::SupportSpec};

/// The bounding slab and active object set certify z translation for boxes
/// and extruded polygons. A sloped/curved object or a crossed z face forces
/// that layer to be integrated independently. Empty slabs are always equal.
pub(super) fn schedule(
    scene: &Scene,
    grid: &Grid,
    support: SupportSpec,
    options: &IntegrationOptions,
) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    if options.periodic_axes[2] {
        return None;
    }
    let nz = support.logical_shape(grid)[2];
    let bounds = scene.objects.iter().map(|object| {
        let geometry = &object.geometry;
        let fixed = matches!(
            geometry,
            Geometry::Box { .. } | Geometry::ExtrudedPolygon(_)
        );
        (geometry.bounds(), usize::from(!fixed))
    });
    let (mut starts, mut ends): (Vec<_>, Vec<_>) = bounds
        .map(|(b, changing)| ((b.min[2], changing), (b.max[2], changing)))
        .unzip();
    starts.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    ends.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let (mut entered, mut left, mut changing) = (0, 0, 0);
    let mut known = HashMap::new();
    let mut layers = Vec::new();
    let mut ids = Vec::with_capacity(nz);
    let mut counts = Vec::new();
    for z in 0..nz {
        let slab = support.volume(grid, [0, 0, z]);
        let (lo, hi) = (slab.min[2], slab.max[2]);
        while entered < starts.len() && starts[entered].0 <= hi {
            changing += starts[entered].1;
            entered += 1;
        }
        while left < ends.len() && ends[left].0 < lo {
            changing -= ends[left].1;
            left += 1;
        }
        let empty = entered == left;
        let interior = !empty && changing == 0 && starts[entered - 1].0 < lo && ends[left].0 > hi;
        let next = layers.len();
        let id = if empty || interior {
            // The sweep's event count identifies the active-object set until
            // its next change. Empty strata share a single background plane.
            let key = if empty {
                (0, 0)
            } else {
                (entered + left, (hi - lo).to_bits())
            };
            *known.entry(key).or_insert(next)
        } else {
            next
        };
        if id == next {
            layers.push(z);
            counts.push(0);
        }
        ids.push(id);
        counts[id] += 1;
    }
    (layers.len() * 2 <= nz).then_some((layers, ids, counts))
}

#[derive(Default)]
pub(super) struct Bank {
    physical: Vec<Vec<f32>>,
    hashes: HashMap<blake3::Hash, Vec<usize>>,
    layers: Vec<Vec<usize>>,
}

impl Bank {
    pub(super) fn add(&mut self, tensor: TensorArray) {
        let [components, _, ny, nx] = tensor.shape;
        let mut layer = Vec::with_capacity(components);
        for values in tensor.values.chunks(nx * ny) {
            let first = values[0].to_bits();
            let values = if values.iter().all(|v| v.to_bits() == first) {
                &values[..1]
            } else {
                values
            };
            let hash = blake3::hash(bytemuck::cast_slice(values));
            let candidates = self.hashes.entry(hash).or_default();
            let id = candidates
                .iter()
                .copied()
                .find(|&i| {
                    bytemuck::cast_slice::<_, u8>(&self.physical[i])
                        == bytemuck::cast_slice::<_, u8>(values)
                })
                .unwrap_or_else(|| {
                    let next = self.physical.len();
                    self.physical.push(values.to_vec());
                    candidates.push(next);
                    next
                });
            layer.push(id);
        }
        self.layers.push(layer);
    }

    pub(super) fn finish(self, spatial: [usize; 3], indices: &[usize]) -> TensorArray {
        let [z, y, x] = spatial;
        let c = self.layers.iter().map(Vec::len).max().unwrap();
        let shape = [c, z, y, x];
        if self.physical.len() == 1 && self.physical[0].len() == 1 {
            return TensorArray {
                shape,
                values: self.physical.into_iter().next().unwrap(),
                plane_ids: vec![],
            };
        }
        let mut plane_ids = Vec::with_capacity(c * z);
        for component in 0..c {
            for &layer in indices {
                let row = &self.layers[layer];
                // The compact-plane path is restricted to diagonal constitutive tensors.
                plane_ids.push(row[component.min(row.len() - 1)]);
            }
        }
        let mut values = Vec::with_capacity(self.physical.len() * x * y);
        for plane in self.physical {
            if plane.len() == 1 {
                values.resize(values.len() + x * y, plane[0]);
            } else {
                values.extend(plane);
            }
        }
        TensorArray {
            shape,
            values,
            plane_ids,
        }
    }
}

pub(super) fn accumulate(out: &mut DiagnosticSummary, layer: &DiagnosticSummary, n: usize) {
    macro_rules! sum {
        ($($field:ident),+) => { $(out.$field += layer.$field * n;)+ };
    }
    sum!(
        samples,
        background_samples,
        uniform_samples,
        exact_samples,
        adaptive_samples,
        unresolved_samples,
        smoothed_samples,
        ambiguous_interface_samples,
        fallback_multiple_orientations,
        fallback_multiple_objects,
        fallback_unresolved_geometry,
        fallback_missing_surface_evidence
    );
    out.candidate_object_tests += layer.candidate_object_tests * n as u64;
    out.estimated_fraction_error_sum += layer.estimated_fraction_error_sum * n as f64;
    out.maximum_estimated_fraction_error = out
        .maximum_estimated_fraction_error
        .max(layer.maximum_estimated_fraction_error);
}
