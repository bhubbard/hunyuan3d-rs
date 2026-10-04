use crate::math::BoundingBox3D;
use crate::mesh::Mesh;
use glam::Vec3;
use rayon::prelude::*;

/// 3D Scalar Field (Voxel Grid / SDF / Occupancy Logits).
#[derive(Debug, Clone)]
pub struct ScalarGrid3D {
    pub dim_x: usize,
    pub dim_y: usize,
    pub dim_z: usize,
    pub data: Vec<f32>,
    pub bounds: BoundingBox3D,
}

impl ScalarGrid3D {
    pub fn new(dim_x: usize, dim_y: usize, dim_z: usize, bounds: BoundingBox3D) -> Self {
        let size = dim_x * dim_y * dim_z;
        Self {
            dim_x,
            dim_y,
            dim_z,
            data: vec![0.0; size],
            bounds,
        }
    }

    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dim_y + y) * self.dim_x + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[self.index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: f32) {
        let idx = self.index(x, y, z);
        self.data[idx] = val;
    }

    /// Evaluates an analytical Signed Distance Function (SDF) over the grid.
    pub fn fill_with_sdf<F>(&mut self, sdf: F)
    where
        F: Fn(Vec3) -> f32 + Sync + Send,
    {
        let dx = (self.bounds.max.x - self.bounds.min.x) / ((self.dim_x - 1) as f32);
        let dy = (self.bounds.max.y - self.bounds.min.y) / ((self.dim_y - 1) as f32);
        let dz = (self.bounds.max.z - self.bounds.min.z) / ((self.dim_z - 1) as f32);

        let min = self.bounds.min;
        let dim_x = self.dim_x;
        let dim_y = self.dim_y;

        self.data
            .par_chunks_mut(dim_x)
            .enumerate()
            .for_each(|(slice_idx, row)| {
                let y = slice_idx % dim_y;
                let z = slice_idx / dim_y;
                let py = min.y + (y as f32) * dy;
                let pz = min.z + (z as f32) * dz;

                for (x, val) in row.iter_mut().enumerate() {
                    let px = min.x + (x as f32) * dx;
                    *val = sdf(Vec3::new(px, py, pz));
                }
            });
    }
}

/// Marching Cubes Iso-surface Extractor.
/// Matches upstream `MCSurfaceExtractor` in Hunyuan3D.
pub struct MarchingCubes;

impl MarchingCubes {
    /// Extracts a triangular mesh from a 3D scalar grid at the specified iso-level.
    pub fn extract(grid: &ScalarGrid3D, isolevel: f32) -> Mesh {
        let nx = grid.dim_x;
        let ny = grid.dim_y;
        let nz = grid.dim_z;

        if nx < 2 || ny < 2 || nz < 2 {
            return Mesh::new();
        }

        let dx = (grid.bounds.max.x - grid.bounds.min.x) / ((nx - 1) as f32);
        let dy = (grid.bounds.max.y - grid.bounds.min.y) / ((ny - 1) as f32);
        let dz = (grid.bounds.max.z - grid.bounds.min.z) / ((nz - 1) as f32);
        let min = grid.bounds.min;

        // Process grid cells in parallel across slices of Z
        let chunks: Vec<(Vec<Vec3>, Vec<u32>)> = (0..nz - 1)
            .into_par_iter()
            .map(|z| {
                let mut local_verts = Vec::new();
                let mut local_indices = Vec::new();

                for y in 0..ny - 1 {
                    for x in 0..nx - 1 {
                        // 8 corner values and positions
                        let p0 = min + Vec3::new(x as f32 * dx, y as f32 * dy, z as f32 * dz);
                        let p1 = min + Vec3::new((x + 1) as f32 * dx, y as f32 * dy, z as f32 * dz);
                        let p2 = min + Vec3::new((x + 1) as f32 * dx, (y + 1) as f32 * dy, z as f32 * dz);
                        let p3 = min + Vec3::new(x as f32 * dx, (y + 1) as f32 * dy, z as f32 * dz);
                        let p4 = min + Vec3::new(x as f32 * dx, y as f32 * dy, (z + 1) as f32 * dz);
                        let p5 = min + Vec3::new((x + 1) as f32 * dx, y as f32 * dy, (z + 1) as f32 * dz);
                        let p6 = min + Vec3::new((x + 1) as f32 * dx, (y + 1) as f32 * dy, (z + 1) as f32 * dz);
                        let p7 = min + Vec3::new(x as f32 * dx, (y + 1) as f32 * dy, (z + 1) as f32 * dz);

                        let v0 = grid.get(x, y, z);
                        let v1 = grid.get(x + 1, y, z);
                        let v2 = grid.get(x + 1, y + 1, z);
                        let v3 = grid.get(x, y + 1, z);
                        let v4 = grid.get(x, y, z + 1);
                        let v5 = grid.get(x + 1, y, z + 1);
                        let v6 = grid.get(x + 1, y + 1, z + 1);
                        let v7 = grid.get(x, y + 1, z + 1);

                        // Cube index
                        let mut cube_idx = 0usize;
                        if v0 < isolevel { cube_idx |= 1; }
                        if v1 < isolevel { cube_idx |= 2; }
                        if v2 < isolevel { cube_idx |= 4; }
                        if v3 < isolevel { cube_idx |= 8; }
                        if v4 < isolevel { cube_idx |= 16; }
                        if v5 < isolevel { cube_idx |= 32; }
                        if v6 < isolevel { cube_idx |= 64; }
                        if v7 < isolevel { cube_idx |= 128; }

                        let edges = EDGE_TABLE[cube_idx];
                        if edges == 0 {
                            continue;
                        }

                        // Interpolate edge vertices
                        let mut vert_list = [Vec3::ZERO; 12];
                        if edges & 1 != 0 { vert_list[0] = interp(isolevel, p0, p1, v0, v1); }
                        if edges & 2 != 0 { vert_list[1] = interp(isolevel, p1, p2, v1, v2); }
                        if edges & 4 != 0 { vert_list[2] = interp(isolevel, p2, p3, v2, v3); }
                        if edges & 8 != 0 { vert_list[3] = interp(isolevel, p3, p0, v3, v0); }
                        if edges & 16 != 0 { vert_list[4] = interp(isolevel, p4, p5, v4, v5); }
                        if edges & 32 != 0 { vert_list[5] = interp(isolevel, p5, p6, v5, v6); }
                        if edges & 64 != 0 { vert_list[6] = interp(isolevel, p6, p7, v6, v7); }
                        if edges & 128 != 0 { vert_list[7] = interp(isolevel, p7, p4, v7, v4); }
                        if edges & 256 != 0 { vert_list[8] = interp(isolevel, p0, p4, v0, v4); }
                        if edges & 512 != 0 { vert_list[9] = interp(isolevel, p1, p5, v1, v5); }
                        if edges & 1024 != 0 { vert_list[10] = interp(isolevel, p2, p6, v2, v6); }
                        if edges & 2048 != 0 { vert_list[11] = interp(isolevel, p3, p7, v3, v7); }

                        // Generate triangles
                        let tri_row = &TRI_TABLE[cube_idx];
                        let mut i = 0;
                        while tri_row[i] != -1 {
                            let e0 = tri_row[i] as usize;
                            let e1 = tri_row[i + 1] as usize;
                            let e2 = tri_row[i + 2] as usize;

                            let base_idx = local_verts.len() as u32;
                            local_verts.push(vert_list[e0]);
                            local_verts.push(vert_list[e1]);
                            local_verts.push(vert_list[e2]);

                            local_indices.push(base_idx);
                            local_indices.push(base_idx + 1);
                            local_indices.push(base_idx + 2);

                            i += 3;
                        }
                    }
                }

                (local_verts, local_indices)
            })
            .collect();

        // Merge thread outputs and weld coincident vertices
        let mut total_verts = Vec::new();
        let mut total_indices = Vec::new();
        let mut vert_map: std::collections::HashMap<(i64, i64, i64), u32> = std::collections::HashMap::new();

        let eps = (dx.min(dy).min(dz) * 0.05).max(1e-5);
        let inv_eps = 1.0 / eps;

        for (v_list, idx_list) in chunks {
            for &idx in &idx_list {
                let p = v_list[idx as usize];
                let key = (
                    (p.x * inv_eps).round() as i64,
                    (p.y * inv_eps).round() as i64,
                    (p.z * inv_eps).round() as i64,
                );

                let welded_idx = if let Some(&existing) = vert_map.get(&key) {
                    existing
                } else {
                    let new_id = total_verts.len() as u32;
                    total_verts.push(p);
                    vert_map.insert(key, new_id);
                    new_id
                };

                total_indices.push(welded_idx);
            }
        }

        let mut mesh = Mesh {
            vertices: total_verts,
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
            tangents: Vec::new(),
            indices: total_indices,
        };

        mesh.compute_normals();
        mesh.generate_uvs();
        mesh
    }
}

#[inline]
fn interp(isolevel: f32, p1: Vec3, p2: Vec3, val1: f32, val2: f32) -> Vec3 {
    if (isolevel - val1).abs() < 1e-6 {
        return p1;
    }
    if (isolevel - val2).abs() < 1e-6 {
        return p2;
    }
    if (val1 - val2).abs() < 1e-6 {
        return p1;
    }

    let mu = ((isolevel - val1) / (val2 - val1)).clamp(0.0, 1.0);
    p1 + mu * (p2 - p1)
}

// Look-up tables for 256 Marching Cubes configurations
include!("marching_cubes_tables.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marching_cubes_sphere() {
        let bounds = BoundingBox3D::new(Vec3::splat(-1.5), Vec3::splat(1.5));
        let mut grid = ScalarGrid3D::new(24, 24, 24, bounds);

        // Sphere SDF: f(p) = ||p|| - r
        let radius = 1.0f32;
        grid.fill_with_sdf(|p| p.length() - radius);

        let mesh = MarchingCubes::extract(&grid, 0.0);
        assert!(!mesh.is_empty());
        assert!(mesh.len_triangles() > 100);

        let bbox = mesh.bounding_box();
        assert!((bbox.max_dimension() - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_marching_cubes_torus() {
        let bounds = BoundingBox3D::new(Vec3::splat(-2.0), Vec3::splat(2.0));
        let mut grid = ScalarGrid3D::new(28, 28, 28, bounds);

        // Torus SDF: (sqrt(x^2 + z^2) - R)^2 + y^2 - r^2
        let big_r = 1.2f32;
        let small_r = 0.4f32;
        grid.fill_with_sdf(|p| {
            let q = Vec3::new(p.x, 0.0, p.z).length() - big_r;
            (q * q + p.y * p.y).sqrt() - small_r
        });

        let mesh = MarchingCubes::extract(&grid, 0.0);
        assert!(!mesh.is_empty());
        assert!(mesh.len_triangles() > 200);
    }
}
