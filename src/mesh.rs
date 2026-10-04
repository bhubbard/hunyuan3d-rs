use crate::math::{compute_tangent_basis, BoundingBox3D};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Core 3D Mesh structure representing surfaces in Hunyuan3D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub colors: Vec<[f32; 4]>,
    pub tangents: Vec<Vec3>,
    /// Triangle face indices (each 3 indices form a triangle).
    pub indices: Vec<u32>,
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
            tangents: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn len_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn len_triangles(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    pub fn bounding_box(&self) -> BoundingBox3D {
        BoundingBox3D::from_points(&self.vertices)
    }

    /// Translates the mesh so its bounding box center is at (0, 0, 0)
    /// and scales it to fit within `target_radius` (default 1.0).
    pub fn center_and_normalize(&mut self, target_radius: f32) {
        if self.vertices.is_empty() {
            return;
        }

        let bbox = self.bounding_box();
        let center = bbox.center();
        let max_dim = bbox.max_dimension();
        let scale = if max_dim > 1e-6 {
            (2.0 * target_radius) / max_dim
        } else {
            1.0
        };

        for v in &mut self.vertices {
            *v = (*v - center) * scale;
        }
    }

    /// Recomputes smooth vertex normals by accumulating weighted face normals.
    pub fn compute_normals(&mut self) {
        let num_verts = self.vertices.len();
        let mut accum_normals = vec![Vec3::ZERO; num_verts];

        for chunk in self.indices.chunks_exact(3) {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            if i0 >= num_verts || i1 >= num_verts || i2 >= num_verts {
                continue;
            }

            let v0 = self.vertices[i0];
            let v1 = self.vertices[i1];
            let v2 = self.vertices[i2];

            let face_normal = (v1 - v0).cross(v2 - v0);
            accum_normals[i0] += face_normal;
            accum_normals[i1] += face_normal;
            accum_normals[i2] += face_normal;
        }

        self.normals = accum_normals
            .into_iter()
            .map(|n| n.normalize_or_zero())
            .collect();
    }

    /// Computes tangent vectors for normal mapping.
    pub fn compute_tangents(&mut self) {
        if self.normals.is_empty() {
            self.compute_normals();
        }
        if self.uvs.is_empty() {
            self.generate_uvs();
        }

        let num_verts = self.vertices.len();
        let mut accum_tangents = vec![Vec3::ZERO; num_verts];

        for chunk in self.indices.chunks_exact(3) {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            if i0 >= num_verts || i1 >= num_verts || i2 >= num_verts {
                continue;
            }

            let (tan, _) = compute_tangent_basis(
                self.vertices[i0],
                self.vertices[i1],
                self.vertices[i2],
                self.uvs[i0],
                self.uvs[i1],
                self.uvs[i2],
                self.normals[i0],
            );

            accum_tangents[i0] += tan;
            accum_tangents[i1] += tan;
            accum_tangents[i2] += tan;
        }

        self.tangents = accum_tangents
            .into_iter()
            .map(|t| t.normalize_or_zero())
            .collect();
    }

    /// Automatically generates UV coordinates using cylindrical/box projection
    /// if UVs are not present.
    pub fn generate_uvs(&mut self) {
        let bbox = self.bounding_box();
        let _size = bbox.size();
        let center = bbox.center();

        let mut uvs = Vec::with_capacity(self.vertices.len());
        for v in &self.vertices {
            let p = *v - center;
            // Spherical / equirectangular unwrapping
            let len = p.length();
            if len < 1e-6 {
                uvs.push(Vec2::new(0.5, 0.5));
                continue;
            }
            let u = 0.5 + p.x.atan2(p.z) / (2.0 * std::f32::consts::PI);
            let v_coord = 0.5 - (p.y / len).clamp(-1.0, 1.0).asin() / std::f32::consts::PI;
            uvs.push(Vec2::new(u.clamp(0.0, 1.0), v_coord.clamp(0.0, 1.0)));
        }
        self.uvs = uvs;
    }

    /// Applies Laplacian smoothing to smooth jagged Marching Cubes vertices.
    pub fn laplacian_smooth(&mut self, iterations: usize, lambda: f32) {
        if self.vertices.is_empty() || self.indices.is_empty() {
            return;
        }

        let num_verts = self.vertices.len();
        // Build adjacency graph
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_verts];
        for chunk in self.indices.chunks_exact(3) {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 < num_verts && i1 < num_verts && i2 < num_verts {
                adj[i0].insert(i1);
                adj[i0].insert(i2);
                adj[i1].insert(i0);
                adj[i1].insert(i2);
                adj[i2].insert(i0);
                adj[i2].insert(i1);
            }
        }

        let mut current_verts = self.vertices.clone();
        for _ in 0..iterations {
            let mut next_verts = current_verts.clone();
            for i in 0..num_verts {
                let neighbors = &adj[i];
                if neighbors.is_empty() {
                    continue;
                }
                let avg = neighbors
                    .iter()
                    .map(|&n| current_verts[n])
                    .sum::<Vec3>()
                    / (neighbors.len() as f32);
                next_verts[i] = current_verts[i] + lambda * (avg - current_verts[i]);
            }
            current_verts = next_verts;
        }

        self.vertices = current_verts;
        self.compute_normals();
    }

    /// Finds and preserves only the largest connected component of the mesh,
    /// discarding disconnected floating artifacts.
    pub fn isolate_largest_component(&mut self) {
        let num_verts = self.vertices.len();
        if num_verts == 0 || self.indices.is_empty() {
            return;
        }

        // Build vertex adjacency through faces
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_verts];
        for chunk in self.indices.chunks_exact(3) {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if i0 < num_verts && i1 < num_verts && i2 < num_verts {
                adj[i0].push(i1);
                adj[i0].push(i2);
                adj[i1].push(i0);
                adj[i1].push(i2);
                adj[i2].push(i0);
                adj[i2].push(i1);
            }
        }

        let mut visited = vec![false; num_verts];
        let mut components: Vec<Vec<usize>> = Vec::new();

        for i in 0..num_verts {
            if visited[i] {
                continue;
            }
            let mut comp = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(i);
            visited[i] = true;

            while let Some(curr) = queue.pop_front() {
                comp.push(curr);
                for &neighbor in &adj[curr] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            components.push(comp);
        }

        if components.is_empty() {
            return;
        }

        // Find component with max vertices
        let largest = components
            .into_iter()
            .max_by_key(|c| c.len())
            .unwrap_or_default();

        let in_largest: HashSet<usize> = largest.iter().copied().collect();

        // Remap indices
        let mut remap = vec![None; num_verts];
        let mut new_verts = Vec::with_capacity(largest.len());
        let mut new_normals = Vec::new();
        let mut new_uvs = Vec::new();
        let mut new_colors = Vec::new();

        for (new_idx, &old_idx) in largest.iter().enumerate() {
            remap[old_idx] = Some(new_idx as u32);
            new_verts.push(self.vertices[old_idx]);
            if !self.normals.is_empty() {
                new_normals.push(self.normals[old_idx]);
            }
            if !self.uvs.is_empty() {
                new_uvs.push(self.uvs[old_idx]);
            }
            if !self.colors.is_empty() {
                new_colors.push(self.colors[old_idx]);
            }
        }

        let mut new_indices = Vec::new();
        for chunk in self.indices.chunks_exact(3) {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            if in_largest.contains(&i0) && in_largest.contains(&i1) && in_largest.contains(&i2) {
                if let (Some(n0), Some(n1), Some(n2)) = (remap[i0], remap[i1], remap[i2]) {
                    new_indices.push(n0);
                    new_indices.push(n1);
                    new_indices.push(n2);
                }
            }
        }

        self.vertices = new_verts;
        self.normals = new_normals;
        self.uvs = new_uvs;
        self.colors = new_colors;
        self.indices = new_indices;
    }

    /// Fast spatial clustering mesh simplification to reduce face count.
    pub fn simplify_mesh(&mut self, target_faces: usize) {
        let current_faces = self.len_triangles();
        if current_faces <= target_faces || current_faces == 0 {
            return;
        }

        let bbox = self.bounding_box();
        let max_dim = bbox.max_dimension();
        if max_dim < 1e-6 {
            return;
        }

        // Estimate grid cell resolution needed for target face count
        // Vertices approximately ~ 0.5 * faces
        let target_verts = (target_faces / 2).max(64);
        let cells_per_axis = ((target_verts as f32).cbrt().ceil() as usize).max(4);
        let cell_size = max_dim / (cells_per_axis as f32);

        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, v) in self.vertices.iter().enumerate() {
            let key = (
                ((v.x - bbox.min.x) / cell_size).floor() as i32,
                ((v.y - bbox.min.y) / cell_size).floor() as i32,
                ((v.z - bbox.min.z) / cell_size).floor() as i32,
            );
            grid.entry(key).or_default().push(i);
        }

        // Cluster centroids
        let mut new_vertices = Vec::new();
        let mut new_normals = Vec::new();
        let mut new_uvs = Vec::new();
        let mut vert_remap = vec![0u32; self.vertices.len()];

        for (_key, cluster) in grid {
            let new_idx = new_vertices.len() as u32;
            let count = cluster.len() as f32;
            let mut avg_pos = Vec3::ZERO;
            let mut avg_norm = Vec3::ZERO;
            let mut avg_uv = Vec2::ZERO;

            for &orig_idx in &cluster {
                vert_remap[orig_idx] = new_idx;
                avg_pos += self.vertices[orig_idx];
                if !self.normals.is_empty() {
                    avg_norm += self.normals[orig_idx];
                }
                if !self.uvs.is_empty() {
                    avg_uv += self.uvs[orig_idx];
                }
            }

            new_vertices.push(avg_pos / count);
            if !self.normals.is_empty() {
                new_normals.push((avg_norm / count).normalize_or_zero());
            }
            if !self.uvs.is_empty() {
                new_uvs.push(avg_uv / count);
            }
        }

        // Filter degenerate triangles
        let mut new_indices = Vec::new();
        for chunk in self.indices.chunks_exact(3) {
            let r0 = vert_remap[chunk[0] as usize];
            let r1 = vert_remap[chunk[1] as usize];
            let r2 = vert_remap[chunk[2] as usize];

            if r0 != r1 && r1 != r2 && r2 != r0 {
                new_indices.push(r0);
                new_indices.push(r1);
                new_indices.push(r2);
            }
        }

        self.vertices = new_vertices;
        self.normals = new_normals;
        self.uvs = new_uvs;
        self.indices = new_indices;
        self.compute_normals();
    }

    /// Constructs a unit cube mesh centered at the origin.
    pub fn create_cube(size: f32) -> Self {
        let h = size * 0.5;
        let vertices = vec![
            // Front face
            Vec3::new(-h, -h,  h), Vec3::new( h, -h,  h), Vec3::new( h,  h,  h), Vec3::new(-h,  h,  h),
            // Back face
            Vec3::new( h, -h, -h), Vec3::new(-h, -h, -h), Vec3::new(-h,  h, -h), Vec3::new( h,  h, -h),
            // Top face
            Vec3::new(-h,  h,  h), Vec3::new( h,  h,  h), Vec3::new( h,  h, -h), Vec3::new(-h,  h, -h),
            // Bottom face
            Vec3::new(-h, -h, -h), Vec3::new( h, -h, -h), Vec3::new( h, -h,  h), Vec3::new(-h, -h,  h),
            // Right face
            Vec3::new( h, -h,  h), Vec3::new( h, -h, -h), Vec3::new( h,  h, -h), Vec3::new( h,  h,  h),
            // Left face
            Vec3::new(-h, -h, -h), Vec3::new(-h, -h,  h), Vec3::new(-h,  h,  h), Vec3::new(-h,  h, -h),
        ];

        let mut indices = Vec::new();
        for i in 0..6 {
            let base = (i * 4) as u32;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        let mut mesh = Self {
            vertices,
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
            tangents: Vec::new(),
            indices,
        };
        mesh.compute_normals();
        mesh.generate_uvs();
        mesh
    }

    /// Constructs an icosphere with subdivision.
    pub fn create_icosphere(radius: f32, subdivisions: usize) -> Self {
        let t = (1.0 + 5.0f32.sqrt()) / 2.0;

        let mut vertices = vec![
            Vec3::new(-1.0,  t, 0.0).normalize(),
            Vec3::new( 1.0,  t, 0.0).normalize(),
            Vec3::new(-1.0, -t, 0.0).normalize(),
            Vec3::new( 1.0, -t, 0.0).normalize(),
            Vec3::new(0.0, -1.0,  t).normalize(),
            Vec3::new(0.0,  1.0,  t).normalize(),
            Vec3::new(0.0, -1.0, -t).normalize(),
            Vec3::new(0.0,  1.0, -t).normalize(),
            Vec3::new( t, 0.0, -1.0).normalize(),
            Vec3::new( t, 0.0,  1.0).normalize(),
            Vec3::new(-t, 0.0, -1.0).normalize(),
            Vec3::new(-t, 0.0,  1.0).normalize(),
        ];

        let mut faces = vec![
            0, 11, 5,  0, 5, 1,   0, 1, 7,   0, 7, 10,  0, 10, 11,
            1, 5, 9,   5, 11, 4,  11, 10, 2, 10, 7, 6,  7, 1, 8,
            3, 9, 4,   3, 4, 2,   3, 2, 6,   3, 6, 8,   3, 8, 9,
            4, 9, 5,   2, 4, 11,  6, 2, 10,  8, 6, 7,   9, 8, 1,
        ];

        // Subdivision
        for _ in 0..subdivisions {
            let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
            let mut new_faces = Vec::new();

            for chunk in faces.chunks_exact(3) {
                let v0 = chunk[0];
                let v1 = chunk[1];
                let v2 = chunk[2];

                let a = get_midpoint(v0, v1, &mut vertices, &mut midpoint_cache);
                let b = get_midpoint(v1, v2, &mut vertices, &mut midpoint_cache);
                let c = get_midpoint(v2, v0, &mut vertices, &mut midpoint_cache);

                new_faces.extend_from_slice(&[v0, a, c]);
                new_faces.extend_from_slice(&[v1, b, a]);
                new_faces.extend_from_slice(&[v2, c, b]);
                new_faces.extend_from_slice(&[a, b, c]);
            }
            faces = new_faces;
        }

        // Scale by radius
        for v in &mut vertices {
            *v *= radius;
        }

        let mut mesh = Self {
            vertices,
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
            tangents: Vec::new(),
            indices: faces,
        };
        mesh.compute_normals();
        mesh.generate_uvs();
        mesh
    }
}

fn get_midpoint(
    p1: u32,
    p2: u32,
    vertices: &mut Vec<Vec3>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if p1 < p2 { (p1, p2) } else { (p2, p1) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let mid = (vertices[p1 as usize] + vertices[p2 as usize]) * 0.5;
    let normalized = mid.normalize();
    let idx = vertices.len() as u32;
    vertices.push(normalized);
    cache.insert(key, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_creation_and_properties() {
        let cube = Mesh::create_cube(2.0);
        assert_eq!(cube.len_triangles(), 12);
        assert_eq!(cube.len_vertices(), 24);

        let bbox = cube.bounding_box();
        assert!((bbox.size().x - 2.0).abs() < 1e-4);
        assert!((bbox.size().y - 2.0).abs() < 1e-4);
        assert!((bbox.size().z - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_icosphere_subdivision() {
        let sphere = Mesh::create_icosphere(1.0, 1);
        assert_eq!(sphere.len_triangles(), 80);
        let bbox = sphere.bounding_box();
        assert!(bbox.center().length() < 1e-4);
    }

    #[test]
    fn test_center_and_normalize() {
        let mut cube = Mesh::create_cube(4.0);
        cube.center_and_normalize(1.0);
        let bbox = cube.bounding_box();
        assert!((bbox.max_dimension() - 2.0).abs() < 1e-4);
        assert!(bbox.center().length() < 1e-4);
    }

    #[test]
    fn test_laplacian_smoothing() {
        let mut sphere = Mesh::create_icosphere(1.0, 1);
        sphere.laplacian_smooth(2, 0.5);
        assert_eq!(sphere.len_triangles(), 80);
        assert!(!sphere.normals.is_empty());
    }

    #[test]
    fn test_mesh_simplification() {
        let mut sphere = Mesh::create_icosphere(1.0, 2); // 320 triangles
        let original_faces = sphere.len_triangles();
        sphere.simplify_mesh(60);
        assert!(sphere.len_triangles() < original_faces);
    }
}
