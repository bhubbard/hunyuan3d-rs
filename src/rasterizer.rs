use crate::camera::Camera;
use crate::math::barycentric_2d;
use crate::mesh::Mesh;
use glam::{Vec2, Vec3};
use image::{Rgba, RgbaImage};

/// Output buffers produced by rendering a 3D mesh from a camera viewpoint.
#[derive(Debug, Clone)]
pub struct RenderBuffer {
    pub width: u32,
    pub height: u32,
    pub depth: Vec<f32>,
    pub normals: Vec<Vec3>,
    pub positions: Vec<Vec3>,
    pub triangle_ids: Vec<i32>,
    pub colors: Vec<[u8; 4]>,
}

impl RenderBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            depth: vec![f32::INFINITY; size],
            normals: vec![Vec3::ZERO; size],
            positions: vec![Vec3::ZERO; size],
            triangle_ids: vec![-1; size],
            colors: vec![[0, 0, 0, 0]; size],
        }
    }

    #[inline]
    pub fn pixel_index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    /// Converts the normal buffer to an RGBA normal map image.
    /// Normals are mapped from [-1, 1] to [0, 255].
    pub fn to_normal_image(&self) -> RgbaImage {
        let mut img = RgbaImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.pixel_index(x, y);
                if self.triangle_ids[idx] >= 0 {
                    let n = self.normals[idx];
                    let r = ((n.x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                    let g = ((n.y * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                    let b = ((n.z * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                    img.put_pixel(x, y, Rgba([r, g, b, 255]));
                } else {
                    img.put_pixel(x, y, Rgba([128, 128, 255, 0])); // Neutral normal
                }
            }
        }
        img
    }

    /// Converts the depth buffer to a normalized 8-bit grayscale depth image.
    pub fn to_depth_image(&self) -> RgbaImage {
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for &z in &self.depth {
            if z.is_finite() {
                min_z = min_z.min(z);
                max_z = max_z.max(z);
            }
        }

        let range = if (max_z - min_z) > 1e-6 {
            max_z - min_z
        } else {
            1.0
        };

        let mut img = RgbaImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.pixel_index(x, y);
                let z = self.depth[idx];
                if z.is_finite() {
                    let norm_z = 1.0 - ((z - min_z) / range).clamp(0.0, 1.0);
                    let val = (norm_z * 255.0) as u8;
                    img.put_pixel(x, y, Rgba([val, val, val, 255]));
                } else {
                    img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
                }
            }
        }
        img
    }

    /// Converts the color buffer to an RGBA image.
    pub fn to_color_image(&self) -> RgbaImage {
        let mut img = RgbaImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.pixel_index(x, y);
                img.put_pixel(x, y, Rgba(self.colors[idx]));
            }
        }
        img
    }
}

/// Pure Rust Software Rasterizer for Hunyuan3D multi-view rendering.
pub struct SoftwareRasterizer;

impl SoftwareRasterizer {
    /// Renders a 3D mesh into a RenderBuffer containing depth, normals, and position maps.
    pub fn render(mesh: &Mesh, camera: &Camera) -> RenderBuffer {
        let mut buffer = RenderBuffer::new(camera.width, camera.height);
        let vp = camera.view_proj_matrix();
        let view_dir = camera.view_direction();

        let num_triangles = mesh.len_triangles();
        for tri_idx in 0..num_triangles {
            let i0 = mesh.indices[tri_idx * 3] as usize;
            let i1 = mesh.indices[tri_idx * 3 + 1] as usize;
            let i2 = mesh.indices[tri_idx * 3 + 2] as usize;

            let v0 = mesh.vertices[i0];
            let v1 = mesh.vertices[i1];
            let v2 = mesh.vertices[i2];

            let face_normal = (v1 - v0).cross(v2 - v0).normalize_or_zero();
            // Back-face culling
            if face_normal.dot(view_dir) >= 0.0 {
                continue;
            }

            // Project to clip space
            let c0 = vp.project_point3(v0);
            let c1 = vp.project_point3(v1);
            let c2 = vp.project_point3(v2);

            // Screen coordinates
            let (sx0, sy0) = camera.ndc_to_screen(c0);
            let (sx1, sy1) = camera.ndc_to_screen(c1);
            let (sx2, sy2) = camera.ndc_to_screen(c2);

            let min_x = sx0.min(sx1).min(sx2).floor().max(0.0) as u32;
            let max_x = sx0.max(sx1).max(sx2).ceil().min((camera.width - 1) as f32) as u32;
            let min_y = sy0.min(sy1).min(sy2).floor().max(0.0) as u32;
            let max_y = sy0.max(sy1).max(sy2).ceil().min((camera.height - 1) as f32) as u32;

            if min_x > max_x || min_y > max_y {
                continue;
            }

            let p0 = Vec2::new(sx0, sy0);
            let p1 = Vec2::new(sx1, sy1);
            let p2 = Vec2::new(sx2, sy2);

            let n0 = if !mesh.normals.is_empty() { mesh.normals[i0] } else { face_normal };
            let n1 = if !mesh.normals.is_empty() { mesh.normals[i1] } else { face_normal };
            let n2 = if !mesh.normals.is_empty() { mesh.normals[i2] } else { face_normal };

            for py in min_y..=max_y {
                for px in min_x..=max_x {
                    let p = Vec2::new(px as f32 + 0.5, py as f32 + 0.5);
                    let (w0, w1, w2) = barycentric_2d(p, p0, p1, p2);

                    if w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4 {
                        let z = w0 * c0.z + w1 * c1.z + w2 * c2.z;
                        let idx = buffer.pixel_index(px, py);

                        if z < buffer.depth[idx] && z >= -1.0 && z <= 1.0 {
                            buffer.depth[idx] = z;
                            buffer.triangle_ids[idx] = tri_idx as i32;

                            let interp_normal = (w0 * n0 + w1 * n1 + w2 * n2).normalize_or_zero();
                            buffer.normals[idx] = interp_normal;

                            let world_pos = w0 * v0 + w1 * v1 + w2 * v2;
                            buffer.positions[idx] = world_pos;

                            // Default shading (Lambertian)
                            let light_dir = Vec3::new(0.5, 0.8, 1.0).normalize();
                            let diffuse = interp_normal.dot(light_dir).max(0.1);
                            let color_val = (diffuse * 240.0) as u8;
                            buffer.colors[idx] = [color_val, color_val, color_val, 255];
                        }
                    }
                }
            }
        }

        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rasterizer_renders_cube() {
        let cube = Mesh::create_cube(2.0);
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::ZERO,
            Vec3::Y,
            45.0,
            64,
            64,
        );

        let buf = SoftwareRasterizer::render(&cube, &camera);
        // Center pixel should hit front face of the cube
        let center_idx = buf.pixel_index(32, 32);
        assert!(buf.triangle_ids[center_idx] >= 0);
        assert!(buf.depth[center_idx].is_finite());
        // Normal should face camera (+Z)
        let normal = buf.normals[center_idx];
        assert!(normal.z > 0.5);

        // Images should convert without error
        let _color_img = buf.to_color_image();
        let _normal_img = buf.to_normal_image();
        let _depth_img = buf.to_depth_image();
    }
}
