use crate::camera::Camera;
use crate::math::barycentric_2d;
use crate::mesh::Mesh;
use crate::pbr::PBRMaterial;
use crate::rasterizer::SoftwareRasterizer;
use glam::{Vec2, Vec3};
use image::{GrayImage, Luma, Rgba, RgbaImage};
use rayon::prelude::*;

/// Configuration for texture baking and multiview projection.
#[derive(Debug, Clone)]
pub struct BakeConfig {
    pub texture_size: u32,
    pub bake_exp: f32, // Exponent for cosine angle weighting (default 4.0 matching upstream)
    pub inpaint_radius: usize,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            texture_size: 1024,
            bake_exp: 4.0,
            inpaint_radius: 4,
        }
    }
}

/// Multiview PBR Texture Painter and Baker.
/// Implements the texture synthesis and back-projection pipeline of Hunyuan3D-Paint-2.1.
pub struct TexturePainter;

impl TexturePainter {
    /// Bakes a texture atlas from multi-view images projected onto a 3D mesh.
    pub fn bake_multiview(
        mesh: &Mesh,
        views: &[(Camera, RgbaImage)],
        config: &BakeConfig,
    ) -> PBRMaterial {
        let size = config.texture_size;
        let mut albedo = RgbaImage::new(size, size);
        let mut mask = GrayImage::new(size, size);

        // Precompute view render buffers for depth occlusion checking
        let mut view_depths = Vec::with_capacity(views.len());
        for (cam, _) in views {
            let buf = SoftwareRasterizer::render(mesh, cam);
            view_depths.push(buf);
        }

        // 1. Rasterize UV space into 3D world positions and normals
        let uv_buffer = rasterize_uv_space(mesh, size, size);

        // 2. Multi-threaded back-projection per texel
        let texels: Vec<(Rgba<u8>, u8)> = (0..size * size)
            .into_par_iter()
            .map(|idx| {
                let world_pos = uv_buffer.positions[idx as usize];
                let normal = uv_buffer.normals[idx as usize];
                let is_valid = uv_buffer.valid[idx as usize];

                if !is_valid {
                    return (Rgba([128, 128, 128, 0]), 0);
                }

                let mut accum_color = Vec3::ZERO;
                let mut total_weight = 0.0f32;

                for (view_idx, (cam, img)) in views.iter().enumerate() {
                    let to_cam = (cam.eye - world_pos).normalize_or_zero();
                    let cos_angle = normal.dot(to_cam);

                    if cos_angle <= 0.0 {
                        continue;
                    }

                    // Project world pos to camera screen
                    let vp = cam.view_proj_matrix();
                    let clip = vp.project_point3(world_pos);

                    if clip.x < -1.0 || clip.x > 1.0 || clip.y < -1.0 || clip.y > 1.0 || clip.z < -1.0 || clip.z > 1.0 {
                        continue;
                    }

                    let (sx, sy) = cam.ndc_to_screen(clip);
                    let px = (sx.round() as u32).min(cam.width - 1);
                    let py = (sy.round() as u32).min(cam.height - 1);

                    // Depth test
                    let depth_buf = &view_depths[view_idx];
                    let expected_z = depth_buf.depth[depth_buf.pixel_index(px, py)];
                    let depth_diff = (clip.z - expected_z).abs();

                    if depth_diff < 0.08 {
                        let weight = cos_angle.powf(config.bake_exp);
                        let p = img.get_pixel(px, py);
                        let c = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32);

                        accum_color += c * weight;
                        total_weight += weight;
                    }
                }

                if total_weight > 1e-4 {
                    let final_c = accum_color / total_weight;
                    let r = final_c.x.clamp(0.0, 255.0) as u8;
                    let g = final_c.y.clamp(0.0, 255.0) as u8;
                    let b = final_c.z.clamp(0.0, 255.0) as u8;
                    (Rgba([r, g, b, 255]), 255)
                } else {
                    (Rgba([0, 0, 0, 0]), 0)
                }
            })
            .collect();

        for y in 0..size {
            for x in 0..size {
                let idx = (y * size + x) as usize;
                albedo.put_pixel(x, y, texels[idx].0);
                mask.put_pixel(x, y, Luma([texels[idx].1]));
            }
        }

        // 3. Inpaint / Dilate UV boundaries to remove seams
        let inpainted_albedo = inpaint_texture(&albedo, &mask, config.inpaint_radius);

        let mut pbr = PBRMaterial::new(size, size);
        pbr.albedo = inpainted_albedo;
        pbr
    }
}

struct UVRasterBuffer {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub valid: Vec<bool>,
}

fn rasterize_uv_space(mesh: &Mesh, width: u32, height: u32) -> UVRasterBuffer {
    let size = (width * height) as usize;
    let mut positions = vec![Vec3::ZERO; size];
    let mut normals = vec![Vec3::ZERO; size];
    let mut valid = vec![false; size];

    let num_triangles = mesh.len_triangles();
    for tri_idx in 0..num_triangles {
        let i0 = mesh.indices[tri_idx * 3] as usize;
        let i1 = mesh.indices[tri_idx * 3 + 1] as usize;
        let i2 = mesh.indices[tri_idx * 3 + 2] as usize;

        let uv0 = mesh.uvs[i0];
        let uv1 = mesh.uvs[i1];
        let uv2 = mesh.uvs[i2];

        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];

        let n0 = if !mesh.normals.is_empty() { mesh.normals[i0] } else { Vec3::Y };
        let n1 = if !mesh.normals.is_empty() { mesh.normals[i1] } else { Vec3::Y };
        let n2 = if !mesh.normals.is_empty() { mesh.normals[i2] } else { Vec3::Y };

        let p0 = Vec2::new(uv0.x * (width as f32), (1.0 - uv0.y) * (height as f32));
        let p1 = Vec2::new(uv1.x * (width as f32), (1.0 - uv1.y) * (height as f32));
        let p2 = Vec2::new(uv2.x * (width as f32), (1.0 - uv2.y) * (height as f32));

        let min_x = p0.x.min(p1.x).min(p2.x).floor().max(0.0) as u32;
        let max_x = p0.x.max(p1.x).max(p2.x).ceil().min((width - 1) as f32) as u32;
        let min_y = p0.y.min(p1.y).min(p2.y).floor().max(0.0) as u32;
        let max_y = p0.y.max(p1.y).max(p2.y).ceil().min((height - 1) as f32) as u32;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let p = Vec2::new(px as f32 + 0.5, py as f32 + 0.5);
                let (w0, w1, w2) = barycentric_2d(p, p0, p1, p2);

                if w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4 {
                    let idx = (py * width + px) as usize;
                    positions[idx] = w0 * v0 + w1 * v1 + w2 * v2;
                    normals[idx] = (w0 * n0 + w1 * n1 + w2 * n2).normalize_or_zero();
                    valid[idx] = true;
                }
            }
        }
    }

    UVRasterBuffer {
        positions,
        normals,
        valid,
    }
}

/// Dilates valid texture pixels into unmapped regions to prevent seam bleeding.
fn inpaint_texture(img: &RgbaImage, mask: &GrayImage, radius: usize) -> RgbaImage {
    let (w, h) = img.dimensions();
    let mut result = img.clone();

    for _ in 0..radius {
        let temp = result.clone();
        for y in 0..h {
            for x in 0..w {
                if mask.get_pixel(x, y)[0] == 0 {
                    // Average neighbor valid pixels
                    let mut accum = Vec3::ZERO;
                    let mut count = 0;

                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                                let p = temp.get_pixel(nx as u32, ny as u32);
                                if p[3] > 0 {
                                    accum += Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32);
                                    count += 1;
                                }
                            }
                        }
                    }

                    if count > 0 {
                        let avg = accum / (count as f32);
                        result.put_pixel(
                            x,
                            y,
                            Rgba([avg.x as u8, avg.y as u8, avg.z as u8, 255]),
                        );
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_painter_baking() {
        let cube = Mesh::create_cube(2.0);
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::ZERO,
            Vec3::Y,
            45.0,
            64,
            64,
        );

        // Fake front photo (solid blue)
        let mut img = RgbaImage::new(64, 64);
        for p in img.pixels_mut() {
            *p = Rgba([50, 100, 200, 255]);
        }

        let views = vec![(cam, img)];
        let config = BakeConfig {
            texture_size: 64,
            bake_exp: 2.0,
            inpaint_radius: 1,
        };

        let pbr = TexturePainter::bake_multiview(&cube, &views, &config);
        assert_eq!(pbr.albedo.dimensions(), (64, 64));
    }
}
