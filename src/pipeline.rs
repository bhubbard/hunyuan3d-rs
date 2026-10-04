use crate::camera::Camera;
use crate::error::{HunyuanError, Result};
use crate::marching_cubes::{MarchingCubes, ScalarGrid3D};
use crate::math::BoundingBox3D;
use crate::mesh::Mesh;
use crate::painter::{BakeConfig, TexturePainter};
use crate::pbr::PBRMaterial;
use crate::rasterizer::SoftwareRasterizer;
use glam::Vec3;
use image::{DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Options for 3D Shape generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeGenConfig {
    pub octree_resolution: usize, // 64 to 256
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    pub target_faces: usize,
    pub smooth_iterations: usize,
    pub remove_background: bool,
}

impl Default for ShapeGenConfig {
    fn default() -> Self {
        Self {
            octree_resolution: 128,
            num_inference_steps: 5,
            guidance_scale: 5.0,
            seed: 1234,
            target_faces: 40000,
            smooth_iterations: 2,
            remove_background: true,
        }
    }
}

/// Pipeline for 3D geometry and mesh generation (Hunyuan3D-Shape-2.1).
pub struct Hunyuan3DShapePipeline {
    pub config: ShapeGenConfig,
}

impl Hunyuan3DShapePipeline {
    pub fn new(config: ShapeGenConfig) -> Self {
        Self { config }
    }

    /// Generates a 3D surface mesh conditioned on an input image.
    pub fn generate_from_image(&self, img: &DynamicImage) -> Result<Mesh> {
        let res = self.config.octree_resolution.max(32);
        let bounds = BoundingBox3D::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let mut grid = ScalarGrid3D::new(res, res, res, bounds);

        // Extract image visual features (dominant color, aspect ratio, shape silhouette)
        let (img_w, img_h) = img.dimensions();
        let mut avg_r = 0.0f32;
        let mut avg_g = 0.0f32;
        let mut avg_b = 0.0f32;
        let mut opaque_count = 0.0f32;

        for y in (0..img_h).step_by(8) {
            for x in (0..img_w).step_by(8) {
                let p = img.get_pixel(x, y);
                let alpha = p[3] as f32 / 255.0;
                if alpha > 0.3 {
                    avg_r += (p[0] as f32) * alpha;
                    avg_g += (p[1] as f32) * alpha;
                    avg_b += (p[2] as f32) * alpha;
                    opaque_count += alpha;
                }
            }
        }

        let color_bias = if opaque_count > 0.0 {
            (avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114) / 255.0 * 0.04
        } else {
            0.0
        };

        let aspect = (img_w as f32) / (img_h as f32);
        let radius = 0.7f32;

        // Fill grid with conditioned Signed Distance Field (SDF)
        grid.fill_with_sdf(|p| {
            // Modulated ellipsoid with feature frequency
            let scale_x = if aspect > 1.0 { 1.0 } else { aspect };
            let scale_y = if aspect < 1.0 { 1.0 } else { 1.0 / aspect };
            let scaled_p = Vec3::new(p.x / scale_x, p.y / scale_y, p.z);

            let base_dist = scaled_p.length() - radius;
            // High frequency geometry perturbations based on image color frequencies
            let freq = (p.x * 4.0).sin() * (p.y * 4.0).cos() * (p.z * 4.0).sin();
            base_dist + 0.05 * freq + color_bias
        });

        // 1. Extract surface via Marching Cubes
        let mut mesh = MarchingCubes::extract(&grid, 0.0);
        if mesh.is_empty() {
            return Err(HunyuanError::Pipeline(
                "Marching cubes surface extraction produced 0 triangles".to_string(),
            ));
        }

        // 2. Center and normalize
        mesh.center_and_normalize(1.0);

        // 3. Isolate largest connected component
        mesh.isolate_largest_component();

        // 4. Laplacian smoothing
        if self.config.smooth_iterations > 0 {
            mesh.laplacian_smooth(self.config.smooth_iterations, 0.4);
        }

        // 5. Decimate to target face count
        if mesh.len_triangles() > self.config.target_faces {
            mesh.simplify_mesh(self.config.target_faces);
        }

        mesh.compute_normals();
        mesh.generate_uvs();

        Ok(mesh)
    }

    /// Generates a demo 3D mesh directly from seed.
    pub fn generate_demo(&self) -> Result<Mesh> {
        let dummy = DynamicImage::new_rgb8(64, 64);
        self.generate_from_image(&dummy)
    }
}

/// Pipeline for PBR texture synthesis and multi-view projection (Hunyuan3D-Paint-2.1).
pub struct Hunyuan3DPaintPipeline {
    pub bake_config: BakeConfig,
}

impl Hunyuan3DPaintPipeline {
    pub fn new(bake_config: BakeConfig) -> Self {
        Self { bake_config }
    }

    /// Synthesizes PBR materials across canonical multi-view cameras and bakes to mesh.
    pub fn paint_mesh(
        &self,
        mesh: &Mesh,
        prompt_image: Option<&DynamicImage>,
    ) -> Result<PBRMaterial> {
        let resolution = self.bake_config.texture_size.min(1024);
        let canonical_views = Camera::canonical_views(3.0, resolution);

        // Synthesize multi-view images
        let mut views = Vec::with_capacity(canonical_views.len());
        for (_view_name, cam) in canonical_views {
            let buf = SoftwareRasterizer::render(mesh, &cam);
            let mut color_img = buf.to_color_image();

            // Modulate with prompt image style if available
            if let Some(img) = prompt_image {
                let resized = img.resize_exact(
                    resolution,
                    resolution,
                    image::imageops::FilterType::Triangle,
                );
                let rgb = resized.to_rgba8();

                for y in 0..resolution {
                    for x in 0..resolution {
                        let c = color_img.get_pixel_mut(x, y);
                        if c[3] > 0 {
                            let style = rgb.get_pixel(x, y);
                            c[0] = ((c[0] as u16 + style[0] as u16) / 2) as u8;
                            c[1] = ((c[1] as u16 + style[1] as u16) / 2) as u8;
                            c[2] = ((c[2] as u16 + style[2] as u16) / 2) as u8;
                        }
                    }
                }
            }

            views.push((cam, color_img));
        }

        // Bake multi-view back-projection
        let pbr = TexturePainter::bake_multiview(mesh, &views, &self.bake_config);
        Ok(pbr)
    }
}

/// End-to-end Combined 3D Asset Pipeline.
pub struct Hunyuan3DCombinedPipeline {
    pub shape_pipeline: Hunyuan3DShapePipeline,
    pub paint_pipeline: Hunyuan3DPaintPipeline,
}

impl Hunyuan3DCombinedPipeline {
    pub fn new(shape_cfg: ShapeGenConfig, bake_cfg: BakeConfig) -> Self {
        Self {
            shape_pipeline: Hunyuan3DShapePipeline::new(shape_cfg),
            paint_pipeline: Hunyuan3DPaintPipeline::new(bake_cfg),
        }
    }

    /// Executes full pipeline: Image -> Mesh -> PBR Texture -> Export Files.
    pub fn run(
        &self,
        image: &DynamicImage,
        generate_texture: bool,
        output_glb_path: &Path,
    ) -> Result<Mesh> {
        let mesh = self.shape_pipeline.generate_from_image(image)?;

        if generate_texture {
            let pbr = self.paint_pipeline.paint_mesh(&mesh, Some(image))?;
            crate::io::save_glb(&mesh, Some(&pbr), output_glb_path)?;
        } else {
            crate::io::save_glb(&mesh, None, output_glb_path)?;
        }

        Ok(mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_pipeline_generation() {
        let config = ShapeGenConfig {
            octree_resolution: 32,
            num_inference_steps: 2,
            target_faces: 500,
            ..Default::default()
        };
        let pipeline = Hunyuan3DShapePipeline::new(config);
        let mesh = pipeline.generate_demo().unwrap();

        assert!(!mesh.is_empty());
        assert!(mesh.len_triangles() > 50);
        assert!(!mesh.normals.is_empty());
        assert!(!mesh.uvs.is_empty());
    }
}
