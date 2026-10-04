use image::{GrayImage, Luma, Rgba, RgbaImage};

/// Physically-Based Rendering (PBR) Material Maps.
#[derive(Debug, Clone)]
pub struct PBRMaterial {
    pub albedo: RgbaImage,
    pub metallic: GrayImage,
    pub roughness: GrayImage,
    pub normal: RgbaImage,
    pub ambient_occlusion: GrayImage,
}

impl PBRMaterial {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            albedo: RgbaImage::from_pixel(width, height, Rgba([200, 200, 200, 255])),
            metallic: GrayImage::from_pixel(width, height, Luma([0])),
            roughness: GrayImage::from_pixel(width, height, Luma([128])), // 0.5 roughness
            normal: RgbaImage::from_pixel(width, height, Rgba([128, 128, 255, 255])), // Flat tangent normal
            ambient_occlusion: GrayImage::from_pixel(width, height, Luma([255])),
        }
    }

    /// Packs Metallic, Roughness, and Ambient Occlusion into a single standard
    /// glTF PBR texture (R: Occlusion, G: Roughness, B: Metallic).
    pub fn pack_gltf_metallic_roughness(&self) -> RgbaImage {
        let (width, height) = self.metallic.dimensions();
        let mut packed = RgbaImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let r = self.ambient_occlusion.get_pixel(x, y)[0];
                let g = self.roughness.get_pixel(x, y)[0];
                let b = self.metallic.get_pixel(x, y)[0];
                packed.put_pixel(x, y, Rgba([r, g, b, 255]));
            }
        }

        packed
    }

    /// Generates procedural test PBR material with metallic highlights and surface color.
    pub fn procedural_demo(width: u32, height: u32) -> Self {
        let mut mat = Self::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let fx = (x as f32) / (width as f32);
                let fy = (y as f32) / (height as f32);

                // Grid pattern
                let check = ((fx * 8.0).floor() + (fy * 8.0).floor()) as i32 % 2 == 0;
                let albedo_col = if check {
                    Rgba([220, 180, 80, 255]) // Gold / brass
                } else {
                    Rgba([60, 70, 90, 255])  // Deep cobalt slate
                };
                mat.albedo.put_pixel(x, y, albedo_col);

                let metal_val = if check { 220 } else { 20 };
                let rough_val = if check { 60 } else { 180 };

                mat.metallic.put_pixel(x, y, Luma([metal_val]));
                mat.roughness.put_pixel(x, y, Luma([rough_val]));
            }
        }

        mat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbr_material_packing() {
        let mat = PBRMaterial::procedural_demo(64, 64);
        let packed = mat.pack_gltf_metallic_roughness();
        assert_eq!(packed.dimensions(), (64, 64));

        let p = packed.get_pixel(0, 0);
        assert_eq!(p[0], 255); // AO
        assert_eq!(p[3], 255); // Alpha
    }
}
