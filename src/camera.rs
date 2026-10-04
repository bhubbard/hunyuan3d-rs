use crate::math::{spherical_to_cartesian, Ray};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

/// Pinhole perspective camera for 3D rendering and texture projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_rad: f32,
    pub width: u32,
    pub height: u32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_deg: f32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            eye,
            target,
            up: up.normalize(),
            fov_y_rad: fov_y_deg.to_radians(),
            width,
            height,
            z_near: 0.01,
            z_far: 100.0,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn proj_matrix(&self) -> Mat4 {
        let aspect = (self.width as f32) / (self.height as f32);
        Mat4::perspective_rh(self.fov_y_rad, aspect, self.z_near, self.z_far)
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    pub fn view_direction(&self) -> Vec3 {
        (self.target - self.eye).normalize()
    }

    /// Converts normalized device coordinates (NDC [-1, 1]) to screen pixel coordinates.
    pub fn ndc_to_screen(&self, ndc: Vec3) -> (f32, f32) {
        let x = (ndc.x + 1.0) * 0.5 * (self.width as f32);
        let y = (1.0 - ndc.y) * 0.5 * (self.height as f32);
        (x, y)
    }

    /// Unprojects a 2D screen pixel coordinate (x, y) into a 3D world ray.
    pub fn screen_to_ray(&self, px: f32, py: f32) -> Ray {
        let ndc_x = (2.0 * px) / (self.width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * py) / (self.height as f32);

        let inv_vp = self.view_proj_matrix().inverse();
        let near_world = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, -1.0));
        let far_world = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));

        let dir = (far_world - near_world).normalize();
        Ray::new(near_world, dir)
    }

    /// Constructs the 6 canonical orthogonal views (Front, Right, Back, Left, Top, Bottom)
    /// exactly matching Hunyuan3D-Paint-2.1.
    pub fn canonical_views(radius: f32, resolution: u32) -> Vec<(String, Camera)> {
        let target = Vec3::ZERO;
        let fov = 45.0f32;

        let azims = [0.0, 90.0, 180.0, 270.0, 0.0, 180.0];
        let elevs = [0.0, 0.0, 0.0, 0.0, 89.9, -89.9];
        let names = ["front", "right", "back", "left", "top", "bottom"];

        let mut views = Vec::new();
        for i in 0..6 {
            let eye = spherical_to_cartesian(elevs[i], azims[i], radius);
            let up = if elevs[i] > 80.0 {
                Vec3::NEG_Z
            } else if elevs[i] < -80.0 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            views.push((
                names[i].to_string(),
                Camera::new(eye, target, up, fov, resolution, resolution),
            ));
        }
        views
    }

    /// Constructs a multi-view camera rig distributed evenly across a sphere
    /// using the Fibonacci lattice algorithm.
    pub fn fibonacci_views(count: usize, radius: f32, resolution: u32) -> Vec<Camera> {
        let mut cameras = Vec::with_capacity(count);
        let golden_ratio = (1.0 + 5.0f32.sqrt()) / 2.0;

        for i in 0..count {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden_ratio;
            let phi = (1.0 - 2.0 * ((i as f32) + 0.5) / (count as f32)).acos();

            let x = radius * phi.sin() * theta.cos();
            let y = radius * phi.cos();
            let z = radius * phi.sin() * theta.sin();

            let eye = Vec3::new(x, y, z);
            let up = if y.abs() > radius * 0.95 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            cameras.push(Camera::new(eye, Vec3::ZERO, up, 45.0, resolution, resolution));
        }
        cameras
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_camera_projection_unprojection() {
        let cam = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            256,
            256,
        );

        // Center pixel should shoot ray forward along -Z
        let ray = cam.screen_to_ray(128.0, 128.0);
        assert_relative_eq!(ray.direction.x, 0.0, epsilon = 1e-4);
        assert_relative_eq!(ray.direction.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(ray.direction.z, -1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_canonical_views() {
        let views = Camera::canonical_views(3.0, 512);
        assert_eq!(views.len(), 6);
        assert_eq!(views[0].0, "front");
        assert_eq!(views[1].0, "right");
        assert_eq!(views[2].0, "back");
        assert_eq!(views[3].0, "left");
        assert_eq!(views[4].0, "top");
        assert_eq!(views[5].0, "bottom");

        // Front view eye should be at (0, 0, 3)
        assert_relative_eq!(views[0].1.eye.x, 0.0, epsilon = 1e-3);
        assert_relative_eq!(views[0].1.eye.z, 3.0, epsilon = 1e-3);
    }
}
