use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Axis-aligned 3D bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox3D {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox3D {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    pub fn from_points(points: &[Vec3]) -> Self {
        let mut bbox = Self::empty();
        for &p in points {
            bbox.include_point(p);
        }
        bbox
    }

    pub fn include_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn center(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn extents(&self) -> Vec3 {
        0.5 * self.size()
    }

    pub fn max_dimension(&self) -> f32 {
        let s = self.size();
        s.x.max(s.y).max(s.z)
    }

    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }
}

/// 3D Ray.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Möller–Trumbore ray-triangle intersection.
/// Returns Some((t, u, v)) where t is ray distance, and u, v are barycentric coords.
pub fn ray_triangle_intersect(
    ray: &Ray,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<(f32, f32, f32)> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);

    if a.abs() < 1e-8 {
        return None; // Ray is parallel to triangle
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(h);

    if u < 0.0 || u > 1.0 {
        return None;
    }

    let q = s.cross(edge1);
    let v = f * ray.direction.dot(q);

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(q);
    if t > 1e-8 {
        Some((t, u, v))
    } else {
        None // Triangle is behind ray origin
    }
}

/// Computes 2D barycentric coordinates for point p with respect to triangle (a, b, c).
/// Returns (w0, w1, w2) where p = w0*a + w1*b + w2*c.
pub fn barycentric_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-8 {
        return (0.3333, 0.3333, 0.3333);
    }

    let inv_denom = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    (u, v, w)
}

/// Computes tangent and bitangent vectors for normal mapping.
pub fn compute_tangent_basis(
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    uv0: Vec2,
    uv1: Vec2,
    uv2: Vec2,
    normal: Vec3,
) -> (Vec3, Vec3) {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let delta_uv1 = uv1 - uv0;
    let delta_uv2 = uv2 - uv0;

    let f = delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y;
    let r = if f.abs() < 1e-8 { 1.0 } else { 1.0 / f };

    let mut tangent = (edge1 * delta_uv2.y - edge2 * delta_uv1.y) * r;
    // Gram-Schmidt orthogonalize
    tangent = (tangent - normal * normal.dot(tangent)).normalize_or_zero();

    let bitangent = normal.cross(tangent).normalize_or_zero();

    (tangent, bitangent)
}

/// Converts spherical coordinates (elevation in degrees, azimuth in degrees, radius)
/// to Cartesian 3D coordinates.
pub fn spherical_to_cartesian(elevation_deg: f32, azimuth_deg: f32, radius: f32) -> Vec3 {
    let elev = elevation_deg.to_radians();
    let azim = azimuth_deg.to_radians();

    let cos_elev = elev.cos();
    let x = radius * cos_elev * azim.sin();
    let y = radius * elev.sin();
    let z = radius * cos_elev * azim.cos();

    Vec3::new(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bounding_box() {
        let points = vec![
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(3.0, -2.0, 1.0),
            Vec3::new(0.0, 4.0, -1.0),
        ];
        let bbox = BoundingBox3D::from_points(&points);
        assert_relative_eq!(bbox.min.x, -1.0);
        assert_relative_eq!(bbox.min.y, -2.0);
        assert_relative_eq!(bbox.min.z, -1.0);
        assert_relative_eq!(bbox.max.x, 3.0);
        assert_relative_eq!(bbox.max.y, 4.0);
        assert_relative_eq!(bbox.max.z, 2.0);
        assert_relative_eq!(bbox.center().x, 1.0);
        assert_relative_eq!(bbox.center().y, 1.0);
        assert_relative_eq!(bbox.center().z, 0.5);
    }

    #[test]
    fn test_ray_triangle_intersection() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        let hit = ray_triangle_intersect(&ray, v0, v1, v2);
        assert!(hit.is_some());
        let (t, u, v) = hit.unwrap();
        assert_relative_eq!(t, 5.0);
        assert!(u >= 0.0 && v >= 0.0 && u + v <= 1.0);
    }

    #[test]
    fn test_spherical_coordinates() {
        // Front view: azim = 0, elev = 0 -> (0, 0, radius)
        let p = spherical_to_cartesian(0.0, 0.0, 2.0);
        assert_relative_eq!(p.x, 0.0, epsilon = 1e-4);
        assert_relative_eq!(p.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(p.z, 2.0, epsilon = 1e-4);

        // Top view: azim = 0, elev = 90 -> (0, radius, 0)
        let top = spherical_to_cartesian(90.0, 0.0, 2.0);
        assert_relative_eq!(top.x, 0.0, epsilon = 1e-4);
        assert_relative_eq!(top.y, 2.0, epsilon = 1e-4);
        assert_relative_eq!(top.z, 0.0, epsilon = 1e-4);
    }
}
