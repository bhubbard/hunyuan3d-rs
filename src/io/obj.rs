use crate::error::Result;
use crate::mesh::Mesh;
use crate::pbr::PBRMaterial;
use glam::{Vec2, Vec3};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Saves a mesh to Wavefront OBJ format.
pub fn save_obj<P: AsRef<Path>>(mesh: &Mesh, path: P, mtl_name: Option<&str>) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# Hunyuan3D-RS Generated OBJ File")?;
    if let Some(mtl) = mtl_name {
        writeln!(writer, "mtllib {}", mtl)?;
        writeln!(writer, "usemtl Material")?;
    }

    // Vertices
    for v in &mesh.vertices {
        writeln!(writer, "v {:.6} {:.6} {:.6}", v.x, v.y, v.z)?;
    }

    // Texture coordinates
    for uv in &mesh.uvs {
        writeln!(writer, "vt {:.6} {:.6}", uv.x, uv.y)?;
    }

    // Normals
    for n in &mesh.normals {
        writeln!(writer, "vn {:.6} {:.6} {:.6}", n.x, n.y, n.z)?;
    }

    // Faces (1-based indexing)
    let has_uv = !mesh.uvs.is_empty();
    let has_normal = !mesh.normals.is_empty();

    for chunk in mesh.indices.chunks_exact(3) {
        let i0 = chunk[0] + 1;
        let i1 = chunk[1] + 1;
        let i2 = chunk[2] + 1;

        if has_uv && has_normal {
            writeln!(writer, "f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}", i0, i1, i2)?;
        } else if has_uv {
            writeln!(writer, "f {0}/{0} {1}/{1} {2}/{2}", i0, i1, i2)?;
        } else if has_normal {
            writeln!(writer, "f {0}//{0} {1}//{1} {2}//{2}", i0, i1, i2)?;
        } else {
            writeln!(writer, "f {} {} {}", i0, i1, i2)?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Saves an OBJ mesh alongside an MTL file and PBR texture maps.
pub fn save_obj_with_pbr<P: AsRef<Path>>(
    mesh: &Mesh,
    pbr: &PBRMaterial,
    obj_path: P,
) -> Result<()> {
    let obj_path = obj_path.as_ref();
    let parent_dir = obj_path.parent().unwrap_or(Path::new("."));
    let stem = obj_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let mtl_filename = format!("{}.mtl", stem);
    let mtl_path = parent_dir.join(&mtl_filename);

    let albedo_filename = format!("{}_albedo.png", stem);
    let albedo_path = parent_dir.join(&albedo_filename);

    let mr_filename = format!("{}_mr.png", stem);
    let mr_path = parent_dir.join(&mr_filename);

    // Save textures
    pbr.albedo.save(&albedo_path)?;
    let packed_mr = pbr.pack_gltf_metallic_roughness();
    packed_mr.save(&mr_path)?;

    // Write MTL file
    let mtl_file = File::create(&mtl_path)?;
    let mut mtl_writer = BufWriter::new(mtl_file);
    writeln!(mtl_writer, "# Hunyuan3D-RS Material")?;
    writeln!(mtl_writer, "newmtl Material")?;
    writeln!(mtl_writer, "Ka 1.0 1.0 1.0")?;
    writeln!(mtl_writer, "Kd 1.0 1.0 1.0")?;
    writeln!(mtl_writer, "Ks 0.2 0.2 0.2")?;
    writeln!(mtl_writer, "d 1.0")?;
    writeln!(mtl_writer, "illum 2")?;
    writeln!(mtl_writer, "map_Kd {}", albedo_filename)?;
    writeln!(mtl_writer, "map_Ks {}", mr_filename)?;
    mtl_writer.flush()?;

    // Write OBJ file
    save_obj(mesh, obj_path, Some(&mtl_filename))?;

    Ok(())
}

/// Loads a mesh from a Wavefront OBJ file.
pub fn load_obj<P: AsRef<Path>>(path: P) -> Result<Mesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut raw_verts = Vec::new();
    let mut raw_uvs = Vec::new();
    let mut raw_normals = Vec::new();

    let mut mesh = Mesh::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let tag = match parts.next() {
            Some(t) => t,
            None => continue,
        };

        match tag {
            "v" => {
                let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                raw_verts.push(Vec3::new(x, y, z));
            }
            "vt" => {
                let u: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let v: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                raw_uvs.push(Vec2::new(u, v));
            }
            "vn" => {
                let nx: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let ny: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let nz: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                raw_normals.push(Vec3::new(nx, ny, nz));
            }
            "f" => {
                let face_verts: Vec<&str> = parts.collect();
                if face_verts.len() < 3 {
                    continue;
                }

                // Simple polygon fan triangulation for >3 verts
                let tri_indices = triangulate_fan(face_verts.len());
                for (t0, t1, t2) in tri_indices {
                    for &vert_str in &[face_verts[t0], face_verts[t1], face_verts[t2]] {
                        let parsed = parse_face_vertex(vert_str);
                        let v_idx = parsed.0.saturating_sub(1);
                        if v_idx < raw_verts.len() {
                            let new_v_idx = mesh.vertices.len() as u32;
                            mesh.vertices.push(raw_verts[v_idx]);

                            if let Some(vt_idx) = parsed.1 {
                                let idx = vt_idx.saturating_sub(1);
                                if idx < raw_uvs.len() {
                                    mesh.uvs.push(raw_uvs[idx]);
                                }
                            }

                            if let Some(vn_idx) = parsed.2 {
                                let idx = vn_idx.saturating_sub(1);
                                if idx < raw_normals.len() {
                                    mesh.normals.push(raw_normals[idx]);
                                }
                            }

                            mesh.indices.push(new_v_idx);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if mesh.normals.is_empty() {
        mesh.compute_normals();
    }
    if mesh.uvs.is_empty() {
        mesh.generate_uvs();
    }

    Ok(mesh)
}

fn triangulate_fan(num_verts: usize) -> Vec<(usize, usize, usize)> {
    let mut tris = Vec::new();
    for i in 1..num_verts - 1 {
        tris.push((0, i, i + 1));
    }
    tris
}

fn parse_face_vertex(token: &str) -> (usize, Option<usize>, Option<usize>) {
    let parts: Vec<&str> = token.split('/').collect();
    let v = parts[0].parse::<usize>().unwrap_or(0);
    let vt = if parts.len() > 1 && !parts[1].is_empty() {
        parts[1].parse::<usize>().ok()
    } else {
        None
    };
    let vn = if parts.len() > 2 && !parts[2].is_empty() {
        parts[2].parse::<usize>().ok()
    } else {
        None
    };
    (v, vt, vn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_obj_save_load_roundtrip() {
        let cube = Mesh::create_cube(1.5);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        save_obj(&cube, path, None).unwrap();
        let loaded = load_obj(path).unwrap();

        assert_eq!(loaded.len_triangles(), cube.len_triangles());
        assert!(!loaded.vertices.is_empty());
    }
}
