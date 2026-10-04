use crate::error::{HunyuanError, Result};
use crate::mesh::Mesh;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use glam::{Vec2, Vec3};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Saves a mesh to binary little-endian PLY format.
pub fn save_ply<P: AsRef<Path>>(mesh: &Mesh, path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let num_verts = mesh.vertices.len();
    let num_faces = mesh.len_triangles();

    let has_normals = mesh.normals.len() == num_verts;
    let has_uvs = mesh.uvs.len() == num_verts;

    // Header
    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "comment Hunyuan3D-RS generated PLY")?;
    writeln!(writer, "element vertex {}", num_verts)?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    if has_normals {
        writeln!(writer, "property float nx")?;
        writeln!(writer, "property float ny")?;
        writeln!(writer, "property float nz")?;
    }
    if has_uvs {
        writeln!(writer, "property float s")?;
        writeln!(writer, "property float t")?;
    }
    writeln!(writer, "element face {}", num_faces)?;
    writeln!(writer, "property list uchar uint vertex_indices")?;
    writeln!(writer, "end_header")?;

    // Binary vertex data
    for i in 0..num_verts {
        let v = mesh.vertices[i];
        writer.write_f32::<LittleEndian>(v.x)?;
        writer.write_f32::<LittleEndian>(v.y)?;
        writer.write_f32::<LittleEndian>(v.z)?;

        if has_normals {
            let n = mesh.normals[i];
            writer.write_f32::<LittleEndian>(n.x)?;
            writer.write_f32::<LittleEndian>(n.y)?;
            writer.write_f32::<LittleEndian>(n.z)?;
        }
        if has_uvs {
            let uv = mesh.uvs[i];
            writer.write_f32::<LittleEndian>(uv.x)?;
            writer.write_f32::<LittleEndian>(uv.y)?;
        }
    }

    // Binary face data
    for chunk in mesh.indices.chunks_exact(3) {
        writer.write_u8(3)?;
        writer.write_u32::<LittleEndian>(chunk[0])?;
        writer.write_u32::<LittleEndian>(chunk[1])?;
        writer.write_u32::<LittleEndian>(chunk[2])?;
    }

    writer.flush()?;
    Ok(())
}

/// Loads a mesh from an ASCII or binary PLY file.
pub fn load_ply<P: AsRef<Path>>(path: P) -> Result<Mesh> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        return Err(HunyuanError::FormatError {
            format: "PLY",
            message: "Missing 'ply' magic header".to_string(),
        });
    }

    let mut is_binary = false;
    let mut num_verts = 0usize;
    let mut num_faces = 0usize;
    let mut has_normals = false;
    let mut has_uvs = false;

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "format" => {
                if parts.len() > 1 && parts[1].starts_with("binary_little_endian") {
                    is_binary = true;
                }
            }
            "element" => {
                if parts.len() > 2 {
                    if parts[1] == "vertex" {
                        num_verts = parts[2].parse().unwrap_or(0);
                    } else if parts[1] == "face" {
                        num_faces = parts[2].parse().unwrap_or(0);
                    }
                }
            }
            "property" => {
                if parts.len() > 2 {
                    if parts[2] == "nx" {
                        has_normals = true;
                    } else if parts[2] == "s" || parts[2] == "u" {
                        has_uvs = true;
                    }
                }
            }
            "end_header" => break,
            _ => {}
        }
    }

    let mut vertices = Vec::with_capacity(num_verts);
    let mut normals = if has_normals { Vec::with_capacity(num_verts) } else { Vec::new() };
    let mut uvs = if has_uvs { Vec::with_capacity(num_verts) } else { Vec::new() };
    let mut indices = Vec::with_capacity(num_faces * 3);

    if is_binary {
        for _ in 0..num_verts {
            let x = reader.read_f32::<LittleEndian>()?;
            let y = reader.read_f32::<LittleEndian>()?;
            let z = reader.read_f32::<LittleEndian>()?;
            vertices.push(Vec3::new(x, y, z));

            if has_normals {
                let nx = reader.read_f32::<LittleEndian>()?;
                let ny = reader.read_f32::<LittleEndian>()?;
                let nz = reader.read_f32::<LittleEndian>()?;
                normals.push(Vec3::new(nx, ny, nz));
            }

            if has_uvs {
                let u = reader.read_f32::<LittleEndian>()?;
                let v = reader.read_f32::<LittleEndian>()?;
                uvs.push(Vec2::new(u, v));
            }
        }

        for _ in 0..num_faces {
            let count = reader.read_u8()?;
            if count == 3 {
                let i0 = reader.read_u32::<LittleEndian>()?;
                let i1 = reader.read_u32::<LittleEndian>()?;
                let i2 = reader.read_u32::<LittleEndian>()?;
                indices.push(i0);
                indices.push(i1);
                indices.push(i2);
            } else {
                for _ in 0..count {
                    let _ = reader.read_u32::<LittleEndian>()?;
                }
            }
        }
    } else {
        // ASCII parser
        for _ in 0..num_verts {
            line.clear();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let x = parts[0].parse().unwrap_or(0.0);
                let y = parts[1].parse().unwrap_or(0.0);
                let z = parts[2].parse().unwrap_or(0.0);
                vertices.push(Vec3::new(x, y, z));
            }
        }

        for _ in 0..num_faces {
            line.clear();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let count: usize = parts[0].parse().unwrap_or(0);
                if count == 3 {
                    let i0: u32 = parts[1].parse().unwrap_or(0);
                    let i1: u32 = parts[2].parse().unwrap_or(0);
                    let i2: u32 = parts[3].parse().unwrap_or(0);
                    indices.push(i0);
                    indices.push(i1);
                    indices.push(i2);
                }
            }
        }
    }

    let mut mesh = Mesh {
        vertices,
        normals,
        uvs,
        colors: Vec::new(),
        tangents: Vec::new(),
        indices,
    };

    if mesh.normals.is_empty() {
        mesh.compute_normals();
    }
    if mesh.uvs.is_empty() {
        mesh.generate_uvs();
    }

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ply_save_load_roundtrip() {
        let cube = Mesh::create_cube(2.0);
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        save_ply(&cube, path).unwrap();
        let loaded = load_ply(path).unwrap();

        assert_eq!(loaded.len_triangles(), cube.len_triangles());
        assert_eq!(loaded.len_vertices(), cube.len_vertices());
    }
}
