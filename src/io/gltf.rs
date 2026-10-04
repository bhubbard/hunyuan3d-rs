use crate::error::Result;
use crate::mesh::Mesh;
use crate::pbr::PBRMaterial;
use byteorder::{LittleEndian, WriteBytesExt};
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Cursor, Write};
use std::path::Path;

/// Exports a 3D Mesh and optional PBR Material to a spec-compliant binary glTF 2.0 (.glb) file.
pub fn save_glb<P: AsRef<Path>>(
    mesh: &Mesh,
    pbr: Option<&PBRMaterial>,
    path: P,
) -> Result<()> {
    let mut bin_data = Vec::new();

    // 1. Pack Positions (VEC3, FLOAT 5126)
    let pos_offset = bin_data.len();
    let mut min_pos = [f32::INFINITY; 3];
    let mut max_pos = [f32::NEG_INFINITY; 3];

    for v in &mesh.vertices {
        min_pos[0] = min_pos[0].min(v.x);
        min_pos[1] = min_pos[1].min(v.y);
        min_pos[2] = min_pos[2].min(v.z);
        max_pos[0] = max_pos[0].max(v.x);
        max_pos[1] = max_pos[1].max(v.y);
        max_pos[2] = max_pos[2].max(v.z);

        bin_data.write_f32::<LittleEndian>(v.x)?;
        bin_data.write_f32::<LittleEndian>(v.y)?;
        bin_data.write_f32::<LittleEndian>(v.z)?;
    }
    let pos_length = bin_data.len() - pos_offset;
    pad_to_4_bytes(&mut bin_data);

    // 2. Pack Normals (VEC3, FLOAT 5126)
    let norm_offset = bin_data.len();
    for n in &mesh.normals {
        bin_data.write_f32::<LittleEndian>(n.x)?;
        bin_data.write_f32::<LittleEndian>(n.y)?;
        bin_data.write_f32::<LittleEndian>(n.z)?;
    }
    let norm_length = bin_data.len() - norm_offset;
    pad_to_4_bytes(&mut bin_data);

    // 3. Pack UVs (VEC2, FLOAT 5126)
    let uv_offset = bin_data.len();
    for uv in &mesh.uvs {
        bin_data.write_f32::<LittleEndian>(uv.x)?;
        bin_data.write_f32::<LittleEndian>(uv.y)?;
    }
    let uv_length = bin_data.len() - uv_offset;
    pad_to_4_bytes(&mut bin_data);

    // 4. Pack Indices (SCALAR, UNSIGNED_INT 5125)
    let indices_offset = bin_data.len();
    for &idx in &mesh.indices {
        bin_data.write_u32::<LittleEndian>(idx)?;
    }
    let indices_length = bin_data.len() - indices_offset;
    pad_to_4_bytes(&mut bin_data);

    // 5. Pack Textures (PNG bytes) if PBR provided
    let mut albedo_view_idx = None;
    let mut mr_view_idx = None;

    if let Some(mat) = pbr {
        // Albedo PNG
        let mut albedo_bytes = Vec::new();
        {
            let mut cursor = Cursor::new(&mut albedo_bytes);
            mat.albedo.write_to(&mut cursor, image::ImageFormat::Png)?;
        }
        let albedo_offset = bin_data.len();
        bin_data.extend_from_slice(&albedo_bytes);
        let albedo_len = bin_data.len() - albedo_offset;
        pad_to_4_bytes(&mut bin_data);

        // Metallic-Roughness PNG
        let packed_mr = mat.pack_gltf_metallic_roughness();
        let mut mr_bytes = Vec::new();
        {
            let mut cursor = Cursor::new(&mut mr_bytes);
            packed_mr.write_to(&mut cursor, image::ImageFormat::Png)?;
        }
        let mr_offset = bin_data.len();
        bin_data.extend_from_slice(&mr_bytes);
        let mr_len = bin_data.len() - mr_offset;
        pad_to_4_bytes(&mut bin_data);

        albedo_view_idx = Some((albedo_offset, albedo_len));
        mr_view_idx = Some((mr_offset, mr_len));
    }

    // Build JSON structure
    let mut buffer_views = vec![
        json!({
            "buffer": 0,
            "byteOffset": pos_offset,
            "byteLength": pos_length,
            "target": 34962 // ARRAY_BUFFER
        }),
        json!({
            "buffer": 0,
            "byteOffset": norm_offset,
            "byteLength": norm_length,
            "target": 34962
        }),
        json!({
            "buffer": 0,
            "byteOffset": uv_offset,
            "byteLength": uv_length,
            "target": 34962
        }),
        json!({
            "buffer": 0,
            "byteOffset": indices_offset,
            "byteLength": indices_length,
            "target": 34963 // ELEMENT_ARRAY_BUFFER
        }),
    ];

    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut materials = Vec::new();

    if let (Some((a_off, a_len)), Some((mr_off, mr_len))) = (albedo_view_idx, mr_view_idx) {
        let a_bv = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": a_off,
            "byteLength": a_len
        }));

        let mr_bv = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": mr_off,
            "byteLength": mr_len
        }));

        images.push(json!({ "bufferView": a_bv, "mimeType": "image/png" }));
        images.push(json!({ "bufferView": mr_bv, "mimeType": "image/png" }));

        textures.push(json!({ "sampler": 0, "source": 0 }));
        textures.push(json!({ "sampler": 0, "source": 1 }));

        materials.push(json!({
            "name": "Hunyuan3D_PBR_Material",
            "pbrMetallicRoughness": {
                "baseColorTexture": { "index": 0 },
                "metallicRoughnessTexture": { "index": 1 },
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0
            },
            "doubleSided": true
        }));
    } else {
        // Default untextured white material
        materials.push(json!({
            "name": "DefaultMaterial",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.85, 0.85, 0.85, 1.0],
                "metallicFactor": 0.1,
                "roughnessFactor": 0.6
            },
            "doubleSided": true
        }));
    }

    let accessors = vec![
        json!({
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126, // FLOAT
            "count": mesh.vertices.len(),
            "type": "VEC3",
            "min": min_pos,
            "max": max_pos
        }),
        json!({
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5126,
            "count": mesh.normals.len(),
            "type": "VEC3"
        }),
        json!({
            "bufferView": 2,
            "byteOffset": 0,
            "componentType": 5126,
            "count": mesh.uvs.len(),
            "type": "VEC2"
        }),
        json!({
            "bufferView": 3,
            "byteOffset": 0,
            "componentType": 5125, // UNSIGNED_INT
            "count": mesh.indices.len(),
            "type": "SCALAR"
        }),
    ];

    let mut gltf_json = json!({
        "asset": {
            "version": "2.0",
            "generator": "Hunyuan3D-RS 2.1 Engine"
        },
        "scene": 0,
        "scenes": [
            { "name": "DefaultScene", "nodes": [0] }
        ],
        "nodes": [
            { "name": "Hunyuan3D_Asset", "mesh": 0 }
        ],
        "meshes": [
            {
                "name": "Mesh",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2
                        },
                        "indices": 3,
                        "material": 0
                    }
                ]
            }
        ],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [
            { "byteLength": bin_data.len() }
        ]
    });

    if !textures.is_empty() {
        gltf_json["textures"] = json!(textures);
        gltf_json["images"] = json!(images);
        gltf_json["samplers"] = json!([
            {
                "magFilter": 9729, // LINEAR
                "minFilter": 9987, // LINEAR_MIPMAP_LINEAR
                "wrapS": 10497,    // REPEAT
                "wrapT": 10497
            }
        ]);
    }

    let json_string = serde_json::to_string(&gltf_json)?;
    let mut json_bytes = json_string.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' '); // Pad JSON with spaces
    }

    // Write binary GLB file
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let total_length = 12 + 8 + json_bytes.len() + 8 + bin_data.len();

    // 12-byte header
    writer.write_u32::<LittleEndian>(0x46546C67)?; // Magic: 'glTF'
    writer.write_u32::<LittleEndian>(2)?;          // Version: 2
    writer.write_u32::<LittleEndian>(total_length as u32)?;

    // Chunk 0: JSON
    writer.write_u32::<LittleEndian>(json_bytes.len() as u32)?;
    writer.write_u32::<LittleEndian>(0x4E4F534A)?; // 'JSON'
    writer.write_all(&json_bytes)?;

    // Chunk 1: BIN
    writer.write_u32::<LittleEndian>(bin_data.len() as u32)?;
    writer.write_u32::<LittleEndian>(0x004E4942)?; // 'BIN\0'
    writer.write_all(&bin_data)?;

    writer.flush()?;
    Ok(())
}

fn pad_to_4_bytes(vec: &mut Vec<u8>) {
    while vec.len() % 4 != 0 {
        vec.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_glb_with_pbr() {
        let cube = Mesh::create_cube(1.0);
        let pbr = PBRMaterial::procedural_demo(32, 32);

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        save_glb(&cube, Some(&pbr), path).unwrap();

        // Verify GLB file magic
        let bytes = std::fs::read(path).unwrap();
        assert!(bytes.len() > 100);
        assert_eq!(&bytes[0..4], b"glTF");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 2);
    }
}
