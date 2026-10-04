# Hunyuan3D-RS: Generative 3D Asset Studio & PBR Texture Synthesis

[![Crates.io](https://img.shields.io/badge/crates.io-v0.1.0-orange.svg)](https://crates.io)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Build & Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()

A high-performance, pure Rust engine and native fork of Tencent's **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** (State-of-the-art Generative 3D Asset Creation with Physically-Based Rendering Textures).

`hunyuan3d-rs` implements the complete 3D asset generation pipeline—including Marching Cubes surface reconstruction, Flow Matching diffusion, multi-view software rasterization, PBR material synthesis, texture baking, and binary glTF 2.0 (GLB) generation—entirely in native Rust. It runs seamlessly cross-platform on Apple Silicon, Linux, and Windows with multi-threaded CPU acceleration.

---

## 🌟 Key Features

### 1. Hunyuan3D-Shape-2.1 (Geometry Generation)
- **Marching Cubes Surface Extractor**: Pure Rust implementation with 256-configuration edge and triangle lookup tables (`MCSurfaceExtractor`), iso-surface interpolation, and bounding box normalization.
- **Rectified Flow Matching ODE**: Numerical solvers including **Euler**, **Midpoint**, **Heun** (2nd-order Runge-Kutta), and **RK4** with shifted $\sigma$-schedule $\sigma_t = \frac{s \cdot t}{1 + (s - 1) \cdot t}$ matching Hunyuan3D-2.1.
- **Spatio-Temporal DiT**: 3D Rotary Position Embeddings (3D RoPE) across spatial axes, scaled dot-product attention, AdaLN-Zero modulation, and Mixture of Experts (MoE) routing.
- **Mesh Post-Processing**:
  - Connected component graph analysis (pruning detached floating debris).
  - Iterative Laplacian smoothing.
  - Quadric Error Metric (QEM) decimation to target polygon count.
  - Smooth vertex normal and tangent basis calculation.

### 2. Hunyuan3D-Paint-2.1 (PBR Texture Synthesis)
- **Multi-View Software Rasterizer**: Pure Rust multi-threaded CPU renderer (`rayon`) producing high-resolution depth maps, normal maps (world- and view-space), and 3D position maps.
- **Physically-Based Rendering (PBR)**: Complete material channels:
  - **Albedo / Base Color** (sRGB / linear)
  - **Roughness** (microfacet specular roughness)
  - **Metallic** (conductor vs dielectric mask)
  - **Tangent-Space Normals**
  - **Ambient Occlusion (AO)**
- **Texture Baking & Inpainting**:
  - Multi-view back-projection with cosine angle weighting ($\cos \theta = \vec{N} \cdot \vec{V}$).
  - Depth buffer occlusion testing to eliminate back-face bleeding.
  - Seam dilation and UV inpainting to remove boundary artifacts.

### 3. Industry-Standard 3D File I/O
- **Binary glTF 2.0 (`.glb`)**: Spec-compliant generator embedding binary buffers, mesh primitives, and embedded PNG PBR textures (`pbrMetallicRoughness`). Opens directly in Blender, Three.js, Babylon.js, Unity, Unreal Engine, and native OS viewers (Mac QuickLook / Windows 3D Viewer).
- **Wavefront OBJ + MTL**: Polygon geometry with material definitions and texture maps.
- **Stanford PLY**: High-speed binary little-endian and ASCII mesh reader and writer.

### 4. REST API & Interactive 3D Web Studio
- Built with [`axum`](https://crates.io/crates/axum) and [`tokio`](https://crates.io/crates/tokio).
- Direct compatibility with upstream `api_server.py`:
  - `POST /generate`: Direct image-to-3D generation returning binary GLB.
  - `POST /api/v1/generate`: Async task submission returning `uid`.
  - `GET /api/v1/status/{uid}`: Task status and base64 GLB asset.
  - `GET /health`: Health check reporting worker status.
- Embedded interactive Three.js 3D WebGL viewer with orbit controls, lighting, and real-time generation HUD.

---

## 🚀 Quick Start

### Installation

Ensure you have Rust 1.80+ installed:

```bash
git clone https://github.com/bhubbard/hunyuan3d-rs.git
cd hunyuan3d-rs
cargo build --release
```

The compiled binary will be located at `target/release/hunyuan3d`.

---

## 🛠️ CLI Usage

```bash
hunyuan3d --help
```

### 1. Shape Generation (`shape`)

Generate 3D mesh geometry from an input image or synthetic prompt:

```bash
# Generate 3D mesh from an input image
hunyuan3d shape --image input.png --octree-res 128 --steps 5 --output shape.obj

# Generate a high-detail mesh (256 grid, 40k faces)
hunyuan3d shape --image photo.jpg --octree-res 256 --faces 40000 --output model.ply
```

### 2. PBR Texture Synthesis (`paint`)

Synthesize and bake PBR material textures onto an existing 3D mesh:

```bash
hunyuan3d paint \
  --mesh shape.obj \
  --image texture_ref.png \
  --texture-size 1024 \
  --output textured.glb
```

### 3. End-to-End Image-to-3D Pipeline (`pipeline`)

Generate a complete textured 3D asset in a single command:

```bash
hunyuan3d pipeline --image character.png --octree-res 128 --output character.glb
```

### 4. Mesh Processing & Conversion (`mesh`)

Simplify, smooth, inspect, or convert 3D files:

```bash
# Inspect mesh topology and bounding box
hunyuan3d mesh --input model.obj --info

# Simplify to 10,000 faces and apply Laplacian smoothing
hunyuan3d mesh --input high_poly.obj --simplify 10000 --smooth 3 --output clean.glb

# Convert OBJ to binary glTF
hunyuan3d mesh --input asset.obj --output asset.glb
```

### 5. Launch REST API Server & 3D Web Studio (`serve`)

```bash
hunyuan3d serve --port 8080
```

Navigate to `http://localhost:8080` in your web browser to access the interactive 3D WebGL studio.

---

## 🌐 HTTP REST API

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Embedded interactive Three.js 3D WebGL studio |
| `/health` | `GET` | Worker health status |
| `/generate` | `POST` | Generates a 3D model from image, returns binary GLB file |
| `/api/v1/generate` | `POST` | Asynchronous task submission, returns `{ "uid": "..." }` |
| `/api/v1/status/{uid}` | `GET` | Checks task status and returns Base64 model |

---

## 📦 Rust Library Usage

Add `hunyuan3d` to your `Cargo.toml`:

```toml
[dependencies]
hunyuan3d = { path = "../hunyuan3d-rs" }
```

### Example: Generating and Exporting a 3D Asset

```rust
use hunyuan3d::{
    Hunyuan3DCombinedPipeline, ShapeGenConfig, BakeConfig,
    save_glb,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure geometry and texture settings
    let shape_cfg = ShapeGenConfig {
        octree_resolution: 128,
        num_inference_steps: 5,
        ..Default::default()
    };
    let bake_cfg = BakeConfig {
        texture_size: 1024,
        bake_exp: 4.0,
        inpaint_radius: 3,
    };

    // 2. Initialize pipeline
    let pipeline = Hunyuan3DCombinedPipeline::new(shape_cfg, bake_cfg);

    // 3. Load input image
    let image = image::open("assets/demo.png")?;

    // 4. Generate textured 3D mesh and export to binary GLB
    let mesh = pipeline.run(&image, true, Path::new("output.glb"))?;
    println!("Generated mesh with {} triangles!", mesh.len_triangles());

    Ok(())
}
```

---

## 🔬 Comparison with Upstream Tencent Hunyuan3D-2.1

| Component | Tencent Hunyuan3D-2.1 (Python / PyTorch) | Hunyuan3D-RS (Pure Rust) | Parity Status |
| :--- | :--- | :--- | :--- |
| **Surface Extraction** | `skimage.measure.marching_cubes` (C/Python) | Pure Rust parallel Marching Cubes (`rayon`) | Full Parity |
| **PBR Texturing** | `DifferentiableRenderer` / PyTorch CUDA | Multi-threaded CPU software rasterizer & baker | Full Parity (No CUDA required) |
| **Mesh Post-Processing** | `trimesh` / `pymeshlab` | Native Laplacian smoothing, QEM decimation | Full Parity |
| **glTF 2.0 Exporter** | `trimesh.export` / Blender script | Spec-compliant native binary GLB generator | Full Parity |
| **Flow Matching ODE** | Diffusers Rectified Flow | `hunyuan3d::flow::RectifiedFlowSolver` (Euler, Heun, RK4) | Full Parity |
| **API Server** | FastAPI + Uvicorn (`api_server.py`) | Axum + Tokio multi-threaded server | Full Parity |
| **Interactive Viewer** | Gradio Python App (`gradio_app.py`) | Zero-dependency Three.js WebGL studio | Enhanced |

---

## 🧪 Testing

Run all unit tests across geometry, Marching Cubes, camera projections, software rasterization, and GLB export:

```bash
cargo test
```

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
