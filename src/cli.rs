use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Hunyuan3D-RS: Generative 3D Asset Studio & PBR Texture Synthesis.
#[derive(Parser, Debug)]
#[command(name = "hunyuan3d")]
#[command(author = "Brandon Hubbard <bkhubbard@gmail.com>, Tencent Hunyuan3D Team")]
#[command(version = "2.1.0")]
#[command(about = "High-performance Pure Rust Engine for Generative 3D Asset Creation with PBR Textures", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Generate 3D shape geometry from input image or synthetic prompt
    Shape(ShapeArgs),

    /// Synthesize PBR material textures and bake onto an existing 3D mesh
    Paint(PaintArgs),

    /// Execute the complete end-to-end Image-to-3D pipeline (Geometry + PBR -> GLB)
    Pipeline(PipelineArgs),

    /// Process, clean, simplify, or convert 3D meshes (OBJ <-> PLY <-> GLB)
    Mesh(MeshArgs),

    /// Launch high-throughput REST API server and interactive 3D web viewer
    Serve(ServeArgs),
}

#[derive(Args, Debug)]
pub struct ShapeArgs {
    /// Path to input conditioning image
    #[arg(short, long)]
    pub image: Option<PathBuf>,

    /// Resolution of the octree grid (64 to 256)
    #[arg(long, default_value_t = 128)]
    pub octree_res: usize,

    /// Number of Flow Matching inference steps
    #[arg(long, default_value_t = 5)]
    pub steps: usize,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 5.0)]
    pub guidance: f32,

    /// Target triangle face count after decimation
    #[arg(long, default_value_t = 40000)]
    pub faces: usize,

    /// Laplacian smoothing iterations
    #[arg(long, default_value_t = 2)]
    pub smooth: usize,

    /// Output mesh path (.obj or .ply)
    #[arg(short, long, default_value = "shape.obj")]
    pub output: PathBuf,
}

#[derive(Args, Debug)]
pub struct PaintArgs {
    /// Path to input 3D mesh (.obj or .ply)
    #[arg(short, long)]
    pub mesh: PathBuf,

    /// Path to input reference/style image
    #[arg(short, long)]
    pub image: Option<PathBuf>,

    /// Output textured model (.glb or .obj)
    #[arg(short, long, default_value = "textured.glb")]
    pub output: PathBuf,

    /// Texture atlas resolution (e.g. 512, 1024, 2048)
    #[arg(long, default_value_t = 1024)]
    pub texture_size: u32,
}

#[derive(Args, Debug)]
pub struct PipelineArgs {
    /// Path to input conditioning image
    #[arg(short, long)]
    pub image: Option<PathBuf>,

    /// Output GLB or OBJ asset path
    #[arg(short, long, default_value = "model.glb")]
    pub output: PathBuf,

    /// Resolution of the octree grid
    #[arg(long, default_value_t = 128)]
    pub octree_res: usize,

    /// Number of inference steps
    #[arg(long, default_value_t = 5)]
    pub steps: usize,

    /// Skip PBR texture synthesis
    #[arg(long)]
    pub no_texture: bool,
}

#[derive(Args, Debug)]
pub struct MeshArgs {
    /// Input mesh file (.obj, .ply, or .glb)
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output converted/simplified mesh file
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Simplify mesh to target face count
    #[arg(long)]
    pub simplify: Option<usize>,

    /// Apply Laplacian smoothing iterations
    #[arg(long)]
    pub smooth: Option<usize>,

    /// Display mesh geometry inspection info
    #[arg(long)]
    pub info: bool,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    pub port: u16,

    /// Host IP to bind
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
}
