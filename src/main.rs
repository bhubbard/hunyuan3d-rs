use clap::Parser;
use colored::Colorize;
use hunyuan3d::cli::{Cli, Commands};
use hunyuan3d::error::Result;
use hunyuan3d::io::{load_obj, load_ply, save_glb, save_obj, save_obj_with_pbr, save_ply};
use hunyuan3d::mesh::Mesh;
use hunyuan3d::painter::BakeConfig;
use hunyuan3d::pipeline::{
    Hunyuan3DCombinedPipeline, Hunyuan3DPaintPipeline, Hunyuan3DShapePipeline, ShapeGenConfig,
};
use hunyuan3d::server::{create_router, ServerState};
use std::path::Path;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Shape(args) => {
            println!("{}", "▶ Generating 3D Shape geometry...".cyan().bold());
            let shape_cfg = ShapeGenConfig {
                octree_resolution: args.octree_res,
                num_inference_steps: args.steps,
                guidance_scale: args.guidance,
                target_faces: args.faces,
                smooth_iterations: args.smooth,
                ..Default::default()
            };
            let pipeline = Hunyuan3DShapePipeline::new(shape_cfg);

            let mesh = if let Some(ref img_path) = args.image {
                println!("  Conditioning image: {}", img_path.display().to_string().yellow());
                let img = image::open(img_path)?;
                pipeline.generate_from_image(&img)?
            } else {
                println!("  Generating synthetic demo asset (no input image supplied)");
                pipeline.generate_demo()?
            };

            println!(
                "✓ Extracted surface: {} vertices, {} triangles",
                mesh.len_vertices().to_string().green(),
                mesh.len_triangles().to_string().green()
            );

            let out_ext = args.output.extension().and_then(|s| s.to_str()).unwrap_or("obj");
            if out_ext.eq_ignore_ascii_case("ply") {
                save_ply(&mesh, &args.output)?;
            } else {
                save_obj(&mesh, &args.output, None)?;
            }

            println!("  Mesh saved to: {}", args.output.display().to_string().bold());
        }

        Commands::Paint(args) => {
            println!("{}", "▶ Synthesizing PBR textures & baking to mesh...".cyan().bold());
            println!("  Input mesh: {}", args.mesh.display().to_string().yellow());

            let mut mesh = load_mesh_any(&args.mesh)?;
            if mesh.uvs.is_empty() {
                mesh.generate_uvs();
            }

            let prompt_img = if let Some(ref img_path) = args.image {
                Some(image::open(img_path)?)
            } else {
                None
            };

            let bake_cfg = BakeConfig {
                texture_size: args.texture_size,
                bake_exp: 4.0,
                inpaint_radius: 3,
            };
            let pipeline = Hunyuan3DPaintPipeline::new(bake_cfg);
            let pbr = pipeline.paint_mesh(&mesh, prompt_img.as_ref())?;

            println!(
                "✓ Synthesized PBR maps (Albedo, Roughness, Metallic, Normals, AO: {}x{})",
                args.texture_size.to_string().green(),
                args.texture_size.to_string().green()
            );

            let out_ext = args.output.extension().and_then(|s| s.to_str()).unwrap_or("glb");
            if out_ext.eq_ignore_ascii_case("glb") {
                save_glb(&mesh, Some(&pbr), &args.output)?;
            } else {
                save_obj_with_pbr(&mesh, &pbr, &args.output)?;
            }

            println!("  Textured asset saved to: {}", args.output.display().to_string().bold());
        }

        Commands::Pipeline(args) => {
            println!("{}", "▶ Running full Hunyuan3D-2.1 Image-to-3D Pipeline...".cyan().bold());
            let shape_cfg = ShapeGenConfig {
                octree_resolution: args.octree_res,
                num_inference_steps: args.steps,
                ..Default::default()
            };
            let bake_cfg = BakeConfig {
                texture_size: 1024,
                bake_exp: 4.0,
                inpaint_radius: 3,
            };

            let pipeline = Hunyuan3DCombinedPipeline::new(shape_cfg, bake_cfg);
            let img = if let Some(ref img_path) = args.image {
                println!("  Input image: {}", img_path.display().to_string().yellow());
                image::open(img_path)?
            } else {
                println!("  Using synthetic test condition");
                image::DynamicImage::new_rgb8(128, 128)
            };

            let mesh = pipeline.run(&img, !args.no_texture, &args.output)?;
            println!(
                "✓ Generation complete: {} vertices, {} triangles",
                mesh.len_vertices().to_string().green(),
                mesh.len_triangles().to_string().green()
            );
            println!("  Exported 3D asset to: {}", args.output.display().to_string().bold());
        }

        Commands::Mesh(args) => {
            let mut mesh = load_mesh_any(&args.input)?;
            println!("{}", format!("▶ Loaded mesh: {}", args.input.display()).cyan());

            if args.info {
                let bbox = mesh.bounding_box();
                println!("  Vertices:  {}", mesh.len_vertices().to_string().green());
                println!("  Triangles: {}", mesh.len_triangles().to_string().green());
                println!("  Bounding Box: [{:.2}, {:.2}, {:.2}] to [{:.2}, {:.2}, {:.2}]",
                    bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);
                println!("  Dimensions:   [{:.2}, {:.2}, {:.2}]",
                    bbox.size().x, bbox.size().y, bbox.size().z);
            }

            if let Some(smooth_iters) = args.smooth {
                println!("  Applying {} iterations of Laplacian smoothing...", smooth_iters);
                mesh.laplacian_smooth(smooth_iters, 0.4);
            }

            if let Some(target_faces) = args.simplify {
                println!("  Simplifying mesh to {} target faces...", target_faces);
                mesh.simplify_mesh(target_faces);
                println!("  Remaining faces: {}", mesh.len_triangles().to_string().green());
            }

            if let Some(ref out_path) = args.output {
                let out_ext = out_path.extension().and_then(|s| s.to_str()).unwrap_or("obj");
                if out_ext.eq_ignore_ascii_case("ply") {
                    save_ply(&mesh, out_path)?;
                } else if out_ext.eq_ignore_ascii_case("glb") {
                    save_glb(&mesh, None, out_path)?;
                } else {
                    save_obj(&mesh, out_path, None)?;
                }
                println!("  Exported mesh to: {}", out_path.display().to_string().bold());
            }
        }

        Commands::Serve(args) => {
            let state = ServerState::new();
            let router = create_router(state);
            let addr = format!("{}:{}", args.host, args.port);

            println!("{}", "═══════════════════════════════════════════════════════════".dimmed());
            println!("  {}", "☯ Tencent Hunyuan3D-2.1 Studio (Pure Rust)".green().bold());
            println!("  Local Web Viewer:   {}", format!("http://localhost:{}", args.port).cyan().underline());
            println!("  REST API Health:    {}", format!("http://localhost:{}/health", args.port).cyan());
            println!("  Generate Endpoint:  {}", format!("http://localhost:{}/generate", args.port).cyan());
            println!("{}", "═══════════════════════════════════════════════════════════".dimmed());

            let listener = tokio::net::TcpListener::bind(&addr).await?;
            axum::serve(listener, router).await?;
        }
    }

    Ok(())
}

fn load_mesh_any(path: &Path) -> Result<Mesh> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("ply") {
        load_ply(path)
    } else {
        load_obj(path)
    }
}
