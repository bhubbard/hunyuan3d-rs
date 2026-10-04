use crate::api::{GenerationRequest, GenerationResponse, HealthResponse, StatusResponse};
use crate::pipeline::{Hunyuan3DCombinedPipeline, ShapeGenConfig};
use axum::{
    extract::{Path as AxumPath, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use base64::Engine;
use image::DynamicImage;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

/// In-memory server state tracking active generation tasks.
#[derive(Clone)]
pub struct ServerState {
    pub worker_id: String,
    pub tasks: Arc<Mutex<HashMap<String, StatusResponse>>>,
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            worker_id: format!("hy3d-{}", &uuid_simple()[..6]),
            tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/health", get(health_handler))
        .route("/generate", post(generate_direct_handler))
        .route("/api/v1/generate", post(generate_async_handler))
        .route("/api/v1/status/{uid}", get(status_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler(State(state): State<ServerState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        worker_id: state.worker_id,
        engine: "Hunyuan3D-RS 2.1 (Pure Rust)".to_string(),
        version: "2.1.0".to_string(),
    })
}

/// Upstream-compatible endpoint: POST /generate returning binary GLB.
async fn generate_direct_handler(
    State(_state): State<ServerState>,
    Json(req): Json<GenerationRequest>,
) -> std::result::Result<Response, StatusCode> {
    let img = decode_image_from_base64(&req.image)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let shape_cfg = ShapeGenConfig {
        octree_resolution: req.octree_resolution.clamp(32, 256),
        num_inference_steps: req.num_inference_steps.clamp(1, 20),
        guidance_scale: req.guidance_scale,
        seed: req.seed,
        target_faces: req.face_count,
        smooth_iterations: 2,
        remove_background: req.remove_background,
    };

    let bake_cfg = crate::painter::BakeConfig {
        texture_size: 512,
        bake_exp: 4.0,
        inpaint_radius: 2,
    };

    let pipeline = Hunyuan3DCombinedPipeline::new(shape_cfg, bake_cfg);
    let tmp_file = tempfile::Builder::new()
        .suffix(".glb")
        .tempfile()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let _mesh = pipeline
        .run(&img, req.texture, tmp_file.path())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let glb_bytes = std::fs::read(tmp_file.path())
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok((
        [
            (header::CONTENT_TYPE, "model/gltf-binary"),
            (header::CONTENT_DISPOSITION, "attachment; filename=\"model.glb\""),
        ],
        glb_bytes,
    ).into_response())
}

/// Upstream async task endpoint: POST /api/v1/generate.
async fn generate_async_handler(
    State(state): State<ServerState>,
    Json(req): Json<GenerationRequest>,
) -> std::result::Result<Json<GenerationResponse>, StatusCode> {
    let uid = format!("task-{}", uuid_simple());

    // Spawn async background generation task
    let tasks_map = state.tasks.clone();
    let task_uid = uid.clone();

    {
        let mut map = tasks_map.lock().unwrap();
        map.insert(
            task_uid.clone(),
            StatusResponse {
                status: "processing".to_string(),
                model_base64: None,
                message: None,
                face_count: None,
                vertex_count: None,
            },
        );
    }

    tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || -> crate::error::Result<(Vec<u8>, usize, usize)> {
            let img = decode_image_from_base64(&req.image)?;
            let shape_cfg = ShapeGenConfig {
                octree_resolution: req.octree_resolution.clamp(32, 256),
                num_inference_steps: req.num_inference_steps.clamp(1, 20),
                guidance_scale: req.guidance_scale,
                seed: req.seed,
                target_faces: req.face_count,
                smooth_iterations: 2,
                remove_background: req.remove_background,
            };
            let bake_cfg = crate::painter::BakeConfig {
                texture_size: 512,
                bake_exp: 4.0,
                inpaint_radius: 2,
            };
            let pipeline = Hunyuan3DCombinedPipeline::new(shape_cfg, bake_cfg);
            let tmp = tempfile::Builder::new().suffix(".glb").tempfile()?;
            let mesh = pipeline.run(&img, req.texture, tmp.path())?;
            let bytes = std::fs::read(tmp.path())?;
            Ok((bytes, mesh.len_triangles(), mesh.len_vertices()))
        }).await;

        let mut map = tasks_map.lock().unwrap();
        match result {
            Ok(Ok((bytes, faces, verts))) => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                map.insert(
                    task_uid,
                    StatusResponse {
                        status: "completed".to_string(),
                        model_base64: Some(b64),
                        message: None,
                        face_count: Some(faces),
                        vertex_count: Some(verts),
                    },
                );
            }
            Ok(Err(e)) => {
                map.insert(
                    task_uid,
                    StatusResponse {
                        status: "error".to_string(),
                        model_base64: None,
                        message: Some(e.to_string()),
                        face_count: None,
                        vertex_count: None,
                    },
                );
            }
            Err(e) => {
                map.insert(
                    task_uid,
                    StatusResponse {
                        status: "error".to_string(),
                        model_base64: None,
                        message: Some(e.to_string()),
                        face_count: None,
                        vertex_count: None,
                    },
                );
            }
        }
    });

    Ok(Json(GenerationResponse {
        uid,
        status: "processing".to_string(),
    }))
}

async fn status_handler(
    State(state): State<ServerState>,
    AxumPath(uid): AxumPath<String>,
) -> std::result::Result<Json<StatusResponse>, StatusCode> {
    let map = state.tasks.lock().unwrap();
    if let Some(status) = map.get(&uid) {
        Ok(Json(status.clone()))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

fn decode_image_from_base64(raw: &str) -> crate::error::Result<DynamicImage> {
    let clean = if let Some(idx) = raw.find(',') {
        &raw[idx + 1..]
    } else {
        raw
    };

    let bytes = base64::engine::general_purpose::STANDARD.decode(clean)?;
    let img = image::load_from_memory(&bytes)?;
    Ok(img)
}

fn uuid_simple() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Embedded zero-dependency HTML5 3D WebGL viewer for Hunyuan3D assets.
async fn index_handler() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Tencent Hunyuan3D-2.1: Generative 3D Asset Studio</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f1117; color: #e6edf3; display: flex; height: 100vh; overflow: hidden; }
    #sidebar { width: 360px; background: #161b22; border-right: 1px solid #30363d; display: flex; flex-direction: column; padding: 20px; gap: 16px; overflow-y: auto; }
    #viewport { flex: 1; position: relative; background: radial-gradient(circle at center, #1b2230 0%, #0d1117 100%); }
    h1 { font-size: 1.25rem; font-weight: 700; color: #58a6ff; display: flex; align-items: center; gap: 8px; }
    .badge { background: #238636; color: white; font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; }
    label { font-size: 0.85rem; color: #8b949e; display: flex; flex-direction: column; gap: 6px; }
    input[type="file"], select, input[type="range"] { width: 100%; }
    button { background: #238636; color: white; border: none; border-radius: 6px; padding: 10px 16px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    button:hover { background: #2ea043; }
    .card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-size: 0.8rem; line-height: 1.5; color: #8b949e; }
    #hud { position: absolute; top: 16px; left: 16px; pointer-events: none; background: rgba(13,17,23,0.85); border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; font-family: monospace; font-size: 0.75rem; }
    canvas { width: 100%; height: 100%; display: block; }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
</head>
<body>
  <div id="sidebar">
    <h1>Hunyuan3D-2.1 <span class="badge">Rust</span></h1>
    <div class="card">
      High-performance Pure Rust 3D Asset Synthesis Engine with PBR Textures.
    </div>

    <label>
      Input Image
      <input type="file" id="imageInput" accept="image/*">
    </label>

    <label>
      Resolution (Octree Grid)
      <select id="octreeRes">
        <option value="64">64 (Ultra Fast)</option>
        <option value="128" selected>128 (Standard)</option>
        <option value="256">256 (High Detail)</option>
      </select>
    </label>

    <label>
      Inference Steps
      <input type="range" id="stepsInput" min="2" max="20" value="5">
      <span id="stepsVal">5 steps (Heun Flow Matching)</span>
    </label>

    <label style="flex-direction: row; align-items: center; gap: 8px;">
      <input type="checkbox" id="textureCheck" checked>
      Generate PBR Textures
    </label>

    <button id="genBtn" onclick="generateAsset()">Generate 3D Asset</button>

    <div class="card" id="statusCard" style="display: none;">
      <span id="statusText">Ready</span>
    </div>
  </div>

  <div id="viewport">
    <div id="hud">
      <div>ENGINE: Hunyuan3D-RS 2.1</div>
      <div>VERTICES: <span id="hudVerts">0</span></div>
      <div>TRIANGLES: <span id="hudFaces">0</span></div>
      <div>STATUS: <span id="hudStatus">Idle</span></div>
    </div>
  </div>

  <script>
    let scene, camera, renderer, controls, currentMesh;

    function initThree() {
      const container = document.getElementById('viewport');
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
      camera.position.set(0, 1.5, 3);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      // Lights
      const ambient = new THREE.AmbientLight(0xffffff, 0.8);
      scene.add(ambient);
      const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
      dirLight.position.set(2, 4, 3);
      scene.add(dirLight);

      // Grid
      const grid = new THREE.GridHelper(4, 20, 0x30363d, 0x21262d);
      grid.position.y = -0.5;
      scene.add(grid);

      // Load sample shape
      const geo = new THREE.TorusKnotGeometry(0.5, 0.18, 100, 16);
      const mat = new THREE.MeshStandardMaterial({ color: 0x58a6ff, metalness: 0.3, roughness: 0.4 });
      currentMesh = new THREE.Mesh(geo, mat);
      scene.add(currentMesh);

      document.getElementById('hudVerts').innerText = geo.attributes.position.count;
      document.getElementById('hudFaces').innerText = geo.index.count / 3;

      window.addEventListener('resize', onWindowResize);
      animate();
    }

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    function onWindowResize() {
      const container = document.getElementById('viewport');
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }

    document.getElementById('stepsInput').addEventListener('input', (e) => {
      document.getElementById('stepsVal').innerText = e.target.value + ' steps (Heun Flow Matching)';
    });

    async function generateAsset() {
      const fileInput = document.getElementById('imageInput');
      const statusCard = document.getElementById('statusCard');
      const statusText = document.getElementById('statusText');
      statusCard.style.display = 'block';
      statusText.innerText = 'Processing input image...';
      document.getElementById('hudStatus').innerText = 'Generating...';

      let base64 = "";
      if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        base64 = await toBase64(file);
      } else {
        // Transparent 1x1 fallback dummy
        base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
      }

      const payload = {
        image: base64,
        remove_background: true,
        texture: document.getElementById('textureCheck').checked,
        seed: 1234,
        octree_resolution: parseInt(document.getElementById('octreeRes').value),
        num_inference_steps: parseInt(document.getElementById('stepsInput').value),
        guidance_scale: 5.0,
        face_count: 40000
      };

      try {
        statusText.innerText = 'Synthesizing 3D geometry & PBR maps...';
        const res = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('Generation failed: ' + res.statusText);

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);

        const loader = new THREE.GLTFLoader();
        loader.load(url, (gltf) => {
          if (currentMesh) scene.remove(currentMesh);
          currentMesh = gltf.scene;
          scene.add(currentMesh);

          let totalVerts = 0, totalFaces = 0;
          currentMesh.traverse((child) => {
            if (child.isMesh) {
              totalVerts += child.geometry.attributes.position.count;
              if (child.geometry.index) totalFaces += child.geometry.index.count / 3;
            }
          });

          document.getElementById('hudVerts').innerText = totalVerts;
          document.getElementById('hudFaces').innerText = totalFaces;
          document.getElementById('hudStatus').innerText = 'Complete';
          statusText.innerText = '✓ Generation successful!';
        });
      } catch (err) {
        statusText.innerText = 'Error: ' + err.message;
        document.getElementById('hudStatus').innerText = 'Error';
      }
    }

    function toBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
      });
    }

    window.onload = initThree;
  </script>
</body>
</html>"#)
}
