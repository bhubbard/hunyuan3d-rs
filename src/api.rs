use serde::{Deserialize, Serialize};

/// Request payload matching upstream Hunyuan3D-2.1 `GenerationRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Base64-encoded image input or data URL.
    pub image: String,
    #[serde(default = "default_true")]
    pub remove_background: bool,
    #[serde(default = "default_false")]
    pub texture: bool,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_octree")]
    pub octree_resolution: usize,
    #[serde(default = "default_steps")]
    pub num_inference_steps: usize,
    #[serde(default = "default_guidance")]
    pub guidance_scale: f32,
    #[serde(default = "default_faces")]
    pub face_count: usize,
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_seed() -> u64 {
    1234
}

fn default_octree() -> usize {
    128
}

fn default_steps() -> usize {
    5
}

fn default_guidance() -> f32 {
    5.0
}

fn default_faces() -> usize {
    40000
}

/// Generation response matching upstream `GenerationResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    pub uid: String,
    pub status: String,
}

/// Status response matching upstream `StatusResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub status: String,
    pub model_base64: Option<String>,
    pub message: Option<String>,
    pub face_count: Option<usize>,
    pub vertex_count: Option<usize>,
}

/// Health check response matching upstream `HealthResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub worker_id: String,
    pub engine: String,
    pub version: String,
}
