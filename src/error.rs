use thiserror::Error;

/// Core error types for Hunyuan3D-RS.
#[derive(Error, Debug)]
pub enum HunyuanError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Base64 decoding error: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("Invalid mesh topology: {0}")]
    InvalidMesh(String),

    #[error("Format error ({format}): {message}")]
    FormatError {
        format: &'static str,
        message: String,
    },

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Rendering error: {0}")]
    Rendering(String),

    #[error("Texture baking error: {0}")]
    TextureBaking(String),

    #[error("Server error: {0}")]
    Server(String),

    #[error("Math/Numerical error: {0}")]
    Math(String),
}

pub type Result<T> = std::result::Result<T, HunyuanError>;
