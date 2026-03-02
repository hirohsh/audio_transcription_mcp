use crate::audio;
use crate::config::Config;
use crate::whisper::WhisperModel;
use anyhow::{Context, Result};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::{schemars, tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info};

#[derive(Deserialize, JsonSchema)]
pub struct TranscribeParams {
    /// Path to the audio file (must be within the allowed work directory)
    pub file_path: String,
    /// Language code (e.g., "ja", "en"). If omitted, language is auto-detected.
    #[serde(default)]
    pub language: Option<String>,
}

pub struct TranscriptionServiceInner {
    config: Config,
    model: Option<WhisperModel>,
}

#[derive(Clone)]
pub struct TranscriptionService {
    inner: Arc<std::sync::Mutex<TranscriptionServiceInner>>,
    tool_router: ToolRouter<Self>,
}

impl TranscriptionService {
    pub fn new(config: Config) -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(TranscriptionServiceInner {
                config,
                model: None,
            })),
            tool_router: Self::tool_router(),
        }
    }

    fn do_transcribe(&self, file_path: &str, language: Option<&str>) -> Result<String> {
        let mut inner = self.inner.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        let path = PathBuf::from(file_path);
        let path = if path.is_absolute() {
            path
        } else {
            inner.config.work_dir.join(&path)
        };

        let canonical_path = inner.config.validate_file_path(&path)?;
        info!("Transcribing: {}", canonical_path.display());
        inner.config.validate_file_size(&canonical_path)?;

        info!("Loading audio file...");
        let samples = audio::load_audio(&canonical_path)?;
        info!(
            "Audio loaded: {} samples ({:.1}s at 16kHz)",
            samples.len(),
            samples.len() as f64 / 16000.0
        );

        // Lazy-load model on first call
        if inner.model.is_none() {
            info!("Loading Whisper ONNX model (first request)...");
            let model = WhisperModel::new(&inner.config.model_dir)
                .with_context(|| format!(
                    "Failed to load Whisper model from {}. Run: cd model_converter && uv run convert --output-dir ../models",
                    inner.config.model_dir.display()
                ))?;
            info!("Whisper model loaded successfully");
            inner.model = Some(model);
        }

        let model = inner.model.as_ref().unwrap();
        let text = model.transcribe(&samples, language)?;
        info!("Transcription complete: {} characters", text.len());

        Ok(text)
    }
}

#[tool_router]
impl TranscriptionService {
    /// Transcribe an audio file to text using Whisper.
    /// Supported formats: wav, mp3, flac, ogg.
    /// The file must be within the allowed work directory and under the size limit.
    #[tool(
        name = "transcribe_audio",
        description = "Transcribe an audio file to text using Whisper. Supported formats: wav, mp3, flac, ogg. The file must be within the allowed work directory."
    )]
    fn transcribe_audio(
        &self,
        rmcp::handler::server::wrapper::Parameters(params): rmcp::handler::server::wrapper::Parameters<TranscribeParams>,
    ) -> String {
        match self.do_transcribe(&params.file_path, params.language.as_deref()) {
            Ok(text) => text,
            Err(e) => {
                error!("Transcription failed: {}", e);
                format!("Transcription failed: {}", e)
            }
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for TranscriptionService {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo {
            instructions: Some(
                "Audio transcription MCP server using Whisper ONNX model. \
                 Use the transcribe_audio tool to convert audio files to text."
                    .to_string(),
            ),
            ..Default::default()
        }
    }
}
