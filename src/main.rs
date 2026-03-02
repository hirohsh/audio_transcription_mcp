mod audio;
mod config;
mod mel;
mod tokenizer;
mod tools;
mod whisper;

use anyhow::{Context, Result};
use rmcp::ServiceExt;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = config::Config::from_env().with_context(|| "Failed to load configuration")?;

    // Initialize logging
    let filter = EnvFilter::try_new(&config.log_level).unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    info!("Audio Transcription MCP Server starting...");
    info!("Work directory: {}", config.work_dir.display());
    info!("Model directory: {}", config.model_dir.display());
    info!(
        "Max file size: {} MB",
        config.max_file_size_bytes / (1024 * 1024)
    );

    // Create transcription service (model is loaded lazily on first request)
    let service = tools::TranscriptionService::new(config);

    // Start MCP server with stdio transport
    info!("Starting MCP server on stdio...");
    let transport = rmcp::transport::io::stdio();
    let server = service
        .serve(transport)
        .await
        .with_context(|| "Failed to start MCP server")?;

    info!("MCP server running. Waiting for requests...");
    server.waiting().await?;

    info!("MCP server shutdown complete");
    Ok(())
}
