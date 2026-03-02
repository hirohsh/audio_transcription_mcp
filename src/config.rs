use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub work_dir: PathBuf,
    pub max_file_size_bytes: u64,
    pub model_dir: PathBuf,
    pub log_level: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let _ = dotenvy::dotenv();

        let work_dir = std::env::var("MCP_WORK_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let work_dir = work_dir
            .canonicalize()
            .with_context(|| format!("Failed to canonicalize work_dir: {}", work_dir.display()))?;

        let max_file_size_mb: u64 = std::env::var("MCP_MAX_FILE_SIZE_MB")
            .unwrap_or_else(|_| "25".to_string())
            .parse()
            .with_context(|| "Invalid MCP_MAX_FILE_SIZE_MB value")?;
        let max_file_size_bytes = max_file_size_mb * 1024 * 1024;

        let model_dir = std::env::var("MCP_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./models"));
        let model_dir = if model_dir.exists() {
            model_dir.canonicalize().with_context(|| {
                format!("Failed to canonicalize model_dir: {}", model_dir.display())
            })?
        } else {
            model_dir
        };

        let log_level = std::env::var("MCP_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());

        Ok(Self {
            work_dir,
            max_file_size_bytes,
            model_dir,
            log_level,
        })
    }

    /// Validate that the given path is within the allowed work directory.
    pub fn validate_file_path(&self, file_path: &Path) -> Result<PathBuf> {
        let canonical = file_path
            .canonicalize()
            .with_context(|| format!("File not found: {}", file_path.display()))?;

        if !canonical.starts_with(&self.work_dir) {
            anyhow::bail!(
                "Access denied: {} is outside the allowed work directory {}",
                canonical.display(),
                self.work_dir.display()
            );
        }

        Ok(canonical)
    }

    /// Validate the file size against the configured maximum.
    pub fn validate_file_size(&self, file_path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(file_path)
            .with_context(|| format!("Cannot read file metadata: {}", file_path.display()))?;

        if metadata.len() > self.max_file_size_bytes {
            anyhow::bail!(
                "File too large: {} bytes (max: {} bytes / {} MB)",
                metadata.len(),
                self.max_file_size_bytes,
                self.max_file_size_bytes / (1024 * 1024)
            );
        }

        Ok(())
    }
}
