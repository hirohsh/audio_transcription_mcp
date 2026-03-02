use crate::mel::{self, N_FRAMES};
use crate::tokenizer::{WhisperTokenizer, EOT_TOKEN};
use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::{Tensor, TensorRef};
use std::path::Path;
use std::sync::Mutex;
use tracing::info;

const WHISPER_N_MELS: usize = 80;
const MAX_DECODE_STEPS: usize = 448;

pub struct WhisperModel {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: WhisperTokenizer,
    mel_filters: Vec<f32>,
}

impl WhisperModel {
    pub fn new(model_dir: &Path) -> Result<Self> {
        Self::init_onnx_runtime()?;

        let encoder_path = model_dir.join("encoder.onnx");
        let decoder_path = model_dir.join("decoder.onnx");
        let mel_filters_path = model_dir.join("mel_filters.json");

        info!("Loading encoder from: {}", encoder_path.display());
        let encoder = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&encoder_path)
            .with_context(|| format!("Failed to load encoder: {}", encoder_path.display()))?;

        info!("Loading decoder from: {}", decoder_path.display());
        let decoder = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&decoder_path)
            .with_context(|| format!("Failed to load decoder: {}", decoder_path.display()))?;

        let tokenizer = WhisperTokenizer::new(model_dir)?;

        let mel_filters = load_mel_filters(&mel_filters_path)?;
        info!(
            "Models loaded successfully. Mel filters: {} values",
            mel_filters.len()
        );

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            tokenizer,
            mel_filters,
        })
    }

    fn init_onnx_runtime() -> Result<()> {
        info!("Initializing ONNX Runtime...");

        // Try GPU execution providers first, fall back to CPU
        #[cfg(target_os = "macos")]
        {
            info!("macOS detected: trying CoreML execution provider");
            let ok = ort::init()
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ])
                .commit();
            if ok {
                info!("ONNX Runtime initialized with CoreML (GPU)");
                return Ok(());
            } else {
                info!("CoreML not available, falling back to CPU");
            }
        }

        #[cfg(target_os = "windows")]
        {
            info!("Windows detected: trying DirectML execution provider");
            let ok = ort::init()
                .with_execution_providers([
                    ort::execution_providers::DirectMLExecutionProvider::default().build(),
                ])
                .commit();
            if ok {
                info!("ONNX Runtime initialized with DirectML (GPU)");
                return Ok(());
            } else {
                info!("DirectML not available, falling back to CPU");
            }
        }

        #[cfg(target_os = "linux")]
        {
            info!("Linux detected: trying CUDA execution provider");
            let ok = ort::init()
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .commit();
            if ok {
                info!("ONNX Runtime initialized with CUDA (GPU)");
                return Ok(());
            } else {
                info!("CUDA not available, falling back to CPU");
            }
        }

        // CPU fallback
        ort::init().commit();
        info!("ONNX Runtime initialized with CPU");
        Ok(())
    }

    /// Transcribe audio samples (16kHz mono f32).
    pub fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<String> {
        let chunk_samples = mel::N_FRAMES * 160; // 30 seconds worth of samples
        let mut full_text = String::new();

        let chunks: Vec<&[f32]> = if samples.len() <= chunk_samples {
            vec![samples]
        } else {
            samples.chunks(chunk_samples).collect()
        };

        info!("Processing {} audio chunk(s)", chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            info!("Processing chunk {}/{}", i + 1, chunks.len());
            let text = self.transcribe_chunk(chunk, language)?;
            if !text.is_empty() {
                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(&text);
            }
        }

        Ok(full_text)
    }

    fn transcribe_chunk(&self, samples: &[f32], language: Option<&str>) -> Result<String> {
        // Compute mel spectrogram
        let mel = mel::log_mel_spectrogram(samples, &self.mel_filters)?;
        let mel_padded = mel::pad_or_trim_mel(&mel, WHISPER_N_MELS);

        // Run encoder: input shape [1, n_mels, n_frames]
        let mel_tensor = Tensor::from_array((
            [1_usize, WHISPER_N_MELS, N_FRAMES],
            mel_padded.into_boxed_slice(),
        ))?;

        // Run encoder and extract hidden states within the lock scope
        let (enc_shape_vec, enc_data_vec) = {
            let mut encoder = self
                .encoder
                .lock()
                .map_err(|e| anyhow::anyhow!("Encoder lock poisoned: {}", e))?;
            let encoder_outputs = encoder.run(ort::inputs![mel_tensor])?;
            let (enc_shape, enc_data) = encoder_outputs[0]
                .try_extract_tensor::<f32>()
                .with_context(|| "Failed to extract encoder output")?;
            let shape: Vec<usize> = enc_shape.iter().copied().map(|d| d as usize).collect();
            let data: Vec<f32> = enc_data.to_vec();
            (shape, data)
        };

        // Autoregressive decoding
        let prompt_tokens = self.tokenizer.build_prompt(language);
        let mut token_ids: Vec<i64> = prompt_tokens;
        let mut output_tokens: Vec<u32> = Vec::new();

        for _step in 0..MAX_DECODE_STEPS {
            let n_tokens = token_ids.len();
            let tokens_tensor = Tensor::from_array((
                [1_usize, n_tokens],
                token_ids.clone().into_boxed_slice(),
            ))?;

            // Run decoder and extract logits within the lock scope
            let (_vocab_size, _last_pos, best_token) = {
                let enc_tensor = TensorRef::from_array_view((
                    enc_shape_vec.as_slice(),
                    enc_data_vec.as_slice(),
                ))?;

                let mut decoder = self
                    .decoder
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Decoder lock poisoned: {}", e))?;
                let decoder_outputs =
                    decoder.run(ort::inputs![tokens_tensor, enc_tensor])?;

                let (logits_shape, logits_data) = decoder_outputs[0]
                    .try_extract_tensor::<f32>()
                    .with_context(|| "Failed to extract decoder logits")?;

                // logits_shape: [1, n_tokens, vocab_size]
                let vocab_size = logits_shape[2] as usize;
                let last_pos = logits_shape[1] as usize - 1;

                // Greedy: pick the token with highest logit
                let mut best_token = 0u32;
                let mut best_logit = f32::NEG_INFINITY;
                let offset = last_pos * vocab_size;
                for v in 0..vocab_size {
                    let logit = logits_data[offset + v];
                    if logit > best_logit {
                        best_logit = logit;
                        best_token = v as u32;
                    }
                }
                (vocab_size, last_pos, best_token)
            };

            if best_token == EOT_TOKEN {
                break;
            }

            output_tokens.push(best_token);
            token_ids.push(best_token as i64);
        }

        self.tokenizer.decode(&output_tokens)
    }
}

fn load_mel_filters(path: &Path) -> Result<Vec<f32>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read mel filters: {}", path.display()))?;
    let filters: Vec<f32> =
        serde_json::from_str(&content).with_context(|| "Invalid mel_filters.json format")?;
    Ok(filters)
}
