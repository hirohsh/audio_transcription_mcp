use anyhow::Result;

const WHISPER_SAMPLE_RATE: usize = 16000;
const WHISPER_N_FFT: usize = 400;
const WHISPER_HOP_LENGTH: usize = 160;
const WHISPER_CHUNK_LENGTH: usize = 30; // seconds

/// Number of frames in a 30-second chunk
pub const N_FRAMES: usize = WHISPER_CHUNK_LENGTH * WHISPER_SAMPLE_RATE / WHISPER_HOP_LENGTH;

/// Compute log-mel spectrogram from 16kHz mono audio samples.
/// Returns a vector of shape [n_mels, n_frames].
pub fn log_mel_spectrogram(samples: &[f32], mel_filters: &[f32], n_mels: usize) -> Result<Vec<f32>> {
    // Whisper center-pads the audio with N_FFT//2 zeros on each side
    let pad = WHISPER_N_FFT / 2;
    let mut padded = vec![0.0f32; pad + samples.len() + pad];
    padded[pad..pad + samples.len()].copy_from_slice(samples);

    let n_padded = padded.len();
    let n_frames = 1 + (n_padded.saturating_sub(WHISPER_N_FFT)) / WHISPER_HOP_LENGTH;

    // Compute STFT magnitudes
    let mut magnitudes = vec![0.0f32; (WHISPER_N_FFT / 2 + 1) * n_frames];
    let hann_window = hann(WHISPER_N_FFT);

    for frame_idx in 0..n_frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;
        let mut windowed = vec![0.0f32; WHISPER_N_FFT];

        for i in 0..WHISPER_N_FFT {
            let sample_idx = start + i;
            if sample_idx < n_padded {
                windowed[i] = padded[sample_idx] * hann_window[i];
            }
        }

        // Real FFT
        let spectrum = rfft(&windowed);
        let n_freq = WHISPER_N_FFT / 2 + 1;
        for i in 0..n_freq {
            let (re, im) = spectrum[i];
            magnitudes[i * n_frames + frame_idx] = re * re + im * im;
        }
    }

    // Apply mel filterbank: mel_filters is [n_mels, n_freq]
    let n_freq = WHISPER_N_FFT / 2 + 1;
    let mut mel_spec = vec![0.0f32; n_mels * n_frames];

    for mel in 0..n_mels {
        for frame in 0..n_frames {
            let mut sum = 0.0f32;
            for freq in 0..n_freq {
                sum += mel_filters[mel * n_freq + freq] * magnitudes[freq * n_frames + frame];
            }
            mel_spec[mel * n_frames + frame] = sum;
        }
    }

    // Log10 scale with clamping (matches Whisper's Python implementation)
    for val in mel_spec.iter_mut() {
        *val = (*val).max(1e-10).log10();
    }

    let max_log = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for val in mel_spec.iter_mut() {
        *val = (*val).max(max_log - 8.0);
        *val = (*val + 4.0) / 4.0;
    }

    Ok(mel_spec)
}

/// Pad or truncate mel spectrogram to exactly N_FRAMES frames.
pub fn pad_or_trim_mel(mel: &[f32], n_mels: usize) -> Vec<f32> {
    let current_frames = mel.len() / n_mels;
    let mut result = vec![0.0f32; n_mels * N_FRAMES];

    let copy_frames = current_frames.min(N_FRAMES);
    for m in 0..n_mels {
        for f in 0..copy_frames {
            result[m * N_FRAMES + f] = mel[m * current_frames + f];
        }
    }

    result
}

/// Hann window function
fn hann(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())
        })
        .collect()
}

/// Discrete Fourier Transform for real input → complex output (first N/2+1 bins).
/// Uses direct DFT to correctly handle non-power-of-2 sizes (e.g., N_FFT=400).
fn rfft(input: &[f32]) -> Vec<(f32, f32)> {
    let n = input.len();
    let n_out = n / 2 + 1;
    let mut result = vec![(0.0f32, 0.0f32); n_out];

    for k in 0..n_out {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for t in 0..n {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
            re += input[t] * angle.cos();
            im += input[t] * angle.sin();
        }
        result[k] = (re, im);
    }

    result
}
