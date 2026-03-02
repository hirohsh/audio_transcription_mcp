use anyhow::Result;

const WHISPER_SAMPLE_RATE: usize = 16000;
const WHISPER_N_FFT: usize = 400;
const WHISPER_HOP_LENGTH: usize = 160;
const WHISPER_N_MELS: usize = 80;
const WHISPER_CHUNK_LENGTH: usize = 30; // seconds

/// Number of frames in a 30-second chunk
pub const N_FRAMES: usize = WHISPER_CHUNK_LENGTH * WHISPER_SAMPLE_RATE / WHISPER_HOP_LENGTH;

/// Compute log-mel spectrogram from 16kHz mono audio samples.
/// Returns a vector of shape [n_mels, n_frames].
pub fn log_mel_spectrogram(samples: &[f32], mel_filters: &[f32]) -> Result<Vec<f32>> {
    let n_samples = samples.len();
    let n_frames = 1 + (n_samples.saturating_sub(WHISPER_N_FFT)) / WHISPER_HOP_LENGTH;

    // Compute STFT magnitudes
    let mut magnitudes = vec![0.0f32; (WHISPER_N_FFT / 2 + 1) * n_frames];
    let hann_window = hann(WHISPER_N_FFT);

    for frame_idx in 0..n_frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;
        let mut windowed = vec![0.0f32; WHISPER_N_FFT];

        for i in 0..WHISPER_N_FFT {
            let sample_idx = start + i;
            if sample_idx < n_samples {
                windowed[i] = samples[sample_idx] * hann_window[i];
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
    let mut mel_spec = vec![0.0f32; WHISPER_N_MELS * n_frames];

    for mel in 0..WHISPER_N_MELS {
        for frame in 0..n_frames {
            let mut sum = 0.0f32;
            for freq in 0..n_freq {
                sum += mel_filters[mel * n_freq + freq] * magnitudes[freq * n_frames + frame];
            }
            mel_spec[mel * n_frames + frame] = sum;
        }
    }

    // Log scale with clamping
    let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_max = (max_val.max(1e-10)).ln();
    let clamp_min = log_max - 8.0; // ~dynamic range clamp

    for val in mel_spec.iter_mut() {
        *val = (*val).max(1e-10).ln().max(clamp_min);
    }

    // Normalize to roughly [-1, 1]
    let max_log = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_log = mel_spec.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = max_log - min_log;
    if range > 0.0 {
        for val in mel_spec.iter_mut() {
            *val = (*val - min_log) / range * 2.0 - 1.0;
        }
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

/// Simple radix-2 DIT FFT (real input → complex output)
fn rfft(input: &[f32]) -> Vec<(f32, f32)> {
    let n = input.len();
    let mut data: Vec<(f32, f32)> = input.iter().map(|&x| (x, 0.0)).collect();

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey FFT
    let mut step = 2;
    while step <= n {
        let half = step / 2;
        let angle = -2.0 * std::f32::consts::PI / step as f32;
        let wn = (angle.cos(), angle.sin());

        for k in (0..n).step_by(step) {
            let mut w = (1.0f32, 0.0f32);
            for j in 0..half {
                let u = data[k + j];
                let t = (
                    w.0 * data[k + j + half].0 - w.1 * data[k + j + half].1,
                    w.0 * data[k + j + half].1 + w.1 * data[k + j + half].0,
                );
                data[k + j] = (u.0 + t.0, u.1 + t.1);
                data[k + j + half] = (u.0 - t.0, u.1 - t.1);
                let new_w = (w.0 * wn.0 - w.1 * wn.1, w.0 * wn.1 + w.1 * wn.0);
                w = new_w;
            }
        }
        step <<= 1;
    }

    // Return first N/2+1 complex values
    data[..n / 2 + 1].to_vec()
}
