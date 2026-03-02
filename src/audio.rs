use anyhow::{Context, Result};
use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const TARGET_SAMPLE_RATE: u32 = 16000;

/// Decode an audio file and return mono f32 samples at 16kHz.
pub fn load_audio(file_path: &Path) -> Result<Vec<f32>> {
    let file =
        std::fs::File::open(file_path).with_context(|| format!("Cannot open audio file: {}", file_path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| "Unsupported audio format")?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .with_context(|| "No audio track found")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .with_context(|| "Unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| "Unsupported codec")?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;

        let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Convert to mono by averaging channels
        if channels > 1 {
            for chunk in samples.chunks(channels) {
                let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                all_samples.push(mono);
            }
        } else {
            all_samples.extend_from_slice(samples);
        }
    }

    // Resample to 16kHz if needed
    if sample_rate != TARGET_SAMPLE_RATE {
        all_samples = resample(&all_samples, sample_rate, TARGET_SAMPLE_RATE)?;
    }

    Ok(all_samples)
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let chunk_size = 1024;

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .with_context(|| "Failed to create resampler")?;

    let mut output = Vec::new();

    for chunk in samples.chunks(chunk_size) {
        let input = if chunk.len() < chunk_size {
            let mut padded = chunk.to_vec();
            padded.resize(chunk_size, 0.0);
            vec![padded]
        } else {
            vec![chunk.to_vec()]
        };

        let resampled = resampler
            .process(&input, None)
            .with_context(|| "Resampling failed")?;

        if let Some(channel) = resampled.first() {
            output.extend_from_slice(channel);
        }
    }

    // Trim output to expected length
    let expected_len = (samples.len() as f64 * ratio).ceil() as usize;
    output.truncate(expected_len);

    Ok(output)
}
