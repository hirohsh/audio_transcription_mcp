"""Convert OpenAI Whisper models to ONNX format."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor


MODEL_NAMES = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}


def export_mel_filters(processor: WhisperProcessor, output_dir: Path) -> None:
    """Export mel filterbank from the Whisper feature extractor."""
    feature_extractor = processor.feature_extractor
    mel_filters = feature_extractor.mel_filters  # shape: (n_freq, n_mels)

    # Transpose to (n_mels, n_freq) for row-major flatten
    mel_filters = mel_filters.T
    mel_list = mel_filters.flatten().tolist()

    output_path = output_dir / "mel_filters.json"
    with open(output_path, "w") as f:
        json.dump(mel_list, f)
    print(f"  Mel filters saved: {output_path} ({len(mel_list)} values)")


def export_tokenizer(processor: WhisperProcessor, output_dir: Path) -> None:
    """Export the tokenizer to tokenizer.json."""
    tokenizer = processor.tokenizer
    tokenizer.save_pretrained(str(output_dir))

    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"  Tokenizer saved: {tokenizer_path}")
    else:
        print("  Warning: tokenizer.json was not created")


def convert_model(model_size: str, output_dir: Path, quantize: bool = False) -> None:
    """Convert a Whisper model to ONNX format."""
    model_name = MODEL_NAMES.get(model_size)
    if model_name is None:
        print(f"Error: Unknown model size '{model_size}'")
        print(f"Available sizes: {', '.join(MODEL_NAMES.keys())}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {model_name} to ONNX...")
    print(f"Output directory: {output_dir}")

    # Load processor for tokenizer and mel filters
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(model_name)

    # Export tokenizer and mel filters
    print("Exporting tokenizer and mel filters...")
    export_tokenizer(processor, output_dir)
    export_mel_filters(processor, output_dir)

    # Export model to ONNX using optimum
    print("Exporting model to ONNX (this may take a while)...")
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        export=True,
    )
    model.save_pretrained(str(output_dir))

    # Rename files to simpler names
    _rename_model_files(output_dir)

    if quantize:
        print("Applying INT8 quantization...")
        _quantize_models(output_dir)

    print("\nConversion complete!")
    print(f"Files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")


def _rename_model_files(output_dir: Path) -> None:
    """Rename optimum output files to simpler names."""
    renames = {
        "encoder_model.onnx": "encoder.onnx",
        "decoder_model.onnx": "decoder.onnx",
        "decoder_model_merged.onnx": "decoder.onnx",
        "decoder_with_past_model.onnx": None,  # Remove if separate decoder exists
    }

    for old_name, new_name in renames.items():
        old_path = output_dir / old_name
        if old_path.exists():
            if new_name is None:
                # Only remove if the merged decoder exists
                if (output_dir / "decoder.onnx").exists():
                    old_path.unlink()
                    print(f"  Removed: {old_name}")
            else:
                new_path = output_dir / new_name
                if not new_path.exists():
                    old_path.rename(new_path)
                    print(f"  Renamed: {old_name} -> {new_name}")


def _quantize_models(output_dir: Path) -> None:
    """Apply INT8 dynamic quantization to ONNX models."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

    for model_file in ["encoder.onnx", "decoder.onnx"]:
        model_path = output_dir / model_file
        if model_path.exists():
            print(f"  Quantizing {model_file}...")
            quantizer = ORTQuantizer.from_pretrained(output_dir, file_name=model_file)
            quantized_name = model_file.replace(".onnx", "_quantized.onnx")
            quantizer.quantize(
                save_dir=output_dir,
                quantization_config=qconfig,
                file_suffix="quantized",
            )
            # Replace original with quantized version
            quantized_path = output_dir / quantized_name
            if quantized_path.exists():
                model_path.unlink()
                quantized_path.rename(model_path)
                print(f"  Replaced {model_file} with quantized version")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenAI Whisper models to ONNX format"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=list(MODEL_NAMES.keys()),
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../models",
        help="Output directory for ONNX files (default: ../models)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()

    convert_model(args.model_size, output_dir, args.quantize)


if __name__ == "__main__":
    main()
