# Whisper Model Converter

OpenAI Whisper モデルを ONNX 形式に変換するツールです。

## 前提条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) パッケージマネージャー

## セットアップ

```bash
cd model_converter
uv sync
```

## 使い方

### 基本的な変換

```bash
uv run convert --model-size base --output-dir ../models
```

### オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--model-size` | モデルサイズ (tiny, base, small, medium, large) | `base` |
| `--output-dir` | 出力ディレクトリ | `../models` |
| `--quantize` | INT8量子化を適用 | `false` |

### モデルサイズ一覧

| サイズ | パラメータ数 | 必要メモリ | 精度 |
|--------|------------|-----------|------|
| tiny | 39M | ~150MB | ★☆☆☆☆ |
| base | 74M | ~300MB | ★★☆☆☆ |
| small | 244M | ~1GB | ★★★☆☆ |
| medium | 769M | ~3GB | ★★★★☆ |
| large | 1550M | ~6GB | ★★★★★ |

### 例

```bash
# tiny モデルを変換 (テスト用)
uv run convert --model-size tiny --output-dir ../models

# large モデルを量子化して変換
uv run convert --model-size large --output-dir ../models --quantize
```

## 出力ファイル

変換後、出力ディレクトリに以下のファイルが生成されます:

- `encoder.onnx` - エンコーダーモデル
- `decoder.onnx` - デコーダーモデル
- `tokenizer.json` - トークナイザー設定
- `mel_filters.json` - メルフィルターバンク
