# Audio Transcription MCP Server

音声ファイルを文字起こしする MCP (Model Context Protocol) サーバーです。OpenAI Whisper モデルを ONNX 形式で使用し、Rust で実装されています。

## 特徴

- 🎙️ **多形式対応**: WAV, MP3, FLAC, OGG (Vorbis) をサポート
- 🌍 **多言語対応**: 99 言語の自動検出・指定が可能
- ⚡ **GPU アクセラレーション**: macOS (CoreML), Windows (DirectML), Linux (CUDA) に対応、CPU 自動フォールバック
- 🔒 **セキュリティ**: 作業ディレクトリ制限、ファイルサイズ制限
- 📦 **MCP 準拠**: Claude Desktop 等の MCP クライアントとシームレスに連携

## 前提条件

- [Rust](https://www.rust-lang.org/tools/install) 1.85+
- [uv](https://docs.astral.sh/uv/) (モデル変換用)
- [ONNX Runtime](https://onnxruntime.ai/) shared library
  - macOS: `brew install onnxruntime`
  - Windows: [公式リリース](https://github.com/microsoft/onnxruntime/releases) からダウンロード
  - Linux: `apt install libonnxruntime-dev` または公式リリースからダウンロード

## クイックスタート

### 1. モデルの変換

まず、Whisper モデルを ONNX 形式に変換します:

```bash
cd model_converter
uv sync
uv run convert --model-size base --output-dir ../models
```

利用可能なモデルサイズ:

| サイズ | パラメータ | 精度 | 速度 |
|--------|-----------|------|------|
| `tiny` | 39M | ★☆☆☆☆ | 最速 |
| `base` | 74M | ★★☆☆☆ | 速い |
| `small` | 244M | ★★★☆☆ | 普通 |
| `medium` | 769M | ★★★★☆ | 遅い |
| `large` | 1550M | ★★★★★ | 最遅 |

INT8 量子化でモデルサイズを削減:

```bash
uv run convert --model-size base --output-dir ../models --quantize
```

### 2. ビルド

```bash
# ONNX Runtime のパスを設定 (macOS の場合)
export ORT_DYLIB_PATH=$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib

# ビルド
cargo build --release
```

### 3. 環境設定

```bash
cp .env.example .env
```

`.env` ファイルを編集:

```env
# 作業ディレクトリ (このディレクトリ以下のファイルのみアクセス可能)
MCP_WORK_DIR=/path/to/your/audio/files

# 最大ファイルサイズ (MB)
MCP_MAX_FILE_SIZE_MB=25

# ONNX モデルディレクトリ
MCP_MODEL_DIR=./models

# ログレベル
MCP_LOG_LEVEL=info
```

### 4. Claude Desktop への登録

`claude_desktop_config.json` に以下を追加:

```json
{
  "mcpServers": {
    "audio-transcription": {
      "command": "/path/to/audio_transcription_mcp/target/release/audio_transcription_mcp",
      "env": {
        "MCP_WORK_DIR": "/path/to/your/audio/files",
        "MCP_MODEL_DIR": "/path/to/audio_transcription_mcp/models",
        "MCP_MAX_FILE_SIZE_MB": "25",
        "MCP_LOG_LEVEL": "info",
        "ORT_DYLIB_PATH": "/opt/homebrew/lib/libonnxruntime.dylib"
      }
    }
  }
}
```

**設定ファイルの場所:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

### 5. 動作確認

Claude Desktop を再起動すると、`transcribe_audio` ツールが利用可能になります。

## MCP ツール

### `transcribe_audio`

音声ファイルを文字起こしします。

**パラメータ:**

| パラメータ | 型 | 必須 | 説明 |
|-----------|------|------|------|
| `file_path` | string | ✅ | 音声ファイルのパス (作業ディレクトリ内) |
| `language` | string | ❌ | 言語コード (例: `ja`, `en`)。省略時は自動検出 |

**使用例 (Claude Desktop で):**

```
この音声ファイルを文字起こしして: meeting.wav
```

```
英語の音声を文字起こしして: interview.mp3
```

## 環境変数

| 変数名 | 説明 | デフォルト値 |
|--------|------|-------------|
| `MCP_WORK_DIR` | アクセス可能な作業ディレクトリ | カレントディレクトリ |
| `MCP_MAX_FILE_SIZE_MB` | 最大ファイルサイズ (MB) | `25` |
| `MCP_MODEL_DIR` | ONNX モデルディレクトリ | `./models` |
| `MCP_LOG_LEVEL` | ログレベル (`trace`/`debug`/`info`/`warn`/`error`) | `info` |
| `ORT_DYLIB_PATH` | ONNX Runtime 共有ライブラリのパス | 自動検出 |

## GPU サポート

本サーバーは GPU を優先して使用し、利用できない場合は CPU にフォールバックします。

| OS | GPU バックエンド | 必要なもの |
|----|----------------|-----------|
| macOS | CoreML | macOS 12+ (自動対応) |
| Windows | DirectML | DirectX 12 対応 GPU |
| Linux | CUDA | NVIDIA GPU + CUDA Toolkit |

GPU の使用状況はログで確認できます:

```
INFO ONNX Runtime initialized with CoreML (GPU)
```

または:

```
INFO CoreML not available, falling back to CPU: ...
INFO ONNX Runtime initialized with CPU
```

## トラブルシューティング

### ONNX Runtime が見つからない

```
error: Failed to load ONNX Runtime
```

→ `ORT_DYLIB_PATH` 環境変数を設定するか、ONNX Runtime をシステムにインストールしてください。

### モデルファイルが見つからない

```
error: Failed to load encoder: models/encoder.onnx
```

→ モデル変換ツールを実行してください: `cd model_converter && uv run convert --model-size base --output-dir ../models`

### ファイルアクセスが拒否された

```
error: Access denied: /path/to/file is outside the allowed work directory
```

→ `MCP_WORK_DIR` を確認し、音声ファイルがそのディレクトリ内にあることを確認してください。

### ファイルサイズ超過

```
error: File too large: XXX bytes (max: 26214400 bytes / 25 MB)
```

→ `MCP_MAX_FILE_SIZE_MB` の値を大きくするか、ファイルを分割してください。

## プロジェクト構成

```
audio_transcription_mcp/
├── Cargo.toml                    # Rust プロジェクト設定
├── README.md                     # このファイル
├── .env.example                  # 環境変数テンプレート
├── .gitignore
├── docs/
│   └── design.md                 # 設計書
├── model_converter/              # モデル変換ツール (Python)
│   ├── pyproject.toml
│   ├── README.md
│   └── src/
│       └── convert.py
├── models/                       # ONNX モデル格納先
│   ├── encoder.onnx
│   ├── decoder.onnx
│   ├── tokenizer.json
│   └── mel_filters.json
└── src/                          # Rust ソースコード
    ├── main.rs                   # エントリポイント
    ├── config.rs                 # 設定管理
    ├── audio.rs                  # 音声デコード
    ├── mel.rs                    # メルスペクトログラム
    ├── whisper.rs                # Whisper 推論エンジン
    ├── tokenizer.rs              # トークナイザー
    └── tools.rs                  # MCP ツール定義
```

## ライセンス

MIT
