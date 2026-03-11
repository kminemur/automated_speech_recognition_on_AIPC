# OpenVINO 2026.0 Realtime ASR App

`prompt.txt` の要件に合わせて、PyQt6 GUI と CLI の両方を持つリアルタイム音声認識アプリを実装しています。認識エンジンは `openvino-genai` の `WhisperPipeline` を使い、入力区間の切り出しには WebRTC VAD を使います。ローカルの OpenVINO IR ディレクトリを直接指定できるほか、Hugging Face の Whisper モデル ID を指定すると初回起動時に自動ダウンロードと OpenVINO IR 変換を行います。

## Requirements

- Windows / PowerShell
- Python 3.10 以上
- OpenVINO 2026.0 が動作する CPU / GPU / NPU
- OpenVINO 形式の Whisper モデルディレクトリ、または Hugging Face の Whisper モデル ID
  - ローカル IR 例: `whisper-kotoba-ov`
  - ローカル IR 例: `ir_kotoba`
  - 自動変換例: `openai/whisper-tiny`
  - 自動変換例: `openai/whisper-small`

## Setup

```powershell
.\setup.bat
```

`setup.bat` は次を順に実行します。

- 利用可能な Python を自動検出して `.venv` を作成
- `requirements.txt` の依存導入
- 既定モデル `openai/whisper-tiny` のダウンロード
- `optimum-cli export openvino` による OpenVINO IR 変換

Python の選択順は `py -3.10` → `py -3` → `python` です。つまり、Python 3.10 固定ではなく、利用可能な Python 3 系でセットアップできます。

別モデルを準備したい場合は環境変数で上書きできます。

```powershell
$env:SETUP_MODEL="openai/whisper-small"
$env:SETUP_MODEL_CACHE_DIR=".cache_whisper"
$env:SETUP_WEIGHT_FORMAT="int8"
.\setup.bat
```

補足:
- 初回セットアップは `torch` なども入るため時間がかかります
- 変換済みモデルは既定で `.cache_whisper\models\...` に保存されます
- 2 回目以降は変換済みキャッシュがあれば再利用します

## Run

GUI:

```powershell
.\run.bat
```

CLI:

```powershell
.\run.bat --cli
```

利用可能なマイク一覧:

```powershell
.\.venv\Scripts\python.exe app.py --list-mics
```

## Main options

```powershell
.\.venv\Scripts\python.exe app.py --model openai/whisper-tiny --device AUTO --chunk-seconds 1.0
```

- `--model` / `--model-id`: OpenVINO Whisper モデルディレクトリ、または Hugging Face モデル ID
- `--device`: `CPU`, `GPU`, `NPU`, `AUTO`
- `--cli`: GUI ではなくコンソールで実行
- `--sample-rate`: マイク入力サンプルレート
- `--chunk-seconds`: 1 セグメントの最大長
- `--language`: 既定値は `"<|ja|>"`
- `--task`: `transcribe` または `translate`
- `--model-cache-dir`: 自動ダウンロード/変換済みモデルの保存先
- `--weight-format`: 自動変換時の精度。`fp16` または `int8`

## Files

- `asr_engine.py`: 共通のリアルタイム認識エンジン
- `asr_gui.py`: PyQt6 GUI
- `app.py`: GUI/CLI の起動エントリーポイント
- `realtime_asr.py`: 互換用 CLI ラッパー
