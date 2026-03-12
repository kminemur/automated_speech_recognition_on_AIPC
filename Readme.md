# OpenVINO 2026.0 Realtime ASR App

OpenVINO 2026.0 と `openvino-genai` の `WhisperPipeline` を使った、Windows 向けのリアルタイム音声認識アプリです。CLI と PyQt6 GUI の両方を提供します。

マイク入力を `sounddevice` で取得し、WebRTC VAD で発話区間を切り出し、その区間だけを Whisper に渡して認識します。モデル指定はローカル OpenVINO IR ディレクトリか Hugging Face model ID のどちらでも受け付けます。

## Requirements

- Windows
- PowerShell または cmd
- Python 3.10 以上
- OpenVINO 2026.0 系

## Setup

```powershell
.\setup.bat
```

`setup.bat` は次を行います。

- `python`、`py -3`、`py` の順で Python 3.10+ を探す
- `.venv` を作成する
- `requirements.txt` をインストールする
- 既定モデル `openai/whisper-tiny` を `.cache_whisper` に準備する

別モデルを使う場合:

```powershell
$env:SETUP_MODEL="openai/whisper-small"
$env:SETUP_MODEL_CACHE_DIR=".cache_whisper"
$env:SETUP_WEIGHT_FORMAT="int8"
.\setup.bat
```

## Run

CLI:

```powershell
.\run.bat
```

マイク一覧:

```powershell
.\run.bat --list-mics
```

GUI:

```powershell
.\run.bat --gui
```

## Main options

```powershell
.\.venv\Scripts\python.exe app.py --model openai/whisper-tiny --device AUTO --chunk-seconds 1.0
```

- `--model` / `--model-id`: ローカル IR ディレクトリまたは Hugging Face model ID
- `--device`: `AUTO`, `CPU`, `GPU`, `NPU`
- `--gui`: GUI を起動
- `--cli`: CLI を明示
- `--sample-rate`: 既定 `16000`
- `--chunk-seconds`: 引数としては残っているが、現行実装では VAD ベースの区切りが中心
- `--language`: 既定 `"<|ja|>"`
- `--task`: `transcribe` または `translate`
- `--model-cache-dir`: 自動変換したモデルの保存先
- `--weight-format`: 例 `int8`, `fp16`
- `--mic`: マイク index

## Model notes

現行アプリは次の IR が揃っていれば有効モデルとして扱います。

- `openvino_encoder_model.xml/.bin`
- `openvino_decoder_model.xml/.bin`
- `openvino_tokenizer.xml`
- `openvino_detokenizer.xml`

`openvino_decoder_with_past_model.xml/.bin` は任意扱いです。不足時はログ警告を出しますが、`WhisperPipeline` の初期化を試みます。

## Files

- `app.py`: エントリポイント
- `model_manager.py`: モデル検証と自動変換
- `asr_engine.py`: 共通リアルタイム ASR エンジン
- `realtime_asr.py`: CLI ランナー
- `asr_gui.py`: PyQt6 GUI
- `setup.bat`: セットアップ
- `run.bat`: 実行ラッパー
