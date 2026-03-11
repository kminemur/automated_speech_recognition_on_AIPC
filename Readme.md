# OpenVINO 2026.0 Realtime ASR App

`prompt.txt` の要件に合わせて、PyQt6 GUI と CLI の両方を持つリアルタイム音声認識アプリを実装しています。認識エンジンは `openvino-genai` の `WhisperPipeline` を使い、入力区間の切り出しには WebRTC VAD を使います。

## Requirements

- Windows / PowerShell
- Python 3.10 以上
- OpenVINO 2026.0 が動作する CPU / GPU / NPU
- OpenVINO 形式の Whisper モデルディレクトリ
  - 例: `whisper-tiny-ov`
  - 例: `whisper-kotoba-ov`
  - 例: `ir_kotoba`

## Setup

```powershell
.\setup.bat
```

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
python app.py --list-mics
```

## Main options

```powershell
python app.py --model .\whisper-kotoba-ov --device NPU --chunk-seconds 1.0
```

- `--model` / `--model-id`: OpenVINO Whisper モデルディレクトリ
- `--device`: `CPU`, `GPU`, `NPU`, `AUTO`
- `--cli`: GUI ではなくコンソールで実行
- `--sample-rate`: マイク入力サンプルレート
- `--chunk-seconds`: 1 セグメントの最大長
- `--language`: 既定値は `"<|ja|>"`
- `--task`: `transcribe` または `translate`

## Files

- `asr_engine.py`: 共通のリアルタイム認識エンジン
- `asr_gui.py`: PyQt6 GUI
- `app.py`: GUI/CLI の起動エントリーポイント
- `realtime_asr.py`: 互換用 CLI ラッパー
