# OpenVINO 2026.0 Realtime ASR App

Windows realtime speech recognition app built on OpenVINO 2026.0 and `openvino-genai` `WhisperPipeline`.
It supports both CLI and PyQt6 GUI modes.

## Requirements

- Windows
- PowerShell or `cmd`
- Python 3.10+
- OpenVINO 2026.0

## Setup

```powershell
.\setup.bat
```

`setup.bat`:

- finds Python 3.10+
- creates `.venv`
- installs `requirements.txt`
- prepares the default model `openai/whisper-tiny` in `.cache_whisper`

Optional setup variables:

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

List microphones:

```powershell
.\run.bat --list-mics
```

GUI:

```powershell
.\run.bat --gui
```

## Main options

- `--model` / `--model-id`: local OpenVINO IR directory or Hugging Face model ID
- `--device`: `AUTO`, `CPU`, `GPU`, `NPU`
- `--gui`: run GUI mode
- `--cli`: run CLI mode
- `--language`: default `"<|ja|>"`
- `--task`: `transcribe` or `translate`
- `--model-cache-dir`: model cache directory
- `--weight-format`: `int8`, `fp16`, `fp32`
- `--mic`: microphone index

## Language handling

- Default `--language` is `"<|ja|>"`.
- `--language ja` is accepted, but it is normalized to the actual token defined by the selected model's `generation_config.json`.
- If the selected model does not contain the requested language in `lang_to_id`, the app stops with a clear configuration error.

## Model notes

Required IR files:

- `openvino_encoder_model.xml/.bin`
- `openvino_decoder_model.xml/.bin`
- `openvino_tokenizer.xml`
- `openvino_detokenizer.xml`

Optional IR files:

- `openvino_decoder_with_past_model.xml/.bin`

## Files

- `app.py`: entry point
- `model_manager.py`: model validation and export
- `asr_engine.py`: shared realtime ASR engine
- `realtime_asr.py`: CLI runner
- `asr_gui.py`: PyQt6 GUI
- `setup.bat`: setup script
- `run.bat`: launcher
