# Realtime ASR App Architecture

## 1. File layout

```text
app.py
  Entry point
  Parses CLI args
  Switches between CLI and GUI

model_manager.py
  Validates local IR directories
  Exports Hugging Face Whisper models to OpenVINO IR
  Distinguishes required IR files from optional decoder_with_past files

asr_engine.py
  Shared realtime ASR engine
  Captures microphone audio
  Applies WebRTC VAD
  Resolves Whisper language tokens from generation_config.json
  Runs WhisperPipeline inference

realtime_asr.py
  CLI runner
  Prints status and transcripts

asr_gui.py
  PyQt6 GUI
  Runs the engine on a worker thread
  Displays status, logs, and transcripts

setup.bat
  Finds Python 3.10+
  Creates .venv
  Installs dependencies
  Prepares default model

run.bat
  Activates .venv
  Runs app.py
```

## 2. Startup flow

### 2.1 Setup

`setup.bat`:

1. Finds Python 3.10+.
2. Creates `.venv`.
3. Installs dependencies.
4. Calls `ModelManager.export_hf_model()` for the default model.

Defaults:

- `SETUP_MODEL=openai/whisper-tiny`
- `SETUP_MODEL_CACHE_DIR=.cache_whisper`
- `SETUP_WEIGHT_FORMAT=int8`

### 2.2 Run

`run.bat` activates `.venv` and runs `python app.py %*`.

## 3. App flow

### 3.1 Entry point

`app.py`:

- parses CLI args
- handles `--list-mics`
- resolves the model through `ModelManager`
- starts CLI or GUI mode

Language argument policy:

- `app.py` defaults `--language` to `"<|ja|>"`.
- `app.py` passes the raw language argument to `ASREngine`.

### 3.2 Model resolution

`model_manager.ensure_model_available()`:

1. uses a local IR directory if `--model` is provided
2. exports a Hugging Face model if `--model-id` is provided
3. otherwise searches default cache locations
4. validates required IR files
5. warns when optional `decoder_with_past` files are missing

### 3.3 Language resolution

`ASREngine`:

1. loads `generation_config.json` from the selected model directory
2. reads the `lang_to_id` map
3. accepts direct keys such as `"<|ja|>"`
4. normalizes shorthand values such as `ja` to a real `lang_to_id` key
5. raises a clear configuration error if resolution fails

`WhisperPipeline.generate()` must be called only with the resolved language token.

## 4. Audio pipeline

`ASREngine`:

1. initializes `WhisperPipeline`
2. opens `sounddevice.InputStream`
3. receives mono 16 kHz audio
4. applies WebRTC VAD on PCM16 frames
5. accumulates speech segments
6. converts buffered audio to float32
7. calls `pipeline.generate(audio.tolist(), language=..., task=...)`
8. pushes recognized text to the result queue

## 5. CLI design

`realtime_asr.py`:

- starts the engine
- prints transcripts as they arrive
- stops cleanly on `SIGINT` and `SIGTERM`

Status values:

- `Loading model`
- `Listening`
- `Transcribing`
- `Error`
- `Stopped`

## 6. GUI design

`asr_gui.py` contains:

- `ASRWorker`
- `MainWindow`
- `ASRGUI`

The worker owns the engine and emits status, log, and transcript signals.

## 7. Error handling policy

- missing Python 3.10+ is a setup error
- missing `.venv` is a run error
- missing required IR files is a model configuration error
- missing optional IR files is a warning
- language mismatch in `generation_config.json` is a startup/configuration error, not a per-segment transcription retry case
