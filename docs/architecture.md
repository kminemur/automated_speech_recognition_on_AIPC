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
  Retries export when cached model directories are incomplete

asr_engine.py
  Shared realtime ASR engine
  Captures microphone audio
  Applies WebRTC VAD
  Resolves Whisper language tokens from generation_config.json
  Queues finished speech segments
  Runs WhisperPipeline inference outside the audio callback

realtime_asr.py
  CLI runner
  Prints transcripts
  Stops on signals or engine shutdown

asr_gui.py
  PyQt6 GUI
  Runs the engine through a worker object
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

1. Probes `python`, then `py -3`, then `py`.
2. Verifies Python 3.10+.
3. Creates `.venv` when missing.
4. Activates `.venv`.
5. Installs `requirements.txt`.
6. Calls `ModelManager.export_hf_model()` for the default model.

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
- rejects `--gui` and `--cli` together
- handles `--list-mics` without loading a model
- resolves the model through `ModelManager`
- builds `EngineConfig`
- starts GUI mode when `--gui` is set
- otherwise starts CLI mode

Language argument policy:

- `app.py` defaults `--language` to `"<|ja|>"`.
- `app.py` passes the raw language argument into `ASREngine`.
- CLI startup validates language before beginning the run loop.

### 3.2 Model resolution

`model_manager.ensure_model_available()`:

1. uses a local IR directory if `--model` is provided
2. exports a Hugging Face model if `--model-id` is provided
3. otherwise exports the default model ID `openai/whisper-tiny`
4. validates required IR files
5. warns when optional `decoder_with_past` files are missing
6. validates `generation_config.json` and `lang_to_id`

`model_manager.export_hf_model()`:

1. derives the output directory from the cache root and model ID
2. accepts an already valid export directory
3. re-exports if the directory exists but is incomplete
4. runs `python -m optimum.commands.optimum_cli export openvino`
5. forces UTF-8 stdio environment variables for the export subprocess on Windows
6. validates the resulting IR directory before returning

### 3.3 Language resolution

`ASREngine`:

1. loads `generation_config.json` from the selected model directory
2. reads the `lang_to_id` map
3. accepts direct keys such as `"<|ja|>"`
4. normalizes shorthand values such as `ja`, `en`, and underscore variants
5. raises a clear configuration error if resolution fails

`WhisperPipeline.generate()` must be called only with the resolved language token.

## 4. Audio pipeline

`ASREngine`:

1. initializes `WhisperPipeline`
2. opens `sounddevice.InputStream`
3. receives mono 16 kHz audio in 30 ms frames
4. converts input frames to PCM16 for WebRTC VAD
5. accumulates speech frames and trailing silence
6. pushes completed segments onto a queue
7. transcribes queued segments outside the callback thread
8. calls `pipeline.generate(audio.tolist(), language=..., task=...)`
9. pushes recognized text to the result queue and callbacks

This separation between audio callback and inference is intentional and should be preserved.

## 5. CLI design

`realtime_asr.py`:

- starts the engine
- reads transcript messages from the engine queue
- prints transcripts to stdout
- stops cleanly on `SIGINT` and `SIGTERM`
- exits when the engine thread is no longer running

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
The main window is responsible for microphone selection, start/stop controls, and transcript/log display.

## 7. Error handling policy

- missing Python 3.10+ is a setup error
- missing `.venv` is a run error
- missing required IR files is a model configuration error
- missing optional IR files is a warning
- language mismatch in `generation_config.json` is a startup/configuration error, not a per-segment transcription retry case
- Windows console encoding issues during export must be handled in the subprocess environment, not ignored in the docs
