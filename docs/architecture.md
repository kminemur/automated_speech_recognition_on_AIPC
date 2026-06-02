# Realtime ASR App Architecture

## 1. File layout

```text
app.py
  Entry point
  Parses CLI args
  Switches between CLI and GUI

model_manager.py
  Validates local IR directories
  Downloads pre-converted Hugging Face OpenVINO IR repositories
  Exports Hugging Face Whisper models to OpenVINO IR when needed
  Distinguishes required IR files from optional decoder_with_past files
  Refreshes cached model directories when they are incomplete

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

pyproject.toml
  Defines Python version and runtime dependencies for uv sync
```

## 2. Startup flow

### 2.1 Setup

`uv sync`:

1. Reads `pyproject.toml`.
2. Resolves Python 3.12+.
3. Creates `.venv` when missing.
4. Installs runtime dependencies.
5. Writes or updates `uv.lock`.

Model download or export is handled by application startup when a model is requested.

Defaults:

- `SETUP_MODEL=OpenVINO/whisper-large-v3-turbo-int8-ov`
- `SETUP_MODEL_CACHE_DIR=.cache_whisper`
- `SETUP_WEIGHT_FORMAT=int8`

### 2.2 Run

Run commands use `uv run`, which executes inside the synced environment:

```powershell
uv run python app.py
```

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
2. downloads a Hugging Face OpenVINO IR repository if `--model-id` points to one
3. exports a Hugging Face model if the repository does not already contain OpenVINO IR files
4. otherwise uses the default model ID `OpenVINO/whisper-large-v3-turbo-int8-ov`
5. validates required IR files
6. warns when optional `decoder_with_past` files are missing
7. validates `generation_config.json` and `lang_to_id`

`model_manager.resolve_hf_model()`:

1. derives the output directory from the cache root and model ID
2. accepts an already valid model directory
3. downloads the repository snapshot when the repo already has the required OpenVINO IR files
4. exports with `python -m optimum.commands.optimum_cli export openvino` when the repo is not pre-converted
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
4. copies callback frames into an in-memory input queue
5. converts queued frames to PCM16 for WebRTC VAD outside the audio callback
6. accumulates speech frames and trailing silence in memory
7. pushes completed segments onto a queue
8. transcribes queued segments outside the callback thread
9. calls `pipeline.generate(audio.tolist(), language=..., task=...)`
10. pushes recognized text to the result queue and callbacks

The audio callback must stay lightweight: it copies microphone frames into memory and returns quickly. VAD, segmentation, and inference run outside the callback.

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

- missing Python 3.12+ is a setup error
- missing or unsynced `.venv` is resolved by `uv sync` or `uv run`
- missing required IR files is a model configuration error
- missing optional IR files is a warning
- language mismatch in `generation_config.json` is a startup/configuration error, not a per-segment transcription retry case
- Windows console encoding issues during export must be handled in the subprocess environment, not ignored in the docs
