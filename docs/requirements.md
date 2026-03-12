# Realtime ASR App Requirements

## 1. Goal

Build a Windows realtime speech recognition app using OpenVINO 2026.0 and `openvino-genai` `WhisperPipeline`.
The app must support both CLI and PyQt6 GUI operation, and the documented flow must match the current code path closely enough that the app can be reimplemented without guessing.

## 2. Supported runtime

- OS: Windows
- Shell: PowerShell / `cmd`
- Python: 3.10+
- Runtime: OpenVINO 2026.0
- Audio input: microphone

## 3. Functional requirements

### 3.1 Setup

`setup.bat` must:

1. Find Python 3.10+ in this order: `python`, `py -3`, `py`.
2. Reject Python versions below 3.10.
3. Create `.venv` if it does not exist.
4. Activate `.venv`.
5. Install `requirements.txt`.
6. Export the default Whisper model into OpenVINO IR.

Environment variables:

- `SETUP_MODEL` default: `openai/whisper-tiny`
- `SETUP_MODEL_CACHE_DIR` default: `.cache_whisper`
- `SETUP_WEIGHT_FORMAT` default: `int8`

### 3.2 Model handling

The app must support two model inputs:

- Local OpenVINO Whisper IR directory via `--model`
- Hugging Face Whisper model ID via `--model-id`

If neither `--model` nor `--model-id` is provided, the app must use the default model ID `openai/whisper-tiny`.

When a model ID is provided, the app must export it with:

- command entrypoint: `python -m optimum.commands.optimum_cli export openvino`
- task: `automatic-speech-recognition-with-past`
- cache dir: `--model-cache-dir` or `SETUP_MODEL_CACHE_DIR`
- weight format: `--weight-format` or `SETUP_WEIGHT_FORMAT`
- output directory: `<cache_dir>/<model_id with '/' replaced by '--'>`

The export subprocess must set UTF-8 related environment variables on Windows:

- `PYTHONIOENCODING=utf-8`
- `PYTHONUTF8=1`
- `NO_COLOR=1`

This is required because third-party progress rendering can fail on `cp932` terminals during model export.

Required IR files:

- `openvino_encoder_model.xml`
- `openvino_encoder_model.bin`
- `openvino_decoder_model.xml`
- `openvino_decoder_model.bin`
- `openvino_tokenizer.xml`
- `openvino_detokenizer.xml`

Optional IR files:

- `openvino_decoder_with_past_model.xml`
- `openvino_decoder_with_past_model.bin`

Behavior rules:

- A local model directory passed by `--model` must be validated and must fail immediately if required files are missing.
- A cached default or exported Hugging Face model directory must not be accepted only because the directory exists.
- If the directory exists but required files are missing, the app must try to export again.
- If optional files are missing, the app should log that clearly and continue.
- `generation_config.json` must exist and must contain `lang_to_id`.

### 3.3 Realtime transcription

The app must:

- capture microphone audio using `sounddevice.InputStream`
- use mono 16 kHz audio
- use 30 ms frames for WebRTC VAD
- use WebRTC VAD to segment speech
- avoid running ASR inference directly inside the audio callback
- queue completed segments and transcribe them outside the callback
- send completed segments to `WhisperPipeline.generate()`
- display transcripts in CLI and GUI

### 3.4 Language token handling

- CLI `--language` default must be `"<|ja|>"`.
- The implementation must normalize shorthand values such as `ja` or `en` to the actual token expected by the model.
- Accepted language values are defined by `generation_config.json` `lang_to_id`.
- Before the first transcription call, the application must validate that the resolved language exists in `lang_to_id`.
- If the requested language is not present, the application must fail fast with a clear configuration error.
- `WhisperPipeline.generate()` must be called only with the resolved token, not the raw user input.

### 3.5 CLI

CLI options must include:

- `--list-mics`
- `--device` with `AUTO`, `CPU`, `GPU`, `NPU`
- `--model`
- `--model-id`
- `--language`
- `--task`
- `--model-cache-dir`
- `--weight-format`
- `--mic`
- `--gui`
- `--cli`

Behavior rules:

- `--gui` and `--cli` cannot be used together.
- If neither is specified, the app may default to CLI mode.
- `--list-mics` must not require model export or model loading.

### 3.6 GUI

The GUI must provide:

- model/device context display
- microphone selection
- `Start`, `Stop`, `Clear`
- status display
- transcript display
- log display

The GUI worker may own the engine and forward status, log, and transcript events through Qt signals.

## 4. Non-functional requirements

- Use a local-first Windows app architecture.
- Setup and run must work via `setup.bat` and `run.bat`.
- Missing model files must be reported clearly.
- Python-side failures must be surfaced as readable messages.
- The documentation must prefer exact current behavior over vague future intentions.

## 5. Dependencies

- `numpy`
- `openvino`
- `openvino-tokenizers`
- `openvino-genai`
- `optimum-intel[openvino]`
- `huggingface_hub`
- `PyQt6`
- `sounddevice`
- `webrtcvad-wheels`

## 6. Known limitations

- VAD-based segmentation can miss very short utterances.
- Transcript export formats such as SRT/TXT are not implemented.
- GUI settings are intentionally minimal.
- First-time model export can take time and may produce non-fatal tracer warnings from upstream libraries.
