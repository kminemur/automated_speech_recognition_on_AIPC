# Realtime ASR App Requirements

## 1. Goal

Build a Windows realtime speech recognition app using OpenVINO 2026.0 and `openvino-genai` `WhisperPipeline`.
The app must support both CLI and PyQt6 GUI operation.

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
2. Create `.venv` if it does not exist.
3. Install `requirements.txt`.
4. Prepare a default model.

Environment variables:

- `SETUP_MODEL` default: `openai/whisper-tiny`
- `SETUP_MODEL_CACHE_DIR` default: `.cache_whisper`
- `SETUP_WEIGHT_FORMAT` default: `int8`

### 3.2 Model handling

The app must support two model inputs:

- Local OpenVINO Whisper IR directory
- Hugging Face Whisper model ID

When a model ID is provided, the app must export it with:

- task: `automatic-speech-recognition-with-past`
- cache dir: `--model-cache-dir` or `SETUP_MODEL_CACHE_DIR`
- weight format: `--weight-format` or `SETUP_WEIGHT_FORMAT`

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

If optional files are missing, the app should log that clearly and continue.

### 3.3 Realtime transcription

The app must:

- capture microphone audio using `sounddevice.InputStream`
- use mono 16 kHz audio
- use WebRTC VAD to segment speech
- send completed segments to `WhisperPipeline.generate()`
- display transcripts in CLI and GUI

### 3.4 Language token handling

- CLI `--language` default must be `"<|ja|>"`.
- The implementation must normalize shorthand values such as `ja` or `en` to the actual token expected by the model.
- Accepted language values are defined by `generation_config.json` `lang_to_id`.
- Before the first transcription call, the app must validate that the resolved language exists in `lang_to_id`.
- If the requested language is not present, the application must fail fast with a clear configuration error.

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

`--gui` and `--cli` cannot be used together.

### 3.6 GUI

The GUI must provide:

- model/device context display
- microphone selection
- `Start`, `Stop`, `Clear`
- status display
- transcript display
- log display

## 4. Non-functional requirements

- Use a local-first Windows app architecture.
- Setup and run must work via `setup.bat` and `run.bat`.
- Missing model files must be reported clearly.
- Python-side failures must be surfaced as readable messages.

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
