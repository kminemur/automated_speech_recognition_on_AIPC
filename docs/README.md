# Docs

This directory is the implementation contract for the current Windows realtime ASR app.
The goal is not a broad product spec. The goal is to describe the exact behavior the codebase should implement so the app can be rebuilt reliably from these docs.

## Covered files

- `setup.bat`
- `run.bat`
- `app.py`
- `model_manager.py`
- `asr_engine.py`
- `realtime_asr.py`
- `asr_gui.py`

## Main documents

1. `requirements.md`
2. `architecture.md`

## Runtime assumptions

- OS: Windows
- Shell: PowerShell or `cmd`
- Python: 3.10+
- Runtime: OpenVINO 2026.0
- ASR backend: `openvino-genai` `WhisperPipeline`
- Model export tool: `python -m optimum.commands.optimum_cli export openvino`

## Implementation-critical rules

- `setup.bat` must probe Python in this order: `python`, `py -3`, `py`.
- The app must support either a local OpenVINO Whisper IR directory or a Hugging Face model ID.
- If no model is given, the app must prepare the default model `openai/whisper-tiny` under `.cache_whisper`.
- A partially created default model directory must not be treated as valid. The app must try export again until required IR files exist.
- Hugging Face export must use task `automatic-speech-recognition-with-past`.
- Export must call `optimum.commands.optimum_cli`, not `optimum.exporters.openvino` directly.
- Export subprocesses must run with UTF-8 stdio settings on Windows to avoid `cp932` failures from third-party progress output.
- Required IR files are `encoder`, `decoder`, `tokenizer`, and `detokenizer`.
- `decoder_with_past` files are optional. Missing optional files must be logged as warnings, not fatal errors.

## Language handling

- Default `--language` is `"<|ja|>"`.
- The app may accept shorthand values such as `ja`, but must normalize them to a key that exists in `generation_config.json` `lang_to_id`.
- If the selected model does not expose the requested language, the app must stop with a clear configuration error before transcription starts.

## Intent of these docs

When implementation and docs disagree, update the docs only if the current code path is intentional and validated.
Do not leave ambiguous statements such as "search default cache locations" when the code actually exports a known default model.
