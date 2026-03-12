# Docs

This directory contains the implementation-facing specification for the realtime ASR app.

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

## Important rules

- `setup.bat` must find Python in this order: `python`, `py -3`, `py`.
- The app supports either a local OpenVINO Whisper IR directory or a Hugging Face model ID.
- Exported models use the `automatic-speech-recognition-with-past` task.
- Required IR files are `encoder`, `decoder`, `tokenizer`, and `detokenizer`.
- `decoder_with_past` files are optional. Missing optional files should be logged clearly.

## Language handling

- Default `--language` is `"<|ja|>"`.
- The app may accept shorthand values such as `ja`, but must normalize them to a key that exists in `generation_config.json` `lang_to_id`.
- If the selected model does not expose the requested language, the app must stop with a clear configuration error before transcription starts.
