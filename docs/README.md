# Docs

このディレクトリは、現在の realtime ASR アプリ実装に対応する仕様書です。

対象実装:

- `setup.bat`
- `run.bat`
- `app.py`
- `model_manager.py`
- `asr_engine.py`
- `realtime_asr.py`
- `asr_gui.py`

読む順序:

1. `requirements.md`
2. `architecture.md`

前提:

- Windows
- PowerShell または cmd
- Python 3.10 以上
- OpenVINO 2026.0 系
- `openvino-genai` の `WhisperPipeline`

実装上の重要事項:

- `setup.bat` は `python`、`py -3`、`py` の順に Python を検出する
- Python 3.10 未満は受け付けない
- モデルはローカル IR ディレクトリ、または Hugging Face の model ID を指定できる
- model ID の場合は `automatic-speech-recognition-with-past` で OpenVINO IR を自動生成する
- 現行アプリでは encoder / decoder / tokenizer / detokenizer が揃っていれば有効モデルとして扱う
- `openvino_decoder_with_past_model.xml` と `openvino_decoder_with_past_model.bin` は任意扱いで、不足時は警告ログのみ出す
