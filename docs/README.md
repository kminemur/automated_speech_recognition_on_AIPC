# docs

このフォルダには、OpenVINO 2026.0 と `openvino-genai` の `WhisperPipeline` を使った
リアルタイム音声認識アプリの設計資料を置く。

読む順番:

1. `requirements.md`
2. `architecture.md`

重要:

- このアプリは `WhisperPipeline` を使うため、Whisper の OpenVINO IR は
  `automatic-speech-recognition-with-past` で変換したものだけを使う。
- `automatic-speech-recognition` で変換した IR は使わない。
- 間違った IR を使うと、推論時に `beam_idx was not found` が発生する。

最低限の確認項目:

- モデルディレクトリに `openvino_decoder_with_past_model.xml` がある
- モデルディレクトリに `openvino_decoder_with_past_model.bin` がある
- `setup.bat` が `--task automatic-speech-recognition-with-past` を使っている

トラブル時:

- 既存キャッシュを削除して再エクスポートする
- ローカル IR を `--model` で直接渡している場合は、その IR 自体を作り直す
