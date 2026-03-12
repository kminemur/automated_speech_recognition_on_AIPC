# Realtime ASR App Requirements

## 1. Goal

Windows 上で動作するリアルタイム音声認識アプリを提供する。  
OpenVINO 2026.0 と `openvino-genai` の `WhisperPipeline` を利用し、CLI と GUI の両方から使えること。

## 2. Supported runtime

- OS: Windows
- Shell: PowerShell / cmd
- Python: 3.10 以上
- Runtime: OpenVINO 2026.0 系
- Audio input: マイク入力

## 3. Functional requirements

### 3.1 Setup

`setup.bat` は次を行うこと。

1. 利用可能な Python 3.10+ を検出する
2. `.venv` がなければ仮想環境を作成する
3. `requirements.txt` の依存をインストールする
4. 既定モデルを準備する

Python 検出順序:

1. `python`
2. `py -3`
3. `py`

どれも Python 3.10+ でなければ、セットアップはエラー終了する。

### 3.2 Model handling

アプリは次の 2 形式のモデル指定を受け付けること。

- OpenVINO Whisper IR ディレクトリ
- Hugging Face の Whisper model ID

model ID の場合は次の条件で自動変換すること。

- エクスポートは `automatic-speech-recognition-with-past` を使う
- 出力先は `--model-cache-dir` または `SETUP_MODEL_CACHE_DIR` 配下とする
- 重み形式は `--weight-format` または `SETUP_WEIGHT_FORMAT` で切り替える

現行アプリで必須とする IR:

- `openvino_encoder_model.xml`
- `openvino_encoder_model.bin`
- `openvino_decoder_model.xml`
- `openvino_decoder_model.bin`
- `openvino_tokenizer.xml`
- `openvino_detokenizer.xml`

任意ファイル:

- `openvino_decoder_with_past_model.xml`
- `openvino_decoder_with_past_model.bin`

任意ファイルが無い場合でも、必須 IR が揃っていれば有効モデルとして扱う。  
この場合、アプリは警告ログを表示して続行する。

### 3.3 Realtime transcription

アプリはマイクから音声を取り込み、発話区間のみを推論に送ること。

要件:

- `sounddevice.InputStream` を使用する
- モノラル 16kHz を既定とする
- WebRTC VAD を使って無音区間を除外する
- 発話終了後にセグメントをまとめて `WhisperPipeline.generate()` に渡す
- 結果テキストを逐次 CLI / GUI に表示する

### 3.4 CLI

CLI は次をサポートすること。

- `--list-mics`
- 標準のリアルタイム認識実行
- `--device` で `AUTO` `CPU` `GPU` `NPU` を選択
- `--model` / `--model-id`
- `--language`
- `--task`
- `--model-cache-dir`
- `--weight-format`
- `--mic`

`--gui` と `--cli` を同時指定した場合はエラーにする。

### 3.5 GUI

GUI は次を提供すること。

- モデル入力欄
- デバイス選択
- マイク選択
- `Start` / `Stop` / `Clear`
- ステータス表示
- 認識結果表示
- ログ表示

モデルのロードと音声認識は UI スレッドをブロックしないこと。

## 4. Non-functional requirements

- 単一ユーザー向けのローカルアプリであること
- セットアップと実行が `setup.bat` / `run.bat` で完結すること
- モデル未準備時は自動準備にフォールバックすること
- 実装は Python ファイル単位で責務分離すること

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

## 6. Operational assumptions

- 初回セットアップではモデル変換に時間がかかる
- モデル変換にはネットワーク接続が必要になる場合がある
- `decoder_with_past` が出力されないモデルでも、必須 IR が揃っていれば現行アプリは利用を試みる

## 7. Known limitations

- 音声区間の切り出しは VAD ベースで、話者分離は行わない
- 出力保存機能や字幕ファイル出力は未実装
- GUI から変更できる設定は最小限
