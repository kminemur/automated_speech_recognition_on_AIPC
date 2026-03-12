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
  Runs WhisperPipeline inference

realtime_asr.py
  CLI runner
  Prints status, logs, and transcripts

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

`setup.bat` の処理順:

1. `python` を試す
2. `py -3` を試す
3. `py` を試す
4. Python 3.10+ なら採用する
5. `.venv` を作成する
6. `requirements.txt` をインストールする
7. `ensure_model_available()` を呼び、既定モデルを準備する

既定値:

- `SETUP_MODEL=openai/whisper-tiny`
- `SETUP_MODEL_CACHE_DIR=.cache_whisper`
- `SETUP_WEIGHT_FORMAT=int8`

### 2.2 Run

`run.bat` は `.venv` の存在を確認し、`python app.py %*` を実行する薄いラッパーである。

## 3. App flow

### 3.1 Entry point

`app.py` は次を担当する。

- 引数解析
- `--list-mics` の即時処理
- `ASRConfig` の生成
- CLI / GUI の分岐

既定動作は CLI 起動で、`--gui` 指定時のみ GUI を起動する。

### 3.2 Model resolution

`model_manager.ensure_model_available()` は次の順でモデルを解決する。

1. 引数が有効な IR ディレクトリか判定する
2. `cache_dir / model.replace("/", "--")` に既存変換済みモデルがあるか判定する
3. なければ `optimum-cli export openvino` を実行する
4. `optimum-cli` が見つからない場合は `python -m optimum.exporters.openvino` を使う
5. 必須 IR が揃っているか再検証する

必須 IR:

- encoder
- decoder
- tokenizer
- detokenizer

任意 IR:

- `decoder_with_past`

任意 IR が無い場合でもモデルは有効とする。  
`asr_engine.py` は不足をログ出力した上で `WhisperPipeline` の初期化を試みる。

エクスポート時の task は固定で `automatic-speech-recognition-with-past` を使う。

## 4. Audio pipeline

`asr_engine.RealtimeASREngine` は共有エンジンであり、CLI と GUI の両方から利用される。

処理の流れ:

1. モデルを準備する
2. 任意 IR の不足を確認し、必要なら警告ログを出す
3. `WhisperPipeline` を生成する
4. generation config に `language` `task` `return_timestamps` を設定する
5. `sounddevice.InputStream` でマイク入力を開始する
6. コールバックで float32 音声をキューへ入れる
7. ワーカースレッド側で PCM16 に変換し、WebRTC VAD で発話判定する
8. 発話セグメントを連結して `pipeline.generate(audio.tolist())` を呼ぶ
9. テキストを抽出してイベントとして返す

既定設定:

- `sample_rate=16000`
- `block_duration_ms=30`
- `silence_ms_to_flush=700`
- `pre_speech_padding_ms=300`
- `vad_aggressiveness=2`
- `max_buffer_seconds=15.0`

## 5. CLI design

`realtime_asr.py` の責務:

- マイク一覧表示
- ステータス出力
- ログ出力
- 認識テキスト出力
- `SIGINT` / `SIGTERM` による停止

ステータスは少なくとも次を使う。

- `Loading model`
- `Listening`
- `Transcribing`
- `Error`
- `Stopped`

ワーカースレッド内の例外はログとステータスに変換し、不要な traceback を表に出さない。

## 6. GUI design

`asr_gui.py` は `QThread` 上でエンジンを動かす。

構成:

- `ASRWorker`
  - `RealtimeASREngine` の起動と停止
  - status / log / transcript の signal 中継
- `MainWindow`
  - モデル入力
  - デバイス選択
  - マイク選択
  - `Start` `Stop` `Clear`
  - ステータス表示
  - 認識結果表示
  - ログ表示

## 7. Error handling policy

- Python 3.10+ が見つからなければ `setup.bat` は即時終了する
- `.venv` がなければ `run.bat` は `setup.bat` 実行を要求して終了する
- 変換後に必須 IR が欠けていれば例外を投げる
- 任意 IR が欠けていても警告ログのみで続行する
- 推論や音声入力で例外が出たら `Error` ステータスを通知する

## 8. Current gaps

実装済みでない項目:

- 認識結果のファイル保存
- GUI 上での高度な VAD 調整
- SRT / TXT 出力
- バックグラウンド再試行や進捗バー
