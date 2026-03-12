# 音声認識アプリ要件

## 1. 目的

Windows 上でマイク入力を受け取り、OpenVINO 2026.0 と
`openvino-genai` の `WhisperPipeline` でリアルタイム音声認識を行う。
GUI と CLI の両方を提供する。

## 2. 最重要制約

### 2.1 モデル変換方式

このアプリは `WhisperPipeline` を使うため、Whisper の OpenVINO IR は必ず次で変換する。

```powershell
optimum-cli export openvino --model <model_id> --task automatic-speech-recognition-with-past <output_dir>
```

使ってはいけない変換:

```powershell
optimum-cli export openvino --model <model_id> --task automatic-speech-recognition <output_dir>
```

理由:

- `WhisperPipeline` は `decoder_with_past` を前提に動作する
- `with-past` でない IR には必要な入力が揃わない
- 代表的な失敗が `beam_idx was not found`

### 2.2 有効なモデルディレクトリの条件

有効な Whisper GenAI 用 IR ディレクトリには、少なくとも次が存在すること。

- `openvino_encoder_model.xml`
- `openvino_encoder_model.bin`
- `openvino_decoder_model.xml`
- `openvino_decoder_model.bin`
- `openvino_decoder_with_past_model.xml`
- `openvino_decoder_with_past_model.bin`
- `openvino_tokenizer.xml`
- `openvino_detokenizer.xml`

上記のどれかが無ければ、そのディレクトリはこのアプリでは無効とみなす。

## 3. 対象ユーザー

- OpenVINO 対応 Windows PC で音声認識を試したいユーザー
- AIPC の CPU / GPU / NPU を切り替えて検証したいユーザー
- Whisper を GUI と CLI の両方で使いたいユーザー

## 4. 機能要件

### 4.1 基本

- マイク入力をリアルタイムに取得できる
- WebRTC VAD で発話区間を切り出せる
- 発話ごとに `WhisperPipeline.generate()` で認識できる
- 認識結果を GUI と CLI に表示できる
- ログと状態を表示できる

### 4.2 モデル管理

- `--model` にローカルの IR ディレクトリまたは Hugging Face の model ID を指定できる
- model ID を指定した場合は自動で OpenVINO IR に変換できる
- 変換時は必ず `automatic-speech-recognition-with-past` を使う
- 不正なキャッシュを検出したら再エクスポートできる

### 4.3 CLI

- 既定では CLI モードで起動できる
- `--list-mics` で入力デバイス一覧を表示できる
- `--gui` で GUI を起動できる
- `--device` で `CPU` `GPU` `NPU` `AUTO` を選べる
- `--language` と `--task` を指定できる

### 4.4 GUI

- モデル、デバイス、マイク、言語、タスクを変更できる
- `Start`、`Stop`、`Clear` を持つ
- テキスト表示、ログ表示、入力レベル表示を持つ
- モデルロードは UI を固めないように別スレッドで行う

## 5. 非機能要件

### 5.1 安定性

- 不正なモデルを使った場合は明示的なエラーを出す
- `beam_idx was not found` のような再発しやすい問題を docs とコードの両方で防ぐ
- `setup.bat` と実行時の自動準備で同じ変換方式を使う

### 5.2 保守性

- GUI / CLI / エンジン / モデル管理の責務を分離する
- 共通設定は 1 つの設定クラスで管理する

### 5.3 実行環境

- Windows
- Python 3.10 以上
- OpenVINO 2026.0

## 6. 依存関係

- `openvino==2026.0.0`
- `openvino-genai==2026.0.0.0`
- `openvino-tokenizers==2026.0.0.0`
- `optimum-intel[openvino]`
- `huggingface_hub`
- `sounddevice`
- `webrtcvad-wheels`
- `PyQt6`

## 7. セットアップ要件

`setup.bat` は次を行う。

- 利用可能な Python を探して `.venv` を作る
- `requirements.txt` をインストールする
- 必要なら `torch` を入れる
- 既定モデルを `automatic-speech-recognition-with-past` で変換する

## 8. 運用ルール

### 8.1 新しいモデルを使うとき

- model ID 指定なら、このアプリの自動変換に任せる
- 手動変換する場合も `automatic-speech-recognition-with-past` を使う

### 8.2 ローカル IR を直接渡すとき

- `openvino_decoder_with_past_model.xml` が無い IR は使わない
- 以前に `automatic-speech-recognition` で作った IR を流用しない

### 8.3 エラー時

次のエラーが出たら、原因はまずモデル変換方式を疑う。

```text
Port for tensor name beam_idx was not found.
```

対処:

1. モデルディレクトリに `openvino_decoder_with_past_model.xml` があるか確認する
2. 無ければキャッシュを削除する
3. `with-past` で再エクスポートする

## 9. 受け入れ条件

- `run.bat --list-mics` が動作する
- `run.bat` で CLI 起動できる
- `run.bat --gui` で GUI 起動できる
- 正しい `with-past` モデルで音声認識できる
- 誤った IR を使った場合は、黙って壊れず分かる形で失敗する
