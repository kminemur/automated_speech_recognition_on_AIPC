# 音声認識アプリ アーキテクチャ

## 1. 構成

```text
app.py
  -> CLI 起動 or GUI 起動
  -> 共通設定 ASRConfig を生成

model_manager.py
  -> モデル入力値の解決
  -> Hugging Face model ID の自動変換
  -> Whisper GenAI 用 IR の妥当性検証

asr_engine.py
  -> マイク入力
  -> VAD
  -> 発話バッファリング
  -> WhisperPipeline 推論

realtime_asr.py
  -> CLI ラッパー

asr_gui.py
  -> PyQt6 GUI
  -> バックグラウンド起動
  -> 状態 / ログ / 結果表示
```

## 2. データフロー

1. ユーザーが `app.py` を起動する
2. 引数を `ASRConfig` にまとめる
3. `model_manager.py` がモデルを解決する
4. model ID の場合は OpenVINO IR に変換する
5. `asr_engine.py` が `WhisperPipeline` をロードする
6. `sounddevice.InputStream` でマイク入力を取得する
7. WebRTC VAD で発話区間を判定する
8. 発話セグメントを `WhisperPipeline.generate()` に渡す
9. 結果を CLI または GUI に表示する

## 3. モデル管理ポリシー

### 3.1 許可するモデル

- OpenVINO Whisper GenAI 用 IR ディレクトリ
- Hugging Face 上の Whisper model ID

### 3.2 model ID からの変換

変換コマンドは必ず次と同義であること。

```powershell
optimum-cli export openvino ^
  --model <model_id> ^
  --task automatic-speech-recognition-with-past ^
  --weight-format <weight_format> ^
  <output_dir>
```

### 3.3 禁止事項

次のような IR を `WhisperPipeline` に渡さない。

- `automatic-speech-recognition` で変換した IR
- `openvino_decoder_with_past_model.xml` を持たない IR
- tokenizer / detokenizer が揃っていない IR

### 3.4 妥当性検証

`model_manager.py` はロード前に必須ファイルを検査する。
キャッシュ済みモデルが不正なら削除し、再エクスポートする。

## 4. エラー設計

### 4.1 再発防止したい代表例

```text
Port for tensor name beam_idx was not found.
```

意味:

- `WhisperPipeline` が期待する decoder-with-past 系の入力が無い
- ほぼ確実にモデル変換方式が誤っている

対策:

- docs に `with-past` 必須を明記する
- `setup.bat` も実行時の自動変換も同じ設定にする
- 必須ファイル検査で早めに落とす

### 4.2 UI / CLI の状態

状態は次を使う。

- `Loading model`
- `Listening`
- `Transcribing`
- `Stopped`
- `Error`

## 5. スレッド設計

### 5.1 CLI

- メインスレッドで起動
- 音声処理は `RealtimeASREngine` 内のワーカースレッドで処理

### 5.2 GUI

- Qt のメインスレッドは UI 専用
- モデルロード開始は `QThread` で行う
- エンジンからの通知は signal 経由で UI に反映する

## 6. 将来拡張

- 認識履歴の保存
- SRT / TXT 出力
- VAD パラメータの GUI 詳細設定
- モデル切り替え時の事前検証 UI
- 失敗モデルの自動隔離
