# 自動音声認識アプリ（OpenVINO + kotoba-whisper-v2.2）

1秒ごとにマイク音声を切り出し、OpenVINOでGPU/NPU/CPU上に最適化した kotoba-whisper-v2.2 でリアルタイム認識します。OpenVINO IR を事前に生成しておけば起動が高速になります。

## 環境前提
- Windows（PowerShell）
- Python 3.10+
- GPU/NPU/CPU で OpenVINO が動作する環境

## セットアップ
```powershell
.\setup.bat
```
仮想環境 `.venv` を作成し、必要なライブラリをインストールします。

## 実行方法
### 既存の IR を自動利用（推奨フロー）
リポジトリ直下で:
```powershell
.\run.bat
```
- `run.bat` は `ir_kotoba` フォルダに `openvino_model.xml` があれば自動で `--model-id ir_kotoba` を指定します。
- 引数に `--model-id` を渡した場合はそちらを優先します。

### 初回: IR を生成してから実行
```powershell
.\run.bat --export-ir .\ir_kotoba --device GPU --model-id kotoba-tech/kotoba-whisper-v2.2
```
生成後は `.\run.bat` だけで IR を使った実行ができます。

### デバイス指定
- 環境変数 `DEVICE` でデフォルトデバイスを指定可能（例: `set DEVICE=CPU` の後に `.\run.bat`）。
- もしくは `--device GPU` のように引数で明示指定してください。

### そのほかの主なオプション（`app.py`）
- `--cache-dir`: モデル/IR キャッシュパス（デフォルト: `./.cache_whisper`）
- `--dynamic-shapes`: 動的形状を有効化（デフォルト無効。NPUでは強制的に無効化され静的形状のみ）
- `--sample-rate`: マイクサンプルレート（デフォルト 16000）
- `--chunk-seconds`: チャンク長（秒）。デフォルト 1.0。NPU では常に 1 秒に固定
- `--task`: `transcribe` もしくは `translate`

## アプリの挙動
- 1秒ごとにマイク音声を取得し、そのまま推論。
- 生成結果を逐次標準出力に表示。Ctrl+C で終了。
- キャッシュは `./.cache_whisper` を使用し、2回目以降の起動を高速化。

## トラブルシュート
- **初回起動が遅い**: IR 生成とコンパイルに時間がかかります。一度 `--export-ir` で生成し、以後は `--model-id .\ir_kotoba` を使ってください。
- **IR 使用時に形状エラー/NPUでチャネル警告**: NPU では動的形状を常に無効化します。`--dynamic-shapes` を付けても静的形状になります。
- **CPUで試したい**: `set DEVICE=CPU` を付けて `.\run.bat`。
