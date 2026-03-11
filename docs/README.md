# docs

このディレクトリには、OpenVINO 2026.0 を使ったリアルタイム音声認識アプリの設計資料を置く。

- `requirements.md`: 機能要件、非機能要件、受け入れ条件
- `architecture.md`: コンポーネント構成、データフロー、責務分割

実装前の合意形成にも、実装後の見直しにも使える内容にしている。

補足:
- アプリはローカルの OpenVINO IR モデルだけでなく Hugging Face の Whisper モデル ID も受け付ける
- モデル ID が指定された場合、初回起動時に自動ダウンロードし `optimum-cli export openvino` で IR に変換して再利用する
- `setup.bat` は Python を自動検出して `.venv` を作成し、依存導入後に既定モデルのダウンロードと IR 変換も先に実行する
- `run.bat` は既定で CLI を起動し、GUI は `--gui` 指定時のみ起動する
- GUI の `Start` は安定性のためモデルロードをメインスレッドで行うため、開始直後に短時間 UI が停止することがある
