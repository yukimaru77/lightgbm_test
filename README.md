# lightgbm_test

LightGBM を **GPU (CUDA)** で動かすテーブルデータ分類サンプルです。

このプロジェクトでは、`scikit-learn` の `make_classification` を使ってダミーの表形式データを生成し、`LightGBM` を **CUDA 有効ビルド** して GPU 学習を行います。

## 何をするプロジェクトか

- ダミーのテーブルデータを 1,000 行生成
- CSV として `data/dummy_tabular_data.csv` に保存
- LightGBM を CUDA 対応でビルド
- GPU で二値分類モデルを学習
- 精度指標と特徴量重要度を `outputs/` に保存

## 使っている主なパッケージ・ツール

- `lightgbm`
- `pandas`
- `scikit-learn`
- `uv`
- `cmake`
- `ninja`
- `CUDA Toolkit`

## 前提条件

以下が入っていることを前提にしています。

- NVIDIA GPU
- `nvidia-smi` が使えること
- CUDA Toolkit
- `cmake`
- `ninja`
- `git`

この環境では実際に以下を確認済みです。

- GPU: `NVIDIA GB10`
- Compute Capability: `12.1`
- CUDA: `13.0`

## セットアップ

まず Python 依存関係を同期します。

```bash
uv sync
```

次に、LightGBM を CUDA 対応でソースビルドします。

```bash
./scripts/build_lightgbm_cuda.sh
```

このスクリプトは以下を行います。

- `nvidia-smi` から GPU の compute capability を取得
- LightGBM v4.6.0 を clone
- CUDA アーキテクチャをその GPU に合わせてパッチ
- `.venv` に CUDA 対応版 LightGBM をインストール

## 実行方法

GPU 版 LightGBM を入れたあとは、`uv` の自動同期で CPU wheel に戻さないために `--no-sync` を付けて実行します。

```bash
uv run --no-sync train-lightgbm
```

まとめて実行する場合:

```bash
./scripts/run_gpu_demo.sh
```

## 実行結果

実行すると、以下のファイルが生成されます。

- `data/dummy_tabular_data.csv`
- `outputs/metrics.json`
- `outputs/feature_importance.csv`
- `outputs/classification_report.txt`

今回の GPU 実行例:

```json
{
  "accuracy": 0.905,
  "roc_auc": 0.9627,
  "device_type": "cuda"
}
```

## ディレクトリ構成

```text
.
├── data/
│   └── dummy_tabular_data.csv
├── outputs/
│   ├── classification_report.txt
│   ├── feature_importance.csv
│   └── metrics.json
├── scripts/
│   ├── build_lightgbm_cuda.sh
│   └── run_gpu_demo.sh
├── src/
│   └── lightgbm_test/
│       ├── __init__.py
│       └── train.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## GPU / CUDA 前提

- このサンプルは **GPU / CUDA 前提** です。
- 学習コードは `device_type="cuda"` を使っています。
- LightGBM の配布 wheel は CUDA 有効でない場合があるため、`scripts/build_lightgbm_cuda.sh` でソースビルドしています。
- GPU アーキテクチャは `nvidia-smi` から取得して自動調整します。

## 既知の制限

- データは実データではなく、合成したダミーデータです。
- LightGBM の GPU ビルドは環境依存です。
- `uv run` を通常実行すると、依存同期の都合で CPU 版 wheel に戻ることがあるため、学習実行時は `uv run --no-sync` を使ってください。
- ハイパーパラメータ最適化、交差検証、モデル保存は未実装です。
