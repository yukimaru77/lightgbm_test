# lightgbm_test

LightGBM を使ったテーブルデータ分類の最小サンプルです。

このプロジェクトでは、`scikit-learn` の `make_classification` を使ってダミーの表形式データを生成し、`LightGBM` で二値分類モデルを学習します。

## 何をするプロジェクトか

- ダミーのテーブルデータを 1,000 行生成
- CSV として `data/dummy_tabular_data.csv` に保存
- LightGBM で分類モデルを学習
- 精度指標と特徴量重要度を `outputs/` に保存

## 使っている主なパッケージ

- `lightgbm`
- `pandas`
- `scikit-learn`
- `uv`（依存管理・実行）

## セットアップ

このリポジトリでは Python 依存管理に `uv` を使います。

```bash
uv sync
```

## 実行方法

```bash
uv run train-lightgbm
```

## 実行結果

実行すると、以下のファイルが生成されます。

- `data/dummy_tabular_data.csv`
- `outputs/metrics.json`
- `outputs/feature_importance.csv`
- `outputs/classification_report.txt`

今回の実行例:

```json
{
  "accuracy": 0.895,
  "roc_auc": 0.9636
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
├── src/
│   └── lightgbm_test/
│       ├── __init__.py
│       └── train.py
├── pyproject.toml
└── README.md
```

## GPU / CUDA 前提

- このサンプルは **CPU 前提** です。
- GPU / CUDA 向けの設定は行っていません。
- LightGBM の GPU 学習を使いたい場合は、別途 GPU 対応ビルドやパラメータ調整が必要です。

## 既知の制限

- データは実データではなく、合成したダミーデータです。
- 学習・評価は単一スクリプトの最小構成です。
- ハイパーパラメータ最適化、交差検証、モデル保存は未実装です。
