**ANIMACY SOUND LEARNING**

動物の鳴き声には生物らしさ(アニマシー)が含まれていると仮定します。
音声から、動物の鳴き声かそうでないかを分類することで、アニマシーの抽出を目指します。

# Environment
|name|value|
|---|---|
| SYSTEM | Ubuntu-22.04 (on WSL) |
| CUDA | 11.3 |

```
AnimacySoundLearning/
├── data/
│   ├── raw/                     # 元の音声データ
│   ├── processed/               # 前処理済みのデータ（メルスペクトログラム画像など）
│   ├── metadata/                # メタデータ（CSVファイル）
│   │   └── dataset_metadata.csv # 音声データとラベル情報を記載
├── src/
│   ├── data_preprocessing/      # データ前処理関連のスクリプト
│   │   ├── generate_melspectrograms.py
│   │   └── dataloader.py        # データローダー
│   ├── models/                  # モデル関連
│   │   ├── resnet_model.py      # ResNetの定義
│   │   └── train.py             # 学習スクリプト
│   ├── evaluation/
│   │   └── evaluate_model.py    # モデル評価スクリプト
│   ├── main.py                  # プロジェクトのエントリーポイント
├── notebooks/
│   ├── exploration.ipynb        # データやモデルの可視化用ノートブック
├── outputs/
│   ├── models/                  # 学習済みモデルの保存先
│   │   └── resnet_best.pth      # ベストモデルのチェックポイント
│   ├── logs/                    # 学習時のログ（TensorBoardやCSV）
│   ├── results/                 # 評価結果や可視化結果
├── requirements.txt             # 必要なPythonパッケージ一覧
├── README.md                    # プロジェクトの概要と使用方法
├── .env                         # 環境変数
└── .gitignore                   # Gitで追跡しないファイルのリスト

```

# Usage

## check Environment
```
make check-env
```
## shell
```
make shell
```
