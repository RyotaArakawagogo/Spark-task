# Spark+ Computer Vision v2 課題

本リポジトリは Spark+ 様から提供された Computer Vision v2 課題に対する実装です。  
CIFAR-10 データセットを用い、ResNet50 をベースに 6 種類の改善施策を比較しました。

---

## 1. 環境構築手順

本プロジェクトは **Python 3.9 ～ 3.11** を推奨します。

```bash
# リポジトリをクローン
git clone <REPO_URL>
cd <repo_name>

# 依存パッケージのインストール
pip install -r requirements.txt

```

## 2. 学習実行手順

train.py の `--exp` 引数で実験を切り替えて実行できます。

```bash
python train.py --exp baseline
python train.py --exp augment
python train.py --exp lr
python train.py --exp transfer
python train.py --exp wd
python train.py --exp transfer_augment
```

## 3. 推論実行手順

      学習済みモデル 'best.pt'を用いて、任意の画像のクラスを予測します。
python predict.py \
  --image sample_data/sample.png \
  --weights artifacts/baseline/<timestamp>/best.pt


## 4. ログの確認方法
      ファイル名              | 内容

      `config.json`           | 実験設定 (ハイパーパラメータ、施策フラグ)
      `history.csv`           | 各epochの train/val loss・accuracy
      `curves_loss.png`       | Lossの学習曲線
      `curves_acc.png`        | Accuracyの学習曲線
      `confusion_matrix.png`  | 最終モデルの混合行列
      `misclassified.png`      | 誤分類サンプル画像の一覧
      `report.txt`            | Precision / Recall / F1 の詳細
      `best.pt`               | 最良 val_acc を達成したモデルの重み

repo_root/
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── sample_data/
    └── sample.png
# artifacts/ は提出不要（train.py 実行で再生成可能）


