# irui-analysis

漫画・ライトノベル／Web小説における異類婚姻・異類恋愛の通時的変化を、教師なしテクスト分析で探索する研究用リポジトリ。

## Research principle

このプロジェクトでは、最初から「怪異性」「親密性」「日常性」などの仮説的軸を特徴量として与えない。

1. 候補抽出は高再現率の弱いルールで行う。
2. 本文をチャンク化し、ラベルなし埋め込みを作る。
3. PCA / UMAP / HDBSCAN で潜在構造を探索する。
4. クラスタ代表文脈を読んで事後的に意味を解釈する。
5. 年代、モチーフ、媒体、ジェンダー配置、古典的異類婚姻譚類型は最後に重ねる。

重要なのは、候補抽出に使った語彙をクラスタリング特徴として混ぜないこと。

## Pipeline

```text
WebNovels-Ja / other legal corpora
        ↓
candidate retrieval
        ↓
text chunking
        ↓
Japanese sentence embedding
        ↓
PCA → UMAP → HDBSCAN
        ↓
cluster interpretation
        ↓
within-work semantic trajectories
        ↓
post-hoc comparison with year / motif / classical typology
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --input '/path/to/*.jsonl'
```

Hugging Face の gated dataset を直接取得する場合は、事前に利用条件を確認し、必要なら `hf auth login` を行う。

## Outputs

- `results/candidates.csv` 候補作品
- `results/chunk_manifest.csv` チャンクと作品内位置
- `results/embeddings.npy` 埋め込み
- `results/clusters.csv` UMAP / HDBSCAN 結果
- `results/cluster_summary.csv` クラスタ規模など
- `results/umap.png` 潜在空間

本文そのものの再配布はしない設計を基本とする。
