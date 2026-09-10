# irui-analysis

現代日本のWeb小説における **人間 × 非人間の恋愛・婚姻物語** を対象に、教師なし意味表現から潜在構造を探索する研究用リポジトリです。

このREADMEは、研究上の現在地・再現方法・既知の限界・次の課題をまとめたものです。

---

## 1. 研究目的

本研究の目的は、現代の「異類婚姻譚」に相当する作品群が、古典的類型と同じような離散的分類として存在するのか、それとも複数の意味的自由度に沿った連続的な構造を形成しているのかを、現代Web小説コーパスから探索することです。

対象は原則として以下を満たす作品です。

- 一方が人間である
- 他方が非人間的存在である
- 両者の間に恋愛・婚姻・婚約・求婚・強制婚などの関係がある
- 二次創作として明示される作品は除外する

ここで最も重要な方法論的原則は、

```text
selection condition != analysis feature
```

です。

人間／非人間／恋愛・婚姻語彙は **コーパス選定のためだけ** に使用し、選定後の意味空間やクラスタリング特徴量には直接入れません。

古典的な「異類女房」「異類聟」「禁忌離別」「計略殺害」などの類型も、探索前のラベルとしては使用せず、必要なら最後に事後的に比較します。

---

## 2. 現在の主コーパス

### Syosetu711K

現在の主コーパスは Hugging Face 上の

```text
botp/RyokoAI_Syosetu711K
```

です。

特徴：

- 約71万作品
- 1 row = 1作品
- `id` / N-code により作品単位を保持できる
- `title`, `author`, `genre`, `chapters`, `text` などを利用可能

以前検討した `OmniAICreator/WebNovels-Ja` は、row単位では本文断片しか持たず、作品ID・タイトル・章順序を復元できなかったため、**作品単位分析の主コーパスとしては不採用**です。

---

## 3. 現在の選定方法

候補抽出では、人間語彙・非人間語彙・恋愛／婚姻語彙が同一文または隣接2文以内に共起することを高精度寄りの代理条件として用いています。

これは真の関係抽出器ではなく、あくまで lexical proxy です。

また、`keywords` に `二次創作` が明示されている作品を除外しています。

したがって現在の「オリジナル作品」制約は厳密には、

```text
original-work proxy = no explicit 「二次創作」 keyword
```

です。

### 最新パイロットコーパス

run 33 で以下を取得しました。

- scanned works: 150,000
- qualifying works: 3,419
- selected works: 500
- unique N-codes: 500
- explicit secondary works excluded: 2,254
- selected explicit secondary works: 0
- bad rows: 0
- missing N-code: 0

重要：選定条件にはまだ false positive の可能性があります。人間語・非人間語・恋愛語が近接していても、それらが実際の恋愛ペアを指していない場合があります。最終的な人文学的結論の前に、無作為サンプルによる選定精度監査が必要です。

---

## 4. 埋め込みと作品表現

日本語埋め込みモデル：

```text
cl-nagoya/ruri-v3-130m
```

主な設定：

- chunk size: 1800 chars
- overlap: 300 chars
- max chunks per work: 8
- clustering prefix: `トピック: `
- raw text と proper-noun anonymized text の両方を作成

run 33 では、

- 500 works
- 3,363 chunks
- mean 6.73 chunks/work
- median 8 chunks/work
- 369 works at cap 8

となっています。

各作品 `i` は、単一ラベルではなく、チャンク埋め込みの集合

```text
X_i = {e_i1, e_i2, ..., e_in_i}
```

として扱います。

現在使っている作品レベル表現は主に2種類です。

### 4.1 作品平均ベクトル

```text
mu_i = mean_j e_ij
```

作品の大域的な意味位置を表します。

### 4.2 作品内局所部分空間

作品内チャンクを中心化し、SVDから rank-3 の局所意味方向を推定します。

作品間の部分空間距離には principal angle を用います。

---

## 5. これまでに分かったこと

### 5.1 チャンク単位HDBSCANは不安定

HDBSCANのクラスタ数は `min_cluster_size` / `min_samples` に強く依存し、2クラスタから200以上まで大きく変化しました。

したがって、現時点では

```text
HDBSCAN cluster count = discovered literary typology
```

とは解釈していません。

離散クラスタを強制するより、作品単位の連続幾何を調べる方針へ移行しました。

### 5.2 作品平均空間はraw / 匿名化でかなり頑健

作品平均距離について：

```text
raw vs anonymized distance Spearman = 0.944410
10-NN overlap = 0.623200
```

作品内 rank-3 subspace では：

```text
subspace distance Spearman = 0.573431
10-NN overlap = 0.311353
```

したがって、作品の大域的位置はかなり頑健ですが、作品内の細かい意味方向は固有名詞や設定差・チャンク数の影響を受けやすいと考えています。

### 5.3 作品群は少数の「島」より高次元の連続空間に近い

作品平均ベクトルのPCAでは：

```text
PC1 EVR = 0.072462
PC2 EVR = 0.040192
PC3 EVR = 0.035036
PC1-3 cumulative = 0.147690
PC1-10 cumulative = 0.305755
50% variance requires 28 PCs
```

また、kNN graph は `k=2` ですでに500作品すべてが1つの連結成分になります。

MSTでも極端な単一ブリッジは観測されませんでした。

このため現在は、

```text
少数の離散類型
```

より、

```text
複数の連続的意味自由度を持つ高次元作品空間
```

として扱う方が妥当だと考えています。

ただし、これは「異類婚姻そのものが連続的」と証明したわけではありません。全文意味空間には一般的なWeb小説ジャンル・文体・作品長などの変動も含まれます。

---

## 6. 一般的な作品意味を除く残差解析

全文作品空間には、学園・戦闘・日常・説明文・性的描写・作品長など、異類婚姻に直接関係しない一般意味が含まれます。

その影響を減らすため、全チャンクからコーパス共通PCA方向を学習し、上位 `m` 成分を除去します。

```text
e'_ij = e_ij - sum_k (e_ij . v_k) v_k
```

現在は

```text
m = 0, 1, 2, 3, 5, 10
```

を比較しています。

### 主な結果

rawでは：

```text
PC1 EVR:       0.0725 -> 0.0256  (m=10)
PC1-10 EVR:    0.3058 -> 0.1898
```

genreとの最大関連も、

```text
eta^2 ~ 0.366 at m=0
eta^2 ~ 0.151 at m=3
```

まで低下しました。

一方、残差化後もraw / anonymizedの作品間距離は高い相関を保ちます。

```text
m=3:  distance Spearman ~ 0.916
m=10: distance Spearman ~ 0.921
```

つまり、一般的なジャンル方向をある程度除いても、作品間の大域的構造そのものは残ります。

現在の主候補は `m=3`、感度解析として `m=0,3,5` を比較する方針です。

---

## 7. 残差PCの解釈についての重要な注意

`m=3` 残差空間でPC1〜PC10の極端作品を抽出しましたが、個々のPCはraw / anonymized間で安定しませんでした。

例えば、raw PC1とanonymized PC1のscore相関は高くなく、極端作品の重なりも小さいです。

これは残差空間そのものが不安定というより、

```text
lambda_1 ~ lambda_2 ~ lambda_3 ...
```

となった結果、PCA基底がraw / anonymized間で回転している可能性が高いです。

そのため、今後は

```text
PC1 = one literary meaning axis
```

と直接読むのではなく、

```text
stable low-dimensional residual subspace
```

をraw / anonymized間で整列させ、その中の再現可能な方向を読む方針です。

候補手法：

- CCA
- orthogonal Procrustes
- principal vectors / principal angles
- shared subspace extraction

---

## 8. 今後の重要課題

### A. 残差空間の安定方向を抽出する

raw / anonymizedの残差部分空間を整列し、再現可能な意味方向のみを文学的に解釈します。

### B. 無作為対照群を導入する

これは非常に重要です。

本来は、

```text
selected human-nonhuman romance works
vs
random Syosetu711K works
```

を比較し、Web小説一般の意味構造からの偏りとして対象群を記述する必要があります。

これにより、

```text
general Web-novel structure
```

と

```text
structure specific to selected works
```

をより明確に分離できます。

### C. 選定精度を監査する

候補500作品から無作為に約100作品を確認し、

- 真に人間 × 非人間の恋愛／婚姻か
- false positive の型は何か

を確認する必要があります。

### D. 作品長・文体の交絡を追加確認する

現在確認できているもの：

- body_text_chars
- chapters
- points
- genre / biggenre

今後追加したいもの：

- dialogue ratio
- narration ratio
- POV proxy
- lexical diversity
- author effect

これらは分析特徴として投入せず、**得られた意味構造との相関を事後的に確認するため**に使います。

---

## 9. 現在の推奨ローカル実行方法

GitHub Actionsは現在、**pushでは自動実行されません**。

`pilot.yml` / `work-space.yml` ともに `workflow_dispatch` のみです。

通常の後段解析はローカルで行います。

### 初回セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

run 33 の既存artifactから埋め込みキャッシュを取得する場合：

```bash
bash bootstrap_local_cache.sh
```

必要なのは以下です。

```text
results/candidates.csv
results/chunk_manifest.csv
results/embeddings_raw.npy
results/embeddings_anonymized.npy
```

### 通常実行

```bash
python run_analysis.py
```

設定は

```text
local_analysis.yaml
```

で管理します。

ローカル解析終了後には

```text
results/local_summary.json
```

が生成されます。

このsummaryと必要なPNG / CSVだけをChatGPTに渡せば、巨大な埋め込みや本文を毎回アップロードせずに次の解析方針を決められます。

詳細は `LOCAL_RUN.md` を参照してください。

---

## 10. 主な解析スクリプト

```text
fetch_hf_candidates.py
    Syosetu711Kから候補作品を選定

run_all.py
    全文チャンク化・Ruri埋め込み生成

analyze_work_space.py
    作品平均・作品内部分空間の構築

analyze_work_geometry.py
    PCA spectrum / kNN / MSTなど連続幾何を診断

analyze_residual_space.py
    コーパス共通成分を除去した残差空間を構築

interpret_residual_axes.py
    残差PCの極端作品と寄与チャンク位置を抽出

run_analysis.py
    ローカル後段解析の統合入口
```

---

## 11. 研究上の現在地

現時点で最も安全に言えることは次です。

> 選定された500作品の全文意味空間は、少数の安定した離散クラスタというより、一つにつながった高次元の連続構造として観測される。

ただし、この空間には一般的なWeb小説ジャンル差も混入している。

コーパス共通成分の除去によりgenre依存は低下するが、残差PC単体はraw / anonymized間で回転しており、PC1などを直接文学的意味軸として読むのはまだ早い。

現在の最優先課題は、

```text
1. residual shared subspace extraction
2. random-control comparison
3. selection precision audit
```

です。

古典的異類婚姻譚との比較は、これらの教師なし構造を確立した後に行います。

---

## 12. データ・著作権上の方針

Web小説本文は分析対象として利用しますが、本文そのものを研究成果物として再配布することは基本的に避けます。

GitHubやartifactには、可能な限り以下のみを保存します。

- N-code
- メタデータ
- 埋め込み
- 距離行列
- 集計値
- チャンク位置
- 短い分析用スニペット（必要時のみ）

元データセットの利用条件および日本の著作権法上の情報解析規定等は、実際の公開・再配布時に別途確認してください。

---

## 13. Status

Current branch:

```text
analysis-pipeline
```

Current execution policy:

```text
Local analysis by default
GitHub Actions only when manually dispatched
Embedding recomputation avoided whenever cache exists
```

Current main analysis target:

```text
Stable residual semantic subspace after removing corpus-wide common components
```
