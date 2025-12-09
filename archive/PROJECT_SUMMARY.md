# SRO Knowledge Editing Platform - プロジェクトサマリー

## 実装完了状況

✅ **完全実装済み** - すべてのコアコンポーネントが動作検証済み

## アーキテクチャ概要

### 1. データ生成 (`src/kg/`)

- **KG生成**: ER/BAグラフ生成
- **言い換え**: エンティティ/リレーションごとに5種の同義語
- **コーパス出力**: 3トークンSRO形式

### 2. モデリング (`src/modeling/`)

- **トークナイザー**: 特殊トークン対応（`[E_DELETED]`等）
- **GPTミニ**: 6/12/24層の軽量Transformer
- **訓練**: AdamW、Cosine decay、20%サンプリング学習

### 3. 編集 (`src/edit/`)

- **Causal Tracing**: 知識保存層の特定
- **ROME**: FFN W2へのrank-1更新

### 4. 評価 (`src/eval/`)

- **局所成功**: Top-1 acc、rank、log-prob
- **グローバル副作用**: 影響度I、次数相関、hop波及
- **CKE**: 保持、上書き、可塑性、安定性

### 5. 可視化 (`src/eval/viz.py`)

- 次数-影響度散布図
- PCA埋め込み
- CKEヒートマップ
- 順序効果箱ひげ図

## CLIコマンド

### KG生成
```bash
python -m src.cli.build_kg --config configs/kg_ba.yaml
```

### モデル訓練
```bash
python -m src.cli.train_lm --config configs/train_gpt_small.yaml
```

### CKE実験
```bash
python -m src.cli.run_seqedit \
  --config configs/cke_pipeline.yaml \
  --model-dir outputs/models/gpt_small \
  --kg-dir data/kg/ba \
  --num-scenarios 10 \
  --steps 5
```

## 設定ファイル

| ファイル | 用途 |
|---------|------|
| `configs/kg_ba.yaml` | BAグラフ生成 |
| `configs/kg_er.yaml` | ERグラフ生成 |
| `configs/train_gpt_small.yaml` | GPTミニ訓練 |
| `configs/edit_rome.yaml` | ROME編集 |
| `configs/cke_pipeline.yaml` | CKE実験 |

## 実験プロトコル

### 影響度測定

1. **編集前評価**: 全SROでacc測定
2. **編集適用**: ROME (layers=[0,1,2])
3. **編集後評価**: 同一SROで再測定
4. **影響度算出**: `I = acc_pre - acc_post`

### 次数相関

- BAグラフ: 高次数エンティティで高影響度を期待
- ERグラフ: 次数と影響度の相関が低いことを期待
- 検定: Pearson/Spearman相関係数、p値

### CKE（継続知識編集）

#### 条件

- **共有型固定**: 共有関係を固定順で編集
- **排他型固定**: 排他関係を固定順で編集
- **混在固定**: 混在を固定順で編集
- **混在シャッフル**: 混在をランダム順で編集

#### 指標

- **Plasticity**: 新規編集の局所成功率
- **Stability**: 既存知識の保持率
- **Retention**: 共有型関係の累積保持
- **Overwrite**: 排他型関係の最新上書き

## テスト

すべてのコアモジュールにユニットテスト完備:

```bash
# 個別実行
python tests/test_kg_generation.py
python tests/test_tokenizer.py
python tests/test_model.py

# 一括実行
bash tests/run_all_tests.sh
```

## 依存パッケージ

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
networkx>=3.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

## 期待される出力

### 学習後

- `acc_train` ≈ 0.95-0.99（サンプリング分）
- `acc_all` ≈ 0.95-0.99（全言い換え=一般化確認）

### ROME編集後

- **局所成功率**: 80-95%
- **影響度I**: BA > ER（構造依存を確認）
- **次数相関**: BA=強、ER=弱

### CKE

- **Plasticity**: 0.8-0.9（新規編集成功率）
- **Stability**: 0.7-0.9（既存保持）
- **順序効果**: Shuffle < Fixed（順序の影響）

## ファイル構成

```
sro-kedit/
├── configs/            # 設定ファイル ✓
├── data/               # 生成データ保存先
│   ├── kg/
│   │   ├── ba/        # BAグラフ
│   │   └── er/        # ERグラフ
│   └── vocab/         # 語彙
├── src/
│   ├── kg/            # KG生成 ✓
│   ├── modeling/      # モデル・訓練 ✓
│   ├── edit/          # Causal Tracing・ROME ✓
│   ├── eval/          # 評価・CKE・可視化 ✓
│   ├── utils/         # ユーティリティ ✓
│   └── cli/           # CLIスクリプト ✓
├── tests/             # ユニットテスト ✓
├── reports/           # 出力図表保存先
│   ├── figures/
│   └── tables/
├── requirements.txt   # 依存パッケージ ✓
├── README.md          # メインドキュメント ✓
├── PROJECT_SUMMARY.md # このファイル
└── example_workflow.sh # 実行例スクリプト ✓
```

## 次のステップ（将来拡張）

### 編集手法追加

- [ ] MEMIT（マルチトークン編集）
- [ ] AlphaEdit（学習率制御）

### 評価拡張

- [ ] 対照編集（ランダムベースライン）
- [ ] 類似度制御（コサイン類似度による条件A/B）
- [ ] レイヤーアブレーション

### 可視化追加

- [ ] 注意重み可視化
- [ ] 活性化変化ヒートマップ
- [ ] 時系列メトリクス

## 論文生成に必要な図表

すべて実装済み:

1. ✅ ER vs BA次数-影響度散布図（相関係数付き）
2. ✅ Hop別波及ヒートマップ
3. ✅ CKEステップ進行図
4. ✅ CKE条件別ヒストグラム
5. ✅ 順序効果箱ひげ図
6. ✅ PCA埋め込み（言い換え凝集確認）

## 再現性保証

- ✅ シード固定（PyTorch, NumPy, サンプリング）
- ✅ 設定ファイルYAML保存
- ✅ メトリクスCSV/JSON出力
- ✅ 詳細ログ記録

## ライセンス

MIT License

## 謝辞

設計書に基づき、研究用プラットフォームとして完全実装されました。
