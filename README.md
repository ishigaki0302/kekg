# SRO Knowledge Editing Platform

SROトリプルを学習したモデルでの知識編集とRipple Effect分析プラットフォーム

---

## クイックスタート

### 1. ナレッジグラフ生成

```bash
python src/kg/generator.py \
    --num-entities 1200 \
    --num-relations 250 \
    --target-triples 30000 \
    --seed 42
```

### 2. モデル学習

```bash
./run/run_training.sh
# または設定ファイルを指定
./run/run_training.sh --config configs/train_gpt_small.yaml
```

### 3. Locating（Causal Tracing）

```bash
./run/run_locating.sh
```

### 4. Editing（知識編集・単発）

```bash
./run/run_editing.sh --subject E_0001 --relation R_022 --target E_0999 --layer 0
```

### 5. Ripple Effect 分析

```bash
# デフォルト（2実験、v_num_grad_steps=20）
./run/run_ripple_analysis.sh

# 件数・勾配ステップを指定
./run/run_ripple_analysis.sh 10 20
```

### 6. Sequential Editing（逐次知識編集）

#### 6.1 単一条件

```bash
# 基本実行（random選択、100ステップ）
./run/run_sequential_edits.sh

# 選択モード・ステップ数を指定
./run/run_sequential_edits.sh --edit-selection-mode degree_high --num-steps 200

# 結果の可視化
./run/run_sequential_analysis.sh --output-dir outputs/sequential
```

**編集選択モード:**
| モード | 説明 |
|---|---|
| `random` | ランダム（デフォルト）|
| `degree_high` | Subject次数が大きいトリプルを優先 |
| `degree_low` | Subject次数が小さいトリプルを優先 |
| `hop_high` | 編集Subject間のホップ距離が大きくなるよう選択 |
| `hop_low` | 編集Subject間のホップ距離が小さくなるよう選択 |

#### 6.2 4モード並列比較

```bash
./run/run_all_selection_modes.sh --num-steps 100 --seed 42

# 比較プロットのみ再生成
./run/compare_results.sh --base-dir outputs/sequential_comparison
```

#### 6.3 Degree Exclusive（排他的 subject 分割）

degree_high と degree_low の subject が重複しないよう分割して並列実行：

```bash
./run/run_degree_high_low_exclusive.sh --num-steps 100

# 多試行版（信頼区間付き）
./run/run_degree_exclusive_multi_trial.sh --num-trials 10 --p-extreme 0.10
```

#### 6.4 Degree Binned 分析

全トリプルを次数順にビン分割して各ビンで並列実験：

```bash
./run/run_degree_binned_analysis.sh --num-steps 1000 --max-bins 5
```

#### 6.5 Hop Multi-Trial 分析

hop_high / hop_low を複数シードで多試行：

```bash
./run/run_hop_multi_trial.sh --num-trials 10 --num-steps 100
```

#### 6.6 Subject Repetition 多試行

同一 subject への編集繰り返し回数（×1/×2/×4）を多試行比較：

```bash
./run/run_subject_repetition_multi_trial.sh --num-trials 10 --num-steps 100
```

#### 6.7 Hop 距離分析

degree 多試行実験の結果を hop 距離別に分析：

```bash
./run/run_hop_analysis.sh \
    --base-dir outputs/degree_exclusive_multi_trial_2 \
    --num-trials 10
```

---

## GPU 負荷分散

並列実行スクリプトはすべて **cuda:0 / cuda:1** に自動で分散します：

| スクリプト | cuda:0 | cuda:1 |
|---|---|---|
| `run_all_selection_modes.sh` | degree_high, hop_high | degree_low, hop_low |
| `run_degree_binned_analysis.sh` | 偶数 bin | 奇数 bin |
| `run_degree_high_low_exclusive.sh` | degree_high | degree_low |
| `run_degree_exclusive_multi_trial.sh` | 偶数 trial | 奇数 trial |
| `run_hop_multi_trial.sh` | 偶数 trial | 奇数 trial |
| `run_subject_repetition_multi_trial.sh` | 偶数 trial | 奇数 trial |

---

## ディレクトリ構造

```
kekg/
├── run/                              # 実行スクリプト
│   ├── run_training.sh               # モデル学習
│   ├── visualize_kg.sh               # KG可視化
│   ├── run_locating.sh               # Causal Tracing
│   ├── run_editing.sh                # 単発 ROME 編集
│   ├── run_ripple_analysis.sh        # N件単発編集 + Ripple 分析
│   ├── run_sequential_edits.sh       # 逐次編集（1モード）
│   ├── run_sequential_analysis.sh    # 逐次編集結果の可視化
│   ├── run_all_selection_modes.sh    # 4モード並列比較
│   ├── compare_results.sh            # 比較プロット再生成
│   ├── run_degree_high_low_exclusive.sh  # degree 排他的比較（2モード）
│   ├── run_degree_binned_analysis.sh # 次数ビン別並列分析
│   ├── run_degree_exclusive_multi_trial.sh  # degree 多試行比較
│   ├── run_hop_multi_trial.sh        # hop 多試行比較
│   ├── run_subject_repetition_multi_trial.sh  # subject 繰り返し多試行
│   └── run_hop_analysis.sh           # hop 距離別分析
│
├── configs/                          # 設定ファイル（YAML）
│   ├── train_gpt_small.yaml          # モデル学習設定
│   └── edit_rome.yaml                # ROME 編集設定
│
├── src/
│   ├── kg/                           # KG 生成
│   │   └── generator.py              # BA KG 生成
│   ├── modeling/                     # モデル定義・学習
│   │   ├── gpt_mini.py               # GPTMini
│   │   ├── tokenizer.py              # SROTokenizer
│   │   └── trainer.py                # Trainer
│   ├── edit/                         # 知識編集（ROME）
│   ├── sequential_edit/              # 逐次編集実験
│   ├── scripts/                      # 分析・可視化スクリプト
│   └── cli/                          # CLI エントリポイント
│
├── data/
│   └── kg/ba/                        # KG データ（gitignore）
│
├── tests/                            # テスト
└── outputs/                          # 実験結果出力（gitignore）
```

---

## テスト

EasyEdit conda 環境でテストを実行：

```bash
/opt/conda/envs/EasyEdit/bin/python -m pytest tests/ -v
```

| テストファイル | 内容 |
|---|---|
| `test_kg_generation.py` | KG 生成・構造・再現性 |
| `test_knowledge_graph.py` | `Triple.from_dict()` / `KnowledgeGraph` |
| `test_tokenizer.py` | `SROTokenizer` エンコード・保存 |
| `test_model.py` | GPTMini forward / `SRODataset` / `compute_accuracy` |
| `test_rome_editing.py` | ROME 編集・重み変更・コピー不変 |
| `test_ripple_analysis.py` | `RippleAnalyzer` logit 計算・ripple 統計 |
| `test_sequential_editing.py` | KG BFS hop・編集ケースサンプリング（全 5 モード）・eval triple 分割 |

---

## インストール

```bash
pip install -r requirements.txt
```

---

## 参考

- ROME: Locating and Editing Factual Associations in GPT (Meng et al., 2022)
