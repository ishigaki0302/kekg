# SRO Knowledge Editing Platform

SROトリプルを学習したモデルでの知識編集とRipple Effect分析プラットフォーム

## クイックスタート

### 1. 知識グラフ生成

```bash
python -m src.cli.build_kg --config configs/kg_ba.yaml
```

### 2. モデル学習

```bash
./run_training.sh
# または
./run_training.sh --config configs/train_gpt_small.yaml
```

### 3. Locating（Causal Tracing）

```bash
./run_locating.sh
```

### 4. Editing（知識編集）

```bash
./run_editing.sh --subject E_000 --relation R_04 --target E_007 --layer 0
```

### 5. Ripple Effect分析

```bash
# デフォルト設定（10実験、v_num_grad_steps=5）
./run_ripple_analysis.sh

# カスタム設定
./run_ripple_analysis.sh [num_experiments] [v_num_grad_steps]
# 例: 20実験、各編集でROMEの勾配ステップ数を10に設定
./run_ripple_analysis.sh 20 10
```

### 6. Sequential Editing（継続知識編集）

ROMEを用いた継続知識編集とforgetting/Ripple Effectの時系列分析。

#### 6.1 単一条件での実行

```bash
# 基本的な実行（デフォルト: 100ステップ、v_num_grad_steps=20）
./run_sequential_edits.sh

# カスタム設定での実行
./run_sequential_edits.sh --num-steps 100 --num-retain-triples 1000 --v-num-grad-steps 20

# 編集知識の選択モード指定
./run_sequential_edits.sh --edit-selection-mode degree_high --output-dir outputs/seq_degree_high
./run_sequential_edits.sh --edit-selection-mode hop_low --output-dir outputs/seq_hop_low

# 可視化（時系列・hop/degreeヒートマップ・故障ステップ分布）
./run_sequential_analysis.sh

# ヘルプを表示
./run_sequential_edits.sh --help
./run_sequential_analysis.sh --help
```

#### 6.2 全条件並列実行＋比較可視化

4つの選択モード（次数大・小、hop数大・小）を並列で実行し、結果を比較：

```bash
# 全4条件を並列実行（GPUメモリに余裕がある場合）
./run_all_selection_modes.sh

# カスタム設定での並列実行
./run_all_selection_modes.sh \
    --num-steps 100 \
    --num-retain-triples 1000 \
    --base-output-dir outputs/my_comparison

# 実行中のログをモニタリング
tail -f outputs/sequential_comparison/degree_high/run.log
tail -f outputs/sequential_comparison/hop_high/run.log

# 結果の比較可視化
./compare_results.sh

# カスタム出力ディレクトリの比較
./compare_results.sh --base-dir outputs/my_comparison
```

**編集知識の選択モード:**
- `random`: ランダムサンプリング（デフォルト）
- `degree_high`: Subjectの次数が大きいトリプルを優先
- `degree_low`: Subjectの次数が小さいトリプルを優先
- `hop_high`: 編集知識同士のSubject間hop距離が大きくなるように選択（互いに離れている）
- `hop_low`: 編集知識同士のSubject間hop距離が小さくなるように選択（互いに近い）

#### 6.3 Degree-Binned分析（次数範囲ごとの検証）

全トリプルを次数順（大→小）にソートし、NUM_STEPSごとにビン分割して実験を実行。
次数範囲ごとの編集特性を比較分析します。

```bash
# デフォルト実行（1000ステップごとにビン分割、完全なビンのみ実行）
./run_degree_binned_analysis.sh

# カスタム設定での実行
./run_degree_binned_analysis.sh \
    --num-steps 1000 \
    --num-retain-triples 1000 \
    --base-output-dir outputs/degree_binned

# 最大ビン数を制限（例：最初の3ビンのみ実行）
./run_degree_binned_analysis.sh --max-bins 3

# 実行中のログをモニタリング
tail -f outputs/degree_binned_no_alias/bin_0/run.log
tail -f outputs/degree_binned_no_alias/bin_1/run.log

# ヘルプを表示
./run_degree_binned_analysis.sh --help
```

**動作:**
1. KG内の全トリプルを次数の大きい順にソート
2. NUM_STEPSごとにビンに分割（例: 1000ステップなら、bin_0=0-999, bin_1=1000-1999, ...）
3. 最後の余り（不完全なビン）は捨てる
4. 各ビンで独立した編集実験を並列実行
5. 次数範囲ごとの結果を比較プロット

**出力:**
- `outputs/degree_binned_no_alias/bin_0/` : 最高次数範囲（bin 0）の結果
  - `config.json` : ビンのメタデータ（次数範囲、トリプルインデックス範囲）
  - `stats.jsonl` : ステップごとのメトリクス
  - `plots_time_series.png` : 時系列プロット
- `outputs/degree_binned_no_alias/bin_1/` : 次の次数範囲（bin 1）の結果
- ...
- `outputs/degree_binned_no_alias/degree_bins_comparison.png` : 全ビンの比較プロット

**主要オプション:**
- `--num-steps`: 各ビンのサイズ（編集回数、デフォルト: 1000）
- `--max-bins`: 最大ビン数（デフォルト: 全ビン）
- その他のオプションは6.2と同様

#### 6.4 Degree High/Low Exclusive分析（重複なしsubject次数比較）

degree_highとdegree_lowの2条件でsubjectエンティティが重複しないように実験を実行。
公平な比較のため、高次数と低次数のエンティティを排他的にサンプリングします。

```bash
# デフォルト実行（各モードでNUM_STEPS=1000）
./run_degree_high_low_exclusive.sh

# カスタム設定での実行
./run_degree_high_low_exclusive.sh \
    --num-steps 1000 \
    --num-retain-triples 1000 \
    --base-output-dir outputs/degree_exclusive

# 実行中のログをモニタリング
tail -f outputs/degree_exclusive_no_alias/degree_high/run.log
tail -f outputs/degree_exclusive_no_alias/degree_low/run.log

# ヘルプを表示
./run_degree_high_low_exclusive.sh --help
```

**動作:**
1. KG内の全unique subjectを次数順にソート
2. NUM_STEPS * 2とunique subject数を比較
3. もしNUM_STEPS * 2 > unique subjects なら、NUM_STEPS = unique_subjects / 2に自動調整
4. degree_high: 上位NUM_STEPS個のunique subjectを使用
5. degree_low: 下位NUM_STEPS個のunique subjectを使用
6. 両モードを並列実行（subjectの重複なし保証）

**特徴:**
- ✓ 高次数と低次数のsubjectが完全に分離
- ✓ 編集対象が重複しないため、公平な比較が可能
- ✓ エンティティ数が少ない場合は自動的にNUM_STEPSを調整
- ✓ 比較プロットを自動生成

**出力:**
- `outputs/degree_exclusive_no_alias/degree_high/` : 高次数subject実験結果
  - `config.json` : 使用したsubject数、次数範囲などのメタデータ
  - `stats.jsonl` : ステップごとのメトリクス
  - `plots_time_series.png` : 時系列プロット
- `outputs/degree_exclusive_no_alias/degree_low/` : 低次数subject実験結果
- `outputs/degree_exclusive_no_alias/comparison.png` : 2条件比較プロット

#### 6.5 Hop Multi-Trial分析（初期選択依存性の統計的評価）

hopのhigh/lowは初期選択に依存するため、複数回の試行を実行して統計的に評価。
各試行を個別の線でプロットし、平均と95%信頼区間を可視化します。

```bash
# デフォルト実行（各モードで10試行）
./run_hop_multi_trial.sh

# カスタム設定での実行
./run_hop_multi_trial.sh \
    --num-trials 20 \
    --num-steps 1000 \
    --num-retain-triples 1000 \
    --base-output-dir outputs/hop_multi_trial

# 実行中のログをモニタリング
tail -f outputs/hop_multi_trial_no_alias/hop_high/trial_0/run.log
tail -f outputs/hop_multi_trial_no_alias/hop_low/trial_0/run.log

# ヘルプを表示
./run_hop_multi_trial.sh --help
```

**動作:**
1. 各モード（hop_high, hop_low）でNUM_TRIALS回の実験を実行
2. 各試行で異なるランダムシード（BASE_SEED + trial_id * 1000）を使用
3. 初期点が異なることで、選択されるトリプルセットが変わる
4. 全試行（NUM_TRIALS * 2個）を並列実行
5. 統計的な可視化を生成

**特徴:**
- ✓ 初期選択のランダム性を考慮した統計的評価
- ✓ 各試行を薄い線で表示（個別の挙動を確認）
- ✓ 平均を太い線で表示（全体的な傾向を確認）
- ✓ 95%信頼区間を塗りつぶしで表示（ばらつきを確認）
- ✓ t分布ベースの正確な信頼区間計算

**可視化内容:**
- Edit Success Rate: 各試行の線 + 平均 + 95%CI
- Retention Rate: 各試行の線 + 平均 + 95%CI
- Edited Triples Accuracy: 各試行の線 + 平均 + 95%CI
- Retain Triples Accuracy: 各試行の線 + 平均 + 95%CI
- Weight Distance: 各試行の線 + 平均 + 95%CI
- Final Metrics Comparison: バーチャート（平均 ± 95%CI）

**出力:**
- `outputs/hop_multi_trial_no_alias/hop_high/trial_0/` : hop_high 1回目の試行
- `outputs/hop_multi_trial_no_alias/hop_high/trial_1/` : hop_high 2回目の試行
- ...
- `outputs/hop_multi_trial_no_alias/hop_low/trial_0/` : hop_low 1回目の試行
- ...
- `outputs/hop_multi_trial_no_alias/multi_trial_comparison.png` : 統計的比較プロット（6パネル）

**主要オプション:**
- `--num-trials`: 各モードの試行回数（デフォルト: 10）
- `--base-seed`: ベースランダムシード（デフォルト: 24）
- その他のオプションは6.2と同様

**共通オプション（6.1-6.5）:**
- `--model-dir`: モデルディレクトリ（デフォルト: outputs/models/gpt_small）
- `--kg-dir`: KGディレクトリ（デフォルト: data/kg/ba）
- `--output-dir`: 出力ディレクトリ（デフォルト: outputs/sequential）
- `--num-steps`: 編集ステップ数（デフォルト: 100）
- `--num-retain-triples`: 未編集評価トリプル数（デフォルト: 1000）
- `--edit-selection-mode`: 編集知識選択モード（デフォルト: random）
- `--max-hop`: 最大ホップ距離（デフォルト: 10、小規模KGでは通常変更不要）
- `--v-num-grad-steps`: ROME勾配ステップ数（デフォルト: 20）
- `--seed`: ランダムシード（デフォルト: 24）
- `--device`: デバイス（cuda/cpu、デフォルト: cuda）

**出力:**
- `outputs/sequential/stats.jsonl` : stepごとのメトリクス
  - `edited_acc`: 編集済みトリプルの精度（累積、ステップNでN個）
  - `retain_acc`: 未編集トリプルの精度（常に同じM個）
- `outputs/sequential/ripple_triples.jsonl` : Δlogit＋hop/degree
- `outputs/sequential/triple_acc.jsonl` : 各トリプル×stepの正誤（edited/retain別）
- `outputs/sequential/plots_time_series.png` : 時系列プロット（5パネル）
- `outputs/sequential_comparison/comparison.png` : 4条件比較プロット（並列実行時）

**評価軸:**
1. **編集済みトリプルの精度（Edited Acc）**: 編集した知識がどれだけ成功しているか（累積評価）
2. **未編集トリプルの精度（Retain Acc）**: 編集していない知識がどれだけ保持されているか（固定M個で評価）

**注意:**
- step 0のメトリクスは編集前のベースラインを示します
- 編集済みトリプルと未編集トリプルは重複しないようサンプリングされます
- 並列実行時は各条件が独立したGPUメモリを使用するため、十分なメモリが必要です

**GPU割り当て（並列実行時）:**
全ての並列実行スクリプトは、2つのGPU（cuda:0とcuda:1）に自動的に負荷分散します：

1. **run_all_selection_modes.sh** (4条件並列)
   - cuda:0: degree_high, hop_high
   - cuda:1: degree_low, hop_low

2. **run_degree_binned_analysis.sh** (複数ビン並列)
   - cuda:0: bin_0, bin_2, bin_4, ... (偶数番号のビン)
   - cuda:1: bin_1, bin_3, bin_5, ... (奇数番号のビン)

3. **run_degree_high_low_exclusive.sh** (2条件並列)
   - cuda:0: degree_high
   - cuda:1: degree_low

4. **run_hop_multi_trial.sh** (複数試行並列)
   - cuda:0: trial_0, trial_2, trial_4, ... (偶数番号の試行、両モード)
   - cuda:1: trial_1, trial_3, trial_5, ... (奇数番号の試行、両モード)

この自動負荷分散により、2つのGPUが均等に使用され、実行時間が短縮されます。

## ディレクトリ構造

```
├── run_training.sh                   # Training実行
├── run_locating.sh                   # Locating実行
├── run_editing.sh                    # Editing実行
├── run_ripple_analysis.sh            # Ripple分析実行
├── run_sequential_edits.sh           # 継続編集実行（単一条件）
├── run_sequential_analysis.sh        # 継続編集可視化
├── run_all_selection_modes.sh        # 全選択モード並列実行
├── run_degree_binned_analysis.sh     # Degree-binned分析並列実行
├── run_degree_high_low_exclusive.sh  # Degree High/Low排他的実行
├── run_hop_multi_trial.sh            # Hop多試行分析 ★NEW
├── compare_results.sh                # 選択モード比較可視化
├── monitor_training.sh               # Training監視（オプション）
├── configs/                          # 設定ファイル
├── data/                             # データ
├── src/
│   ├── scripts/                      # 実行スクリプト
│   │   ├── run_sequential_edits.py
│   │   ├── run_degree_binned_edits.py
│   │   ├── run_degree_exclusive_edits.py
│   │   ├── compare_hop_multi_trial.py     ★NEW
│   │   ├── analyze_sequential_effects.py
│   │   ├── compare_selection_modes.py
│   │   └── compare_degree_bins.py
│   ├── cli/                          # CLIエントリ
│   ├── modeling/                     # モデル
│   ├── edit/                         # 編集手法（ROME）
│   └── sequential_edit/              # 継続編集実験
│       ├── config.py                 # 設定（選択モード追加）
│       ├── runner.py                 # 実行ロジック（選択アルゴリズム追加）
│       ├── analysis.py               # 可視化（edited/retain分離）
│       └── kg_utils.py               # KGユーティリティ
├── tests/                            # テスト
└── outputs/                          # 出力
    ├── sequential_comparison/        # 並列実行時の出力
    │   ├── degree_high/
    │   ├── degree_low/
    │   ├── hop_high/
    │   ├── hop_low/
    │   └── comparison.png
    ├── degree_binned_no_alias/       # Degree-binned分析の出力
    │   ├── bin_0/                    # 最高次数範囲
    │   ├── bin_1/                    # 次の次数範囲
    │   ├── ...
    │   └── degree_bins_comparison.png
    ├── degree_exclusive_no_alias/    # Degree High/Low排他的実行の出力
    │   ├── degree_high/              # 高次数subject実験
    │   ├── degree_low/               # 低次数subject実験
    │   └── comparison.png
    └── hop_multi_trial_no_alias/     # Hop多試行分析の出力 ★NEW
        ├── hop_high/
        │   ├── trial_0/              # 1回目の試行
        │   ├── trial_1/              # 2回目の試行
        │   └── ...
        ├── hop_low/
        │   ├── trial_0/
        │   ├── trial_1/
        │   └── ...
        └── multi_trial_comparison.png
```

## インストール

```bash
pip install -r requirements.txt
```

## 参考

- ROME: Locating and Editing Factual Associations in GPT (Meng et al., 2022)
