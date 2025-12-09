# ROME Locating Visualization

ROMEリポジトリのcausal tracingコードを参考に実装した可視化機能です。

## 実装した機能

### 1. 個別トレースのヒートマップ (ROME スタイル) - 3種類のコンポーネント

**ファイル**: `run_locating_batch.py` の `plot_heatmap()` 関数

**特徴**:
- ROMEの `causal_trace.py` の `plot_trace_heatmap()` を参考に実装
- **3種類のコンポーネント別トレース**:
  - **All states (紫色)**: TransformerBlock全体の効果
  - **MLP (緑色)**: Feed-Forward Network (FFN) のみの効果
  - **Attention (赤色)**: Multi-Head Attention のみの効果
- トークン x レイヤーのヒートマップ
- Subject トークンに `*` マークを付加
- 確率値 `p(answer)` をカラーバーに表示
- 低スコア（corrupted input）を vmin として使用

**出力例** (各サンプルで3つのPNG):
- `outputs/locating_results/trace_000_E_008_R_27.png` (全体 - 紫)
- `outputs/locating_results/trace_000_E_008_R_27_mlp.png` (MLP - 緑)
- `outputs/locating_results/trace_000_E_008_R_27_attn.png` (Attention - 赤)

### 2. 集約統計分析プロット

**ファイル**: `run_locating_batch.py` の `plot_aggregate_statistics()` 関数

**含まれるプロット** (6つのサブプロット):

1. **Best Layer の分布** (ヒストグラム)
   - どのレイヤーが最も重要かの頻度分布

2. **Effect Magnitude vs Layer** (散布図)
   - レイヤーごとの効果の大きさ
   - カラーマップで効果の強さを可視化

3. **Average Effect per Layer** (エラーバー付き折れ線グラフ)
   - レイヤーごとの平均効果と標準偏差
   - どのレイヤーが平均的に重要かを示す

4. **Clean vs Corrupted Scores** (対数スケール散布図)
   - 正常入力と破損入力でのスコアの比較
   - y=x の対角線も表示

5. **Cumulative Distribution of Effects** (累積分布関数)
   - 効果の大きさの累積分布
   - 中央値を赤線で表示

6. **Summary Statistics** (テキスト表示)
   - サンプル数、成功数
   - Best Layer の平均、標準偏差、最頻値
   - Effect の統計量（平均、中央値、標準偏差、最大、最小）
   - スコアの統計

**出力**: `outputs/locating_results/aggregate_analysis.png`

### 3. レイヤー重要度分布プロット

**ファイル**: `run_locating_batch.py` の `create_average_heatmap()` 関数

**特徴**:
- Best layer として選ばれた頻度を棒グラフで表示
- Purples カラーマップで重要度を色分け
- 各バーに頻度の数値を表示

**出力**: `outputs/locating_results/layer_importance.png`

### 4. 平均ヒートマップ

**ファイル**: `run_locating_batch.py` の `plot_average_heatmap()` 関数

**特徴**:
- **全サンプルのヒートマップを平均化**して可視化
- ROMEの論文にある "Average Causal Effects" に相当
- 3種類のコンポーネント別に生成:
  - **All states (紫色)**: 全体的な傾向
  - **MLP (緑色)**: FFNの平均効果
  - **Attention (赤色)**: Attentionの平均効果
- サンプル数 `n` を表示
- 各レイヤー・トークン位置での平均的な因果効果を可視化

**出力** (3つのPNG):
- `outputs/locating_results/average_heatmap.png` (全体)
- `outputs/locating_results/average_heatmap_mlp.png` (MLP)
- `outputs/locating_results/average_heatmap_attn.png` (Attention)

### 5. T0 AIE（Subject Entity）レイヤー別プロット (NEW!)

**ファイル**: `run_locating_batch.py` の `plot_t0_aie_by_layer()` 関数

**特徴**:
- **T0（Subject Entityの最初のトークン）位置**におけるAIE（Average Indirect Effect）を可視化
- 横軸: レイヤー番号
- 縦軸: AIE（因果効果の大きさ）
- **4つのサブプロット**:
  1. **左上**: All/MLP/Attentionを重ねて表示（比較用）
  2. **右上**: All States のみ（紫色）
  3. **左下**: MLP のみ（緑色）
  4. **右下**: Attention のみ（赤色）
- エラーバンド（標準偏差）付き
- マーカー: All=○、MLP=□、Attention=△

**解釈**:
- どのレイヤーでsubject entityの情報が最も重要か
- MLP vs Attention のどちらが主に働いているか
- ROMEの論文で示される "中間レイヤーでのピーク" を確認可能

**出力**:
- `outputs/locating_results/t0_aie_by_layer.png`

## 使用方法

### 基本的な実行

```bash
# 50サンプルでlocatingを実行（デフォルト）
./run_locating.sh --num-samples 50

# 10サンプルでテスト実行
./run_locating.sh --num-samples 10

# カスタムパラメータ
./run_locating.sh \
    --model-dir outputs/models/gpt_small \
    --kg-file data/kg/ba/graph.jsonl \
    --num-samples 100 \
    --noise-level 3.0 \
    --num-noise-samples 10
```

### 出力ファイル

すべての結果は `outputs/locating_results/` に保存されます:

```
outputs/locating_results/
├── locating_summary.json              # サマリー情報（JSON）
├── aggregate_analysis.png             # 集約統計分析（6つのサブプロット）
├── layer_importance.png               # レイヤー重要度分布
│
├── average_heatmap.png                # 平均ヒートマップ (全体)
├── average_heatmap_mlp.png            # 平均ヒートマップ (MLP)
├── average_heatmap_attn.png           # 平均ヒートマップ (Attention)
├── t0_aie_by_layer.png                # T0 AIEレイヤー別プロット (NEW!)
│
├── trace_000_E_008_R_27.png          # 個別トレース #0 (全体)
├── trace_000_E_008_R_27_mlp.png      # 個別トレース #0 (MLP)
├── trace_000_E_008_R_27_attn.png     # 個別トレース #0 (Attention)
│
├── trace_001_E_003_R_09.png          # 個別トレース #1 (全体)
├── trace_001_E_003_R_09_mlp.png      # 個別トレース #1 (MLP)
├── trace_001_E_003_R_09_attn.png     # 個別トレース #1 (Attention)
└── ...
```

**注意**: 各サンプルについて3つのヒートマップが生成されるため、N サンプルで **3N + 7** 個のPNGファイルが生成されます。
- 個別トレース: 3N 個
- 平均ヒートマップ: 3 個
- T0 AIEプロット: 1 個
- 集約統計: 2 個
- その他: 1 個

## ROMEとの対応関係

| ROME の実装 | 本実装 | 説明 |
|------------|--------|------|
| `experiments/causal_trace.py` の `plot_trace_heatmap()` | `plot_heatmap()` | 個別トレースのヒートマップ |
| `experiments/causal_trace.py` の `kind` パラメータ | `CausalTracer.trace_important_states(kind=...)` | MLP/Attention 個別トレース |
| `notebooks/average_causal_effects.ipynb` | `plot_average_heatmap()` | 平均ヒートマップ |
| - | `plot_t0_aie_by_layer()` | T0 AIEレイヤー別プロット（独自拡張） |
| - | `plot_aggregate_statistics()` | 集約統計の可視化（独自拡張） |
| - | `create_average_heatmap()` | レイヤー重要度の棒グラフ（独自拡張） |

### カラーマップの対応

ROMEと同じカラースキームを使用:
- **紫 (Purples)**: 全体の hidden state
- **緑 (Greens)**: MLP/FFN
- **赤 (Reds)**: Attention

## 可視化の解釈

### ヒートマップの読み方

- **縦軸**: 入力トークンの位置（`*` はsubjectトークン）
- **横軸**: モデルのレイヤー番号
- **色の濃さ**: そのレイヤーのその位置を復元したときの効果（濃いほど効果が大きい）
- **カラーマップ**:
  - 紫色 (Purples): 全体のhidden state
  - 緑色 (Greens): MLP/FFN
  - 赤色 (Reds): Attention

### 3種類のトレースの解釈

1. **All States (紫)**: TransformerBlock全体の効果
   - MLPとAttentionの両方の効果を含む
   - 最も一般的な可視化

2. **MLP (緑)**: Feed-Forward Network のみ
   - 事実知識の保存に重要とされる
   - ROMEではMLPに知識が格納されると仮定

3. **Attention (赤)**: Multi-Head Attention のみ
   - トークン間の関係性を捉える
   - コンテキストの伝播に重要

### Aggregate Analysis の読み方

1. **Distribution of Best Layers**: knowledge が主に格納されているレイヤーの分布
2. **Effect Magnitude vs Layer**: レイヤーごとの効果のばらつき
3. **Average Effect per Layer**: 平均的にどのレイヤーが重要か（エラーバーは標準偏差）
4. **Clean vs Corrupted Scores**: モデルが正常時と破損時でどれだけ確率が変化するか
5. **Cumulative Distribution**: 効果の大きさの分布（中央値で半分のサンプルより大きい）
6. **Summary Statistics**: 数値での要約

## 参考文献

- ROME 論文: Locating and Editing Factual Associations in GPT (Meng et al., 2022)
- ROME リポジトリ: https://github.com/kmeng01/rome
  - 特に `experiments/causal_trace.py` の実装を参考
