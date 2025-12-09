# Ripple Effect Analysis

単発編集におけるRipple Effectの大きさ（logitの変化）を次数とhop数ごとに分析するスクリプトです。

## 概要

このスクリプトは以下の処理を自動で実行します：

1. **テストケース選択**: 知識グラフから10個の多様な編集ケースを選択
2. **編集実験実行**: 各ケースに対してROME編集を実行し、Ripple Effectを分析（各実験で1000トリプルを分析）
3. **結果分析と可視化**: 次数とhop距離（Subject基準、Before-Object基準、After-Object基準）ごとにlogit変化をプロット

## 実行方法

```bash
./run_ripple_analysis.sh
```

### 前提条件

- 学習済みモデル: `outputs/models/gpt_small/`
- 知識グラフ: `data/kg/ba/graph.jsonl`, `data/kg/ba/corpus.base.txt`
- Python環境: numpy, matplotlib, seaborn, networkx

### 実行時間

約10-15分（GPUありの場合）

## 出力ファイル

### 1. テストケース
- `outputs/ripple_test_cases.json`: 選択された10個の編集ケース

### 2. 個別実験結果（10個）
各実験につき以下のファイルが生成されます：

- `outputs/ripple_exp_N/edit_result.json`: 編集結果
- `outputs/ripple_exp_N/ripple_analysis.json`: Ripple Effect分析データ
- `outputs/ripple_exp_N/locating_result.png`: Causal Tracing可視化
- `outputs/ripple_exp_N/editing_result.png`: 編集結果可視化
- `outputs/ripple_exp_N/ripple_graph.png`: Ripple Effectのグラフ可視化
- `outputs/ripple_exp_N/ripple_stats.png`: Ripple Effect統計
- `outputs/ripple_exp_N/ripple_top_affected.png`: 最も影響を受けたトリプル

### 3. 統合分析結果
- `outputs/ripple_effects_analysis.png`: 6つのプロットを含む統合可視化
  - 次数別Ripple Effect
  - Subject基準hop距離別Ripple Effect
  - Before-Object基準hop距離別Ripple Effect
  - After-Object基準hop距離別Ripple Effect
  - 次数 vs Ripple Effectの散布図とトレンドライン
  - 3つのhop基準の比較
- `outputs/ripple_effects_stats.json`: 統計サマリー

## 分析内容

### 1. 次数（Degree）分析
- ノードの次数（入次数+出次数）とRipple Effectの関係を分析
- 次数範囲: 0-5, 5-10, 10-15, 15-20, 20-30, 30-50, 50+

### 2. Hop距離分析（3つの基準）

#### Subject基準
編集されたSubjectからのhop距離に基づく影響度測定

#### Before-Object基準
編集前のObjectからのhop距離に基づく影響度測定

#### After-Object基準
編集後のObjectからのhop距離に基づく影響度測定

## カスタマイズ

### テストケース数を変更

`run_ripple_analysis.sh`内の以下を変更：
```bash
for i in {1..10}; do  # 10を任意の数に変更
```

また、テストケース選択部分の`if len(selected) >= 10:`も同じ数に変更

### 分析するトリプル数を変更

```bash
--max-ripple-triples 1000  # 1000を任意の数に変更
```

### 編集レイヤーを変更

```bash
--layer 0  # 0を他のレイヤー番号に変更（自動検出の場合は削除）
```

## トラブルシューティング

### エラー: Model not found
学習済みモデルが存在しない場合：
```bash
python -m src.cli.train_lm --config configs/train_gpt_small.yaml
```

### エラー: KG corpus not found
知識グラフが生成されていない場合：
```bash
python -m src.cli.build_kg --config configs/kg_ba.yaml
```

### メモリ不足
分析するトリプル数を減らす：
```bash
--max-ripple-triples 500  # 1000から500に減らす
```

## 結果の解釈

### Logit変化（Ripple Effect）
- 値が大きいほど、その知識トリプルへの影響が大きい
- 平均的には4-10の範囲
- 編集対象に近いトリプルほど大きい傾向

### Hop距離
- Hop 0: 編集対象に直接関連するトリプル
- Hop 1: 1ステップ離れたトリプル
- Hop 2+: より遠いトリプル
- 一般的にhop距離が増えると影響は減少

### 次数
- 次数が高いノードは多くのトリプルに接続
- 次数とRipple Effectには正の相関があることが多い

## 参考

- ROME論文: [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- プロジェクトドキュメント: `README.md`, `ROME_USAGE.md`
