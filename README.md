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
./run_ripple_analysis.sh
```

## ディレクトリ構造

```
├── run_training.sh          # Training実行
├── run_locating.sh          # Locating実行
├── run_editing.sh           # Editing実行
├── run_ripple_analysis.sh   # Ripple分析実行
├── monitor_training.sh      # Training監視（オプション）
├── configs/                 # 設定ファイル
├── data/                    # データ
├── src/
│   ├── scripts/             # 実行スクリプト
│   ├── cli/                 # CLIエントリ
│   ├── modeling/            # モデル
│   └── edit/                # 編集手法（ROME）
├── tests/                   # テスト
└── outputs/                 # 出力
```

## インストール

```bash
pip install -r requirements.txt
```

## 参考

- ROME: Locating and Editing Factual Associations in GPT (Meng et al., 2022)
