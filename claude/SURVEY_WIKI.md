# SURVEY_WIKI.md — knowledge_base_lab 発散サーベイ（ローカル cron 用プロンプト）

`run_survey.sh` が `claude -p --model sonnet` にこのファイルを渡す（08:00 / 20:00 JST）。
カレントディレクトリは `knowledge_base_lab` の作業用 worktree（origin/main から切った専用ブランチ）。
git / gh 操作は `run_survey.sh` が行う。**このプロンプトの実行者は git コマンドを使わない。**

---

あなたは KEKG 研究プロジェクトの「発散サーベイ」担当。カレントディレクトリの
`knowledge_base_lab`（LLM Wiki 形式の文献管理リポジトリ）にサーベイ結果を追加する。

## 研究の枠組み（前提）

- KEKG = 制御された合成知識グラフ世界で LLM に知識を学習させ、知識編集（KE）の効果を測る研究。
- 主線は **「KG 構造 → 内部表現 → 編集可塑性（Knowledge Plasticity）」**。
- 可塑性は 強度 / 安定度 / 波及度 / 復元力 の 4 指標。継続編集 × IRT で測る。
- 可塑性の driver は次数単独ではなく、頻度・中心性・表現・fan-out を脱相関して切り分ける。
- 「低次数の victim が脆弱」という一点に話を固定しない。

## 手順

1. `CLAUDE.md` を読み、ページ規約（フロントマター、命名、wikilink、index/log 更新）に従う。
2. `wiki/index.md`、`wiki/log.md` の直近エントリ、`wiki/queries/divergent-directions.md` を読み、
   既にカバー済みの探索軸と「次回候補軸」を把握する。
3. 今回の探索軸を **1 つ** 選ぶ（未カバー・次回候補を優先）。候補例:
   - 知識編集の副作用と entity の popularity / degree / frequency
   - ripple effects・論理的波及・多段推論への伝播
   - testlet / 階層 IRT・局所依存、モデル評価への心理測定
   - 記号的 KG / 論理閉包ベンチマーク、制御世界（合成データ）設計
   - 実ベンチマークへの転移（degree / frequency bin が計算できるもの）
   - 解釈可能性（内部表現・回路・SAE）と編集の接続
4. WebSearch / WebFetch で関連論文を **2〜3 本** 調べる。
   - arXiv ID・会議 / ジャーナル・年を **一次情報で確認できたものだけ** 書く。確認できない主張は書かない。
   - 既に `wiki/sources/` にある論文は新規作成せず、必要なら更新にとどめる。
   - 論文 PDF はこの環境に無い。フロントマターの `sources:` には arXiv 等の URL を書く。
   - 検索が使えない場合は推測で書かず、正確な検索クエリを `divergent-directions.md` に TODO として残す。
5. Wiki を更新する:
   - `wiki/sources/<kebab-case>.md`（論文要約: claim / method / dataset / result / KEKG への接続 / 限界）
   - 関連する `wiki/concepts/` 等を作成・更新し、`[[wikilink]]` を張る
   - `wiki/queries/divergent-directions.md` に今回のアイデアと次回候補軸を追記
   - `wiki/index.md` と `wiki/log.md` を更新
6. 最後に、カレントディレクトリ直下に `.survey_pr.md` を書く（git 管理外。PR に使う）:
   - 1 行目: `survey: <探索軸の短い名前> — <主な論文/概念>`（PR タイトル）
   - 2 行目: 空行
   - 3 行目以降（PR 本文）: `## 探索軸`（選んだ軸と理由）/ `## 追加・更新ページ`（表）/
     `## KEKG への接続可能性` / 具体的な実験案があれば `## handoff-to-LOCAL`

## 禁止事項

- git / gh コマンドを実行しない（ブランチ・コミット・PR は呼び出し側スクリプトが行う）。
- `raw/` を変更しない。既存ページを削除しない。
- カレントディレクトリ外のファイルを変更しない。
- GPU を使う処理・長時間ジョブを実行しない。
- 1 回の追加は小さく保つ（sources 2〜3 本程度）。
