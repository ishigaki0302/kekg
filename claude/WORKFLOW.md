# WORKFLOW.md — デイリータスクの進め方

KEKG のデイリータスクを Claude Code / Codex / ユーザで回すための運用ルール。
細かい手順は `claude/DAILY.md`（ローカル/GPU）と `claude/SURVEY.md`（サーベイ）を参照。

## 1. 体制

| トラック | 担当 | 実行 | 中身 |
|---|---|---|---|
| メイン収束（朝） | Codex | ローカル cron `0 9 * * *` → `run_daily.sh` | `claude/DAILY.md` の優先キュー。GPU 不可（CPU 解析・整備） |
| メイン収束（夜） | **Claude Code (Sonnet)** | ローカル cron `0 21 * * *` → `AGENT=claude run_daily.sh` | GPU が必要なジョブ（学習・eval）を優先して投入 |
| 発散サーベイ | Claude クラウドルーティン（Sonnet 5, 08:00/20:00 JST） | クラウド | `knowledge_base_lab` の main に直接 push（PR は作らない） |

- Codex の cron サンドボックスからは GPU が見えない（`nvidia-smi` が失敗する）。
  GPU が必要なタスクは Codex がコマンドと完了条件を残し、夜の Claude (Sonnet) 回で拾う。
- 使用量上限対策として、定期実行の Claude は Sonnet を使う（`CLAUDE_MODEL` で変更可、既定 `sonnet`）。

## 2. ファイル配置

- **指示用 md は `claude/` に集約し、git で push する。**
  `AGENTS.md`（共通指示）/ `CLAUDE.md`（Claude 入口）/ `DAILY.md` / `SURVEY.md` /
  `WORKFLOW.md`（本書）/ `agent-mail.md`（エージェント間の文通）。
- ルートの `CLAUDE.md` / `AGENTS.md` は自動読込用のポインタだけにする（中身を書かない）。
- **細かいデイリーレポート・ログは `docs/` に置く**（gitignored・push しない）。
  - 1 日 1 ファイル：`docs/YYYY-MM-DD-daily-log.md`（JST 日付）。単一ファイルにまとめない。
  - 無ければ作る。新しいエントリはファイルの先頭側に追加する。
  - その他の日付付きレポート（進捗・サーベイまとめ等）も `docs/YYYY-MM-DD-<topic>.md`。
- 実行ログ・中間成果物は `outputs/`（gitignored）。

## 3. 1 回のデイリー実行の流れ

1. **読む**：`docs/2026-09-27-recall-summary.md` → `claude/AGENTS.md` → runbook
   → 直近数日の `docs/*-daily-log.md`（新しい順）→ `claude/agent-mail.md` の自分宛 Open。
2. **状態確認**：`git status --short`、`nvidia-smi`、走行中ジョブ（`pgrep`）、成果物の件数。
3. **Claim**：作業前に当日のログへ
   `Claim: actor=<name>, files=<paths>, intent=<one line>` を書く。
4. **実行**：小さく検証可能な単位で進める。長時間ジョブはコマンド・ログパス・
   完了条件（例 `ls ... | wc -l == N`）・PID を記録する。
5. **GPU を遊ばせない**：2 枚とも最大稼働。離脱前に次の resumable なジョブを必ず投入する。
6. **Closeout**：当日のログに、読んだもの・実行コマンド・変更ファイル・結果・blocker・
   次アクションを書く。見出しは `## YYYY-MM-DD HH:MM JST — <actor> — <LOCAL|SURVEY|ADMIN>`。

## 4. コード変更とコミット

- 他エージェントやユーザの未コミット変更を revert / 上書きしない。編集前に diff を見る。
- コミットしてよいのは**差分を検証してから**：
  - 差分を読んでレビューする（バグ・名前衝突・既定挙動の変化）。
  - `py_compile` を通し、変更したスクリプトを小さいサブサンプルでスモーク実行する。
  - 既存出力と結果が整合するか確認する。
- コミットに含めないもの：無関係なバイナリ（zip 等）、`outputs/` `data/` `docs/`。
- コミットメッセージは日本語で要点を書き、作業ブランチに push する（main へ直接 push しない）。

## 5. 連絡

- 作業履歴・claim は `docs/YYYY-MM-DD-daily-log.md`。
- 質問・依頼・申し送りは `claude/agent-mail.md`（`mail-NNN`、Open/Closed）。
  起動時に自分宛の Open を先に処理する。
- 判断が要るもの（方針の分岐・外部公開・破壊的操作）はユーザに確認する。
