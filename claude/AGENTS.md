# KEKG shared agent instructions

This repository is the active KEKG research workspace. These instructions are
shared by Codex, Claude Code, and any other coding agent working here.

Before doing research or code work:
- Read `docs/2026-09-27-recall-summary.md`.
- Read `docs/daily-log.md` for coordination with the user, Claude Code, and Codex.
- Read `claude/agent-mail.md` and reply to any `## Open` item addressed to you.
- Inspect `git status --short`.

Coordination:
- File layout (user rule): 指示用 md は `claude/` に集約して git 管理。細かいデイリーレポート・ログ類は `docs/` (gitignored) に置く。
- `claude/DAILY.md` is the local/GPU main-track runbook.
- `claude/SURVEY.md` is the lightweight/cloud survey runbook.
- `docs/daily-log.md` = work history (時系列ログ・完了報告・実行コマンド・結果・claim).
- `claude/agent-mail.md` = correspondence (質問・依頼・返答。未解決を見つけやすくする連絡板).
- Append a dated entry to `docs/daily-log.md` after meaningful work.
- Do not revert or overwrite uncommitted changes from the user or another agent.
- If you plan to edit a file that is already modified, inspect the diff first
  and record your intent in `docs/daily-log.md` when the work may overlap.
- If a conflict with another agent is likely, stop before editing and write the
  conflict and proposed owner to `docs/daily-log.md`.

agent-mail.md rules:
- 新しい文通は `## Open` の一番上に追加。id は `mail-NNN` の連番。
- 見出しは `### <date> — to:<agent> — from:<agent> — id:mail-NNN` + `status:` (open/answered)。
- 返答は同じ項目の `Reply:` に追記。解決したら `## Closed` に移動。
- 重要な結論だけ `docs/daily-log.md` にも要約する。
- コード変更の所有権は `docs/daily-log.md` の claim で管理する。

GPU policy (user rule):
- GPU を遊ばせない。ローカル作業中は 2 枚とも最大稼働させる
  (複数 shard/slot 並列・VRAM を埋める batch・常に重いジョブを1本キュー)。
- アイドルを見つけたら次の resumable なジョブを投入してから離脱する。

Current priority:
1. Mediation analysis for structure -> representation -> plasticity difficulty.
2. Enriched-covariate explanatory IRT.
3. Progress-slide updates only after results are verified.

Headless/local runs:
- Codex local run: `./run_daily.sh`
- Codex survey run: `./run_daily.sh --survey`
- Claude Code can use the same runbooks directly: read `claude/CLAUDE.md`, then
  `claude/DAILY.md` or `claude/SURVEY.md`, and append to `docs/daily-log.md`.
