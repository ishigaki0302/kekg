# CLAUDE.md

This repository is the active KEKG research workspace.

Claude Code should use the same coordination system as Codex:
- Read `docs/2026-09-27-recall-summary.md` first.
- Read `claude/AGENTS.md` and `claude/WORKFLOW.md`.
- Read `docs/YYYY-MM-DD-daily-log.md` before changing files.
- Read `claude/agent-mail.md` and reply to any `## Open` item addressed to Claude Code.
- Use `claude/DAILY.md` for local/GPU main-track work.
- Use `claude/SURVEY.md` for lightweight/cloud survey work.

Channels:
- `docs/YYYY-MM-DD-daily-log.md` = work history / claims. `claude/agent-mail.md` = correspondence
  (質問・依頼・返答). Mail rules and GPU policy are in `claude/AGENTS.md`.

Coexistence rules:
- Treat uncommitted changes as user- or agent-owned unless `docs/YYYY-MM-DD-daily-log.md` says otherwise.
- Before editing a file already modified in `git status --short`, inspect the diff.
- For substantial or overlapping work, add a claim entry to `docs/YYYY-MM-DD-daily-log.md` before editing:
  `Claim: actor=Claude Code, files=<paths>, intent=<one line>`.
- Close out in `docs/YYYY-MM-DD-daily-log.md` with commands run, files changed, outputs, blockers, and next action.
- Do not revert Codex changes unless the user explicitly asks.

Current priority:
1. Mediation analysis for structure -> representation -> plasticity difficulty.
2. Enriched-covariate explanatory IRT.
3. Progress-slide updates only after results are verified.


## Git ワークフロー

- 細かい変更は検証後に main へ直接コミット・push してよい。ユーザ指示による大掛かりな仕様追加・変更は
  `feature/<説明>` / `fix/<説明>` / `chore/<説明>` ブランチで作業する（基準は `claude/WORKFLOW.md` §4）。
- ブランチで作業した場合は `gh pr create` で PR を作り、関連 Issue があれば本文に `Closes #<番号>`。マージはユーザ確認後。
- 無関係な変更を 1 つの PR にまとめない。
- コミット前の検証手順は `claude/WORKFLOW.md` §4。

## コーディング規約

- フォーマッタ `black`、リンタ `ruff`。Python 3.11 互換で書く（3.12+ 専用構文は使わない）。
- コミットメッセージは日本語可。prefix は `feat:` / `fix:` / `docs:` / `style:` / `refactor:` / `test:` / `chore:`。
