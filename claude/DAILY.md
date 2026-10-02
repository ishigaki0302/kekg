# DAILY.md — local main-track runbook

Purpose: advance the main KEKG research line locally, using the current repo
state and GPU when useful. This runbook is shared by Codex and Claude Code.

Primary context:
- Read `docs/2026-09-27-recall-summary.md` first.
- Treat `docs/2026-06-29-experiment-design-locked.md` as the experimental design source of truth.
- Use `docs/2026-07-01-kg-design.md` for symbolic-world design decisions.
- Use `docs/daily-log.md` as the coordination log with the user, Claude Code, and Codex.

Current main track:
1. Finish the mediation-analysis path: structure -> internal representation -> plasticity difficulty.
2. Re-fit explanatory IRT on `outputs/plasticity/responses_matrix_enriched.csv` after checking collinearity.
3. Update `docs/2026-06-29-progress-slides.md` only after results are verified.

Operating rules:
- Start by reading recent `docs/daily-log.md` entries and `git status --short`.
- Never revert or overwrite uncommitted changes unless the log explicitly says they are yours and should be replaced.
- If another agent appears to be editing the same file, stop and write the conflict to `docs/daily-log.md`.
- When starting a substantial task, append a short "claim" entry to
  `docs/daily-log.md` naming the actor, track, files likely to be touched, and the
  intended next step.
- When finishing or pausing, append a closeout entry. Include enough detail that
  another agent can resume without redoing context gathering.
- Prefer small, verifiable steps over large refactors.
- GPU policy (user rule): GPU を遊ばせない。ローカル作業中は 2 枚とも最大稼働
  させる (複数 shard/slot 並列・VRAM を埋める batch・常に重いジョブを1本キュー)。
  離脱前に必ず次の resumable なジョブを投入する。
- Before launching long GPU jobs, verify inputs and write the exact command to `docs/daily-log.md`.
- Long jobs should be resumable and should write logs under `outputs/`.
- At the end, append a dated entry to `docs/daily-log.md` with: context read, commands run, files changed, results, blockers, and next action.

Suggested first tasks:
1. Inspect the current diff in `src/scripts/fit_irt_scalable.py`.
2. Run or dry-run `src/scripts/mediation_analysis.py` only after confirming required `outputs/plasticity/repr/*_repr.csv` coverage.
3. If mediation inputs are incomplete, identify the missing representation files and the script/command needed to generate them.
4. If editing IRT code, make coefficient names match the expanded structural design matrix.
