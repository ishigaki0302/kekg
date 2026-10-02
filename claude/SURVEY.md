# SURVEY.md — survey runbook

Purpose: explore adjacent research directions without consuming local GPU or disturbing the main experiment pipeline.
This runbook is shared by Codex and Claude Code.

Primary context:
- Read `docs/2026-09-27-recall-summary.md` first.
- Read `docs/deep-research-report.md` only for framing and open questions.
- Use `docs/daily-log.md` for coordination.

Survey track:
1. Map recent work related to knowledge editing, ripple effects, logical propagation, IRT/psychometrics for model evaluation, and symbolic reasoning benchmarks.
2. Convert findings into short actionable notes: claim, method, dataset, result, relevance to KEKG, risk/limitation.
3. Prefer notes that can become a concrete experiment, slide, related-work paragraph, or reviewer-response argument.

Operating rules:
- Do not run GPU jobs.
- Do not modify core experiment code.
- Do not make broad literature claims without source details.
- If browsing is unavailable, write a todo with exact search queries instead of guessing.
- Append all findings to `docs/daily-log.md` under a dated `SURVEY` entry.
- If the survey creates a concrete experiment idea, end with a short
  "handoff-to-LOCAL" note and do not start GPU work from the survey track.

High-value survey questions:
1. Is there recent work after the current docs on knowledge-editing side effects by entity popularity/degree/frequency?
2. Are there stronger references for testlet or hierarchical IRT under local dependence?
3. Are there symbolic KG or logical-closure benchmarks that strengthen the controlled-world framing?
4. Are there real-benchmark transfer candidates where degree/frequency bins can be computed cleanly?
