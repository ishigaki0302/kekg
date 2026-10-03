#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNBOOK="${RUNBOOK:-claude/DAILY.md}"
TRACK="${TRACK:-LOCAL}"
AGENT="${AGENT:-codex}"
STAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_DIR="$ROOT/outputs/agent-runs"
PROMPT_FILE="$LOG_DIR/prompt-$STAMP.txt"
EVENT_LOG="$LOG_DIR/events-$AGENT-$STAMP.jsonl"
LAST_MESSAGE="$LOG_DIR/last-$STAMP.md"

usage() {
  printf 'Usage: %s [--survey] [--agent codex|claude] [--dry-run]\n' "$0"
  printf '  default   Run Codex with claude/DAILY.md\n'
  printf '  --survey  Run the survey track with claude/SURVEY.md\n'
  printf '  --agent   Select headless agent implementation (default: codex)\n'
  printf '  --dry-run Print the generated prompt and exit\n'
}

DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --survey)
      RUNBOOK="claude/SURVEY.md"
      TRACK="SURVEY"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --agent)
      if [ "$#" -lt 2 ]; then
        printf '%s requires an argument: codex or claude\n' "$1" >&2
        exit 2
      fi
      AGENT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$AGENT" in
  codex|claude) ;;
  *)
    printf 'Unknown agent: %s\n' "$AGENT" >&2
    usage >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"

cat > "$PROMPT_FILE" <<EOF
You are $AGENT working headlessly on the KEKG research repository.

Repository: $ROOT
Track: $TRACK
Runbook: $RUNBOOK
Agent: $AGENT
Date: $(date '+%Y-%m-%d %H:%M:%S %Z')

Instructions:
1. Read docs/2026-09-27-recall-summary.md.
2. Read $RUNBOOK.
3. Read claude/WORKFLOW.md, claude/AGENTS.md and, if you are Claude Code, claude/CLAUDE.md.
4. Read the newest docs/*-daily-log.md files (latest dates first; newest entry at top of each).
5. Read claude/agent-mail.md and respond to open items addressed to you when relevant.
6. Inspect git status before changing anything.
7. Follow the runbook conservatively.
8. Do not revert user or other-agent changes.
9. Keep work scoped and verifiable.
10. Before finishing, add a concise entry at the top of docs/$(date +%F)-daily-log.md (create it with a one-line "# <date> daily log" header if missing).
11. In the final response, report what changed, what was verified, and the next action.
EOF

if [ "$DRY_RUN" -eq 1 ]; then
  cat "$PROMPT_FILE"
  exit 0
fi

cd "$ROOT"
case "$AGENT" in
  codex)
    codex --ask-for-approval never exec \
      --cd "$ROOT" \
      --sandbox workspace-write \
      --json \
      --output-last-message "$LAST_MESSAGE" \
      - < "$PROMPT_FILE" | tee "$EVENT_LOG"
    ;;
  claude)
    claude -p \
      --permission-mode dontAsk \
      --output-format stream-json --verbose \
      < "$PROMPT_FILE" | tee "$EVENT_LOG"
    ;;
esac

printf '\nAgent event log: %s\n' "$EVENT_LOG"
if [ "$AGENT" = "codex" ]; then
  printf 'Codex final message: %s\n' "$LAST_MESSAGE"
fi
