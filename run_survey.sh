#!/usr/bin/env bash
# knowledge_base_lab の発散サーベイをローカルで実行し、PR を出す（cron: 08:00 / 20:00 JST）。
# Claude はファイル編集と Web 検索だけを行い、git / gh 操作はこのスクリプトが行う。
# PR 作成後、要約を Slack キャンバス（CANVAS_ID、空なら無効）の先頭へ追記する。失敗してもサーベイは成功扱い。
set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KB_REPO="${KB_REPO:-/mnt/sda/ishigaki/knowledge_base_lab}"
PROMPT_SRC="${PROMPT_SRC:-$ROOT/claude/SURVEY_WIKI.md}"
MODEL="${MODEL:-sonnet}"
STAMP="$(date '+%Y%m%d-%H%M')"
BRANCH="survey/auto-$STAMP"
# worktree は kekg の外に置く（kekg の CLAUDE.md を読み込ませないため）
WT_BASE="${WT_BASE:-$(dirname "$KB_REPO")/.kb-survey-worktrees}"
WT="$WT_BASE/$STAMP"
LOG_DIR="$ROOT/outputs/agent-runs"
EVENT_LOG="$LOG_DIR/events-survey-$STAMP.jsonl"
DAILY_LOG="$ROOT/docs/$(date +%F)-daily-log.md"
CANVAS_ID="${CANVAS_ID-F08A9BHSXR7}"
CANVAS_PROMPT="${CANVAS_PROMPT:-$ROOT/claude/SURVEY_CANVAS.md}"
CANVAS_LOG="$LOG_DIR/events-survey-canvas-$STAMP.jsonl"

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  -h|--help)
    printf 'Usage: %s [--dry-run]\n' "$0"
    printf '  --dry-run  worktree を作らず、実行内容だけ表示する\n'
    exit 0
    ;;
  "") ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac

# 当日のデイリーログの先頭（見出し直後）に SURVEY エントリを追加する
log_daily() {
  python3 - "$DAILY_LOG" "$MODEL" "$1" <<'PY'
import datetime, os, sys
path, model, body = sys.argv[1:4]
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
entry = f"## {now} JST — Claude Code (survey cron, {model}) — SURVEY\n\n{body.rstrip()}\n"
s = open(path).read() if os.path.exists(path) else ""
if s.startswith("# "):
    first, _, rest = s.partition("\n")
else:
    first, rest = "# " + os.path.basename(path)[:10] + " daily log", s
rest = rest.strip("\n")
open(path, "w").write(first + "\n\n" + entry + ("\n" + rest + "\n" if rest else ""))
PY
}

# PR の要約を Slack キャンバスへ追記する（ツールは canvas の読み書きのみ許可）
post_canvas() {
  local pr_url="$1" title="$2" body="$3"
  {
    cat "$CANVAS_PROMPT"
    printf '\n## 入力\n\n- キャンバス ID: %s\n- 実行時刻: %s JST\n- PR URL: %s\n- PR タイトル: %s\n\n### PR 本文\n\n%s\n' \
      "$CANVAS_ID" "$(date '+%Y-%m-%d %H:%M')" "$pr_url" "$title" "$body"
  } | claude -p \
    --model "$MODEL" \
    --permission-mode dontAsk \
    --allowedTools "mcp__claude_ai_Slack__slack_read_canvas" "mcp__claude_ai_Slack__slack_update_canvas" \
    --output-format stream-json --verbose \
    > "$CANVAS_LOG"
  # dontAsk では許可外ツールが拒否されるだけで exit 0 になりうるため、更新の成功を event log で確認する
  # （tool_result 内では JSON 文字列として埋め込まれ、引用符が \" にエスケープされる）
  grep -qE 'canvas_url\\*":' "$CANVAS_LOG"
}

for cmd in claude git gh python3 flock; do
  command -v "$cmd" >/dev/null || { printf 'command not found: %s\n' "$cmd" >&2; exit 1; }
done

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'KB_REPO=%s\nBRANCH=%s\nWT=%s\nMODEL=%s\nPROMPT=%s\nEVENT_LOG=%s\nDAILY_LOG=%s\n' \
    "$KB_REPO" "$BRANCH" "$WT" "$MODEL" "$PROMPT_SRC" "$EVENT_LOG" "$DAILY_LOG"
  exit 0
fi

mkdir -p "$LOG_DIR" "$WT_BASE"
exec 9>"$WT_BASE/.lock"
flock -n 9 || { printf 'another survey run is in progress\n' >&2; exit 0; }

on_error() {
  log_daily "- 失敗（exit $1, line $2）。worktree: \`$WT\`, event log: \`$EVENT_LOG\`。手動確認が必要。"
}
trap 'on_error $? $LINENO' ERR

git -C "$KB_REPO" fetch --quiet origin main
git -C "$KB_REPO" worktree add --quiet -b "$BRANCH" "$WT" origin/main

cd "$WT"
claude -p \
  --model "$MODEL" \
  --permission-mode dontAsk \
  --allowedTools "Read" "Write" "Edit" "Glob" "Grep" "WebSearch" "WebFetch" \
  --output-format stream-json --verbose \
  < "$PROMPT_SRC" > "$EVENT_LOG"

PR_FILE="$WT/.survey_pr.md"
if [ -z "$(git status --porcelain -- . ':!.survey_pr.md')" ]; then
  trap - ERR
  log_daily "- 変更なしのため PR なし。event log: \`$EVENT_LOG\`"
  cd "$ROOT"
  git -C "$KB_REPO" worktree remove --force "$WT"
  git -C "$KB_REPO" branch -D "$BRANCH" >/dev/null
  exit 0
fi

if [ -s "$PR_FILE" ]; then
  TITLE="$(head -n 1 "$PR_FILE")"
  BODY="$(tail -n +3 "$PR_FILE")"
else
  TITLE="survey: auto $STAMP"
  BODY="(.survey_pr.md が生成されなかったため本文なし。event log: $EVENT_LOG)"
fi

git add -A -- . ':!.survey_pr.md'
git commit --quiet -m "$TITLE"
git push --quiet -u origin "$BRANCH"
PR_URL="$(gh pr create --base main --head "$BRANCH" --title "$TITLE" --body "$BODY")"

trap - ERR
if [ -z "$CANVAS_ID" ]; then
  CANVAS_STATUS="無効（CANVAS_ID 未設定）"
elif post_canvas "$PR_URL" "$TITLE" "$BODY"; then
  CANVAS_STATUS="追記済み（$CANVAS_ID）"
else
  CANVAS_STATUS="**失敗**（event log: \`$CANVAS_LOG\`）"
fi
log_daily "- PR: $PR_URL
- タイトル: $TITLE
- Slack キャンバス: $CANVAS_STATUS
- event log: \`$EVENT_LOG\`"

cd "$ROOT"
git -C "$KB_REPO" worktree remove --force "$WT"
printf '%s\n' "$PR_URL"
