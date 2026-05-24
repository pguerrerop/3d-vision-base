#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/data/runtime"
RUNNER_DIR="$RUNTIME_DIR/screen_runners"
LOG_DIR="$RUNTIME_DIR/screen_logs"

API_SESSION="sensor_api"
FRONTEND_SESSION="sensor_frontend"
FTP_SESSION="sensor_ftp"
WORKER_SESSION="sensor_25d_worker"

PIPELINE_ID="${PIPELINE_ID:-mining_steel_ball_classification_25d}"
DATA_DIR="${DATA_DIR:-data}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-1.0}"
API_PORT="${API_PORT:-8380}"
FRONTEND_PORT="${FRONTEND_PORT:-5074}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status|attach|logs> [session]

Starts the 2.5D TriSpector FTP runtime stack in detached screen sessions.

Sessions:
  $API_SESSION
  $FRONTEND_SESSION
  $FTP_SESSION
  $WORKER_SESSION

Examples:
  $(basename "$0") start
  $(basename "$0") status
  $(basename "$0") attach sensor_ftp
  $(basename "$0") logs sensor_25d_worker
  $(basename "$0") stop

Environment overrides:
  PIPELINE_ID=$PIPELINE_ID
  DATA_DIR=$DATA_DIR
  POLL_INTERVAL_SEC=$POLL_INTERVAL_SEC
EOF
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

screen_sessions() {
  screen -ls 2>/dev/null || true
}

screen_exists() {
  local session_name="$1"
  local sessions
  sessions="$(screen_sessions)"
  awk -v name="$session_name" '$1 ~ ("\\." name "$") { found = 1 } END { exit(found ? 0 : 1) }' <<<"$sessions"
}

session_log() {
  local session_name="$1"
  echo "$LOG_DIR/$session_name.log"
}

python_prelude() {
  cat <<'EOF'
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
  PYTHON_CMD="python"
else
  echo "WARNING: .venv/bin/activate not found. Using python3 from PATH."
  PYTHON_CMD="${PYTHON_CMD:-python3}"
fi
EOF
}

write_runner() {
  local session_name="$1"
  local title="$2"
  local workdir="$3"
  local command_body="$4"
  local runner="$RUNNER_DIR/$session_name.sh"
  local log_file
  log_file="$(session_log "$session_name")"

  cat >"$runner" <<EOF
#!/usr/bin/env bash
set -uo pipefail

cd "$workdir"
mkdir -p "$(dirname "$log_file")"

echo "[$title] starting at \$(date)"
echo "[$title] cwd: \$(pwd)"
echo "[$title] log: $log_file"
echo "[$title] detach with Ctrl-a then d"
echo

{
$command_body
} 2>&1 | tee -a "$log_file"

status=\${PIPESTATUS[0]}
echo
echo "[$title] exited with status \$status at \$(date)"
echo "[$title] keeping screen open for inspection. Type exit to close."
exec bash
EOF

  chmod +x "$runner"
}

start_session() {
  local session_name="$1"
  local title="$2"
  local workdir="$3"
  local command_body="$4"

  if screen_exists "$session_name"; then
    echo "$session_name already exists. Attach with: screen -r $session_name"
    return
  fi

  write_runner "$session_name" "$title" "$workdir" "$command_body"
  screen -dmS "$session_name" "$RUNNER_DIR/$session_name.sh"
  echo "Started $session_name"
}

start_all() {
  require_command screen
  require_command awk
  require_command tee
  require_command npm

  mkdir -p "$RUNNER_DIR" "$LOG_DIR"

  if [[ ! -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    echo "WARNING: .venv not found. Run setup first or ensure python3 has the required packages."
  fi
  if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
    echo "WARNING: frontend/node_modules not found. Run: cd frontend && npm install"
  fi

  start_session "$API_SESSION" "FastAPI backend" "$ROOT_DIR" "$(
    python_prelude
    cat <<'EOF'
exec "$PYTHON_CMD" scripts/run_api.py
EOF
  )"

  start_session "$FRONTEND_SESSION" "Vite frontend" "$ROOT_DIR/frontend" "$(cat <<'EOF'
exec npm run dev
EOF
  )"

  start_session "$FTP_SESSION" "TriSpector FTP runtime" "$ROOT_DIR" "$(
    python_prelude
    cat <<'EOF'
exec "$PYTHON_CMD" scripts/runtime.py start trispector_ftp --foreground
EOF
  )"

  start_session "$WORKER_SESSION" "2.5D processing worker" "$ROOT_DIR" "$(
    python_prelude
    cat <<EOF
exec "\$PYTHON_CMD" scripts/run_25d_worker.py \\
  --data-dir "$DATA_DIR" \\
  --pipeline-id "$PIPELINE_ID" \\
  --poll-interval-sec "$POLL_INTERVAL_SEC"
EOF
  )"

  echo
  status_all
  echo
  echo "Backend health: http://localhost:${API_PORT}/api/health"
  echo "Open Studio: http://localhost:${FRONTEND_PORT}/studio"
  echo "Open Operations: http://localhost:${FRONTEND_PORT}/operations"
  echo "FTP target: ftp://<this-machine-ip>:2121 (anonymous)"
}

stop_all() {
  require_command screen
  local sessions=("$API_SESSION" "$FRONTEND_SESSION" "$FTP_SESSION" "$WORKER_SESSION")
  local session_name
  for session_name in "${sessions[@]}"; do
    if screen_exists "$session_name"; then
      screen -S "$session_name" -X quit
      echo "Stopped $session_name"
    else
      echo "$session_name is not running"
    fi
  done
}

status_all() {
  require_command screen
  local sessions=("$API_SESSION" "$FRONTEND_SESSION" "$FTP_SESSION" "$WORKER_SESSION")
  local session_name
  for session_name in "${sessions[@]}"; do
    if screen_exists "$session_name"; then
      echo "running  $session_name"
    else
      echo "stopped  $session_name"
    fi
  done
}

attach_session() {
  local session_name="${1:-}"
  if [[ -z "$session_name" ]]; then
    echo "Choose one session to attach:"
    status_all
    exit 2
  fi
  if ! screen_exists "$session_name"; then
    echo "No running screen session named $session_name" >&2
    exit 1
  fi
  exec screen -r "$session_name"
}

show_logs() {
  local session_name="${1:-}"
  local lines="${LINES_TO_SHOW:-120}"
  if [[ -z "$session_name" ]]; then
    for log_file in "$LOG_DIR"/*.log; do
      [[ -f "$log_file" ]] || continue
      echo "==> $log_file <=="
      tail -n "$lines" "$log_file"
      echo
    done
    return
  fi
  local log_file
  log_file="$(session_log "$session_name")"
  if [[ ! -f "$log_file" ]]; then
    echo "No log file found for $session_name: $log_file" >&2
    exit 1
  fi
  tail -n "$lines" "$log_file"
}

main() {
  local action="${1:-start}"
  local target="${2:-}"

  case "$action" in
    start)
      start_all
      ;;
    stop)
      stop_all
      ;;
    restart)
      stop_all
      start_all
      ;;
    status)
      status_all
      ;;
    attach)
      attach_session "$target"
      ;;
    logs)
      show_logs "$target"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
