#!/usr/bin/env bash
# Codespaces / dev container friendly launcher.
#
# Usage:
#   ./scripts/run.sh           # start server in background
#   ./scripts/run.sh stop      # stop background server
#   ./scripts/run.sh logs      # tail server log
#   ./scripts/run.sh restart   # stop + start
#   ./scripts/run.sh deps      # pip install -r requirements.txt
#
# The server survives terminal close (uses nohup). Logs go to server.log.
set -e
cd "$(dirname "$0")/.."

PID_FILE="server.pid"
LOG_FILE="server.log"
PORT="${PORT:-8000}"

# Pick a Python interpreter — prefer python3, then python.
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
    echo "✗ python interpreter not found"; exit 1
fi

ensure_uvicorn() {
    if ! "$PY" -c 'import uvicorn' 2>/dev/null; then
        echo "✗ uvicorn 미설치 — 다음 중 한 가지 실행하세요:"
        echo "   ./scripts/run.sh deps       (자동 — 권장)"
        echo "   $PY -m pip install -r requirements.txt"
        exit 1
    fi
}

install_deps() {
    echo "▶ installing dependencies via $PY ..."
    "$PY" -m pip install --user -q -r requirements.txt
    echo "✓ deps installed"
}

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "✓ server already running (PID $(cat $PID_FILE)) — http://localhost:$PORT"
        return
    fi
    ensure_uvicorn
    echo "▶ starting uvicorn on port $PORT (via $PY -m uvicorn) ..."
    # Use 'python -m uvicorn' to avoid PATH issues with the uvicorn shim.
    nohup "$PY" -m uvicorn web_server:app --host 0.0.0.0 --port "$PORT" \
        > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "✓ PID $(cat $PID_FILE) — http://localhost:$PORT"
        echo "  logs: ./scripts/run.sh logs"
        echo "  stop: ./scripts/run.sh stop"
    else
        echo "✗ start failed — see $LOG_FILE"; tail -n 30 "$LOG_FILE"
        rm -f "$PID_FILE"; exit 1
    fi
}

stop() {
    if [ -f "$PID_FILE" ]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "✓ stopped"
    else
        pkill -f 'web_server:app' 2>/dev/null && echo "✓ killed stray uvicorn" \
            || echo "(no running server)"
    fi
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    logs)    tail -f "$LOG_FILE" ;;
    deps)    install_deps ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "✓ running (PID $(cat $PID_FILE))"
        else
            echo "✗ not running"
        fi
        ;;
    *) echo "usage: $0 {start|stop|restart|logs|deps|status}"; exit 1 ;;
esac
