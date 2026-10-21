#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

mkdir -p .pids .logs
export PYTHONPATH="$DIR"

start_daemon() {
    local name="$1"
    local script="$2"
    local port="$3"

    if [ -f ".pids/$name.pid" ] && kill -0 "$(cat ".pids/$name.pid")" 2>/dev/null; then
        echo "  [$port] $name already running (PID: $(cat ".pids/$name.pid"))"
    else
        setsid python3 "$script" "$port" </dev/null >".logs/$name.log" 2>&1 &
        echo $! > ".pids/$name.pid"
        echo "  [$port] $name started (PID: $(cat ".pids/$name.pid"))"
    fi
}

echo "Starting zGPU Services..."
start_daemon "server" "llm/server.py" "8880"
start_daemon "dashboard" "dashboard/server.py" "9000"
start_daemon "compiler" "compiler/service.py" "9100"
start_daemon "worker" "cluster/worker/cpu_worker.py" "9200"

echo "Waiting for services to initialize..."
sleep 1
