#!/usr/bin/env bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Stopping zGPU Services..."
for svc in server dashboard compiler worker; do
    if [ -f ".pids/$svc.pid" ]; then
        PID="$(cat ".pids/$svc.pid")"
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null && echo "  Stopped $svc (PID: $PID)"
        fi
        rm -f ".pids/$svc.pid"
    fi
done

fuser -k 8880/tcp 9000/tcp 9100/tcp 9200/tcp 2>/dev/null || true
echo "All services stopped."
