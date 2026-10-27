#!/usr/bin/env bash
set -euo pipefail

critical_pids=()
cleanup() {
    local pid
    trap - EXIT INT TERM
    for pid in "${critical_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p /agibot/data/var
Xvfb "$DISPLAY" -screen 0 1280x800x24 -nolisten tcp -ac +extension GLX &
critical_pids+=("$!")

ready=0
for ((attempt = 0; attempt < 100; attempt++)); do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "${critical_pids[0]}" 2>/dev/null; then
        echo "Xvfb exited before the display was ready." >&2
        exit 1
    fi
    sleep 0.1
done
if ((ready == 0)); then
    echo "Xvfb display did not become ready within 10 seconds." >&2
    exit 1
fi

fluxbox &
critical_pids+=("$!")
# Preserve held modifiers for MuJoCo shortcuts (Shift+Tab toggles the right UI).
x11vnc -display "$DISPLAY" -listen 127.0.0.1 -rfbport 5900 \
    -forever -shared -nopw -nomodtweak &
critical_pids+=("$!")
websockify --web=/usr/share/novnc 6080 127.0.0.1:5900 &
critical_pids+=("$!")

echo "X2 desktop: http://127.0.0.1:6080/vnc.html"
echo "Desktop only. Start official sim_mujoco, MC and AimDK separately."

# Every service is expected to stay running. Even a clean early exit is a failure.
status=0
wait -n "${critical_pids[@]}" || status=$?
echo "A desktop service stopped unexpectedly (exit $status)." >&2
if ((status == 0)); then
    status=1
fi
exit "$status"
