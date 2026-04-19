set shell := ["bash", "-cu"]

runtime := "/tmp/hyprs-run"
socket  := "wayland-1"
png     := "/tmp/hyprs-headless.png"
# AMD card on this box has the monitor attached; card1 is the NVIDIA GPU with
# no connected outputs. Override with HYPRS_DRM_DEVICE=... if you moved things.
drm_card := env_var_or_default("HYPRS_DRM_DEVICE", "/dev/dri/card2")

# list recipes
default:
    @just --list

# cargo build the whole workspace
build:
    cargo build

# cargo check the whole workspace
check:
    cargo check

# blow away target/
clean:
    cargo clean

# create the runtime dir (idempotent)
_prep:
    mkdir -p {{runtime}} && chmod 700 {{runtime}}

# remove any leftover socket / lock files
clean-socket:
    rm -f {{runtime}}/wayland-* {{runtime}}/*.lock || true

# run the compositor in the foreground (ctrl-c to stop)
run: build _prep clean-socket
    XDG_RUNTIME_DIR={{runtime}} cargo run --bin hyprs

# run the registry probe against an already-running server
probe:
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} cargo run --bin hyprs-probe

# run the paint client against an already-running server
paint:
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} cargo run --bin hyprs-paint

# open the most recent headless PNG
view:
    xdg-open {{png}}

# B1 demo: start server, enumerate globals, shut down
demo-b1: build _prep clean-socket
    #!/usr/bin/env bash
    set -euo pipefail
    XDG_RUNTIME_DIR={{runtime}} cargo run --quiet --bin hyprs > {{runtime}}/server.log 2>&1 &
    server_pid=$!
    trap "kill $server_pid 2>/dev/null; wait 2>/dev/null || true" EXIT
    sleep 1
    echo "=== probe ==="
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} cargo run --quiet --bin hyprs-probe
    echo "=== server log ==="
    cat {{runtime}}/server.log

# B2 demo: paint stripes, dump PNG
demo-b2: build _prep clean-socket
    #!/usr/bin/env bash
    set -euo pipefail
    rm -f {{png}}
    XDG_RUNTIME_DIR={{runtime}} cargo run --quiet --bin hyprs > {{runtime}}/server.log 2>&1 &
    server_pid=$!
    trap "kill $server_pid 2>/dev/null; wait 2>/dev/null || true" EXIT
    sleep 1
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} cargo run --quiet --bin hyprs-paint
    sleep 0.3
    echo "=== png: {{png}} ==="
    ls -la {{png}}
    file {{png}}
    echo "tip: 'just view' to open it"

# B3 info: list connectors/modes for the chosen DRM card (safe from inside any session)
drm-info card=drm_card:
    cargo run --bin hyprs-drm-smoke -- info {{card}}

# B3 gpu: render the checkerboard shader offscreen via GBM+EGL+GLES and write
# the result to a PNG. Works from inside a live Wayland session because it
# doesn't need DRM master — validates the whole GPU path that demo-b3 relies on.
demo-b3-gpu card=drm_card out="/tmp/hyprs-gpu.png": build
    cargo run --release --bin hyprs-drm-smoke -- gpu-render {{card}} {{out}}
    @file {{out}}
    @echo "tip: feh {{out}}  (or xdg-open)"

# B5 demo: compositor with a software cursor driven by libinput.
# Needs root for /dev/input/event* unless you're in the 'input' group; we
# elevate with sudo -E so your XDG_RUNTIME_DIR and HYPRS_* env survive.
# Must run from a free TTY (Ctrl+Alt+F2..F6). Move the mouse to verify.
demo-b5 card=drm_card seconds="12": _prep clean-socket
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${SKIP_TTY_CHECK:-}" && "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
        cat >&2 <<'EOF'
    Refusing to run: XDG_SESSION_TYPE=wayland suggests you're still inside
    a graphical session. Switch to a free TTY (Ctrl+Alt+F2..F6), log in,
    and run the same recipe there. Set SKIP_TTY_CHECK=1 to override.
    EOF
        exit 1
    fi
    echo "=== building ==="
    cargo build --release --bin hyprs --bin hyprs-paint
    echo
    echo "=== launching hyprs (needs sudo for /dev/input/event*) ==="
    echo "move the mouse — a small red-and-white circular cursor should"
    echo "follow. the process auto-exits after {{seconds}}s; ctrl+c is fine too."
    echo
    sudo -E HYPRS_DRM_DEVICE={{card}} XDG_RUNTIME_DIR={{runtime}} \
        timeout -s INT {{seconds}} ./target/release/hyprs --backend=drm \
        || echo "(hyprs exited; rc=$?)"

# B4 demo: run hyprs as a real compositor on the DRM backend, connect
# hyprs-paint, show its stripes on the physical panel. MUST be run from a
# free TTY (Ctrl+Alt+F2..F6). Server runs in background, client paints, then
# the script sleeps so you can see it, then tears everything down.
demo-b4 card=drm_card hold="6": _prep clean-socket
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${SKIP_TTY_CHECK:-}" && "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
        cat >&2 <<'EOF'
    Refusing to run: XDG_SESSION_TYPE=wayland suggests you're still inside
    a graphical session. Switch to a free TTY (Ctrl+Alt+F2..F6), log in,
    and run the same recipe there. Set SKIP_TTY_CHECK=1 to override.
    EOF
        exit 1
    fi
    # Pre-build everything up front so the server doesn't race cargo and
    # any build errors show up before we start redirecting output.
    echo "=== building ==="
    cargo build --release --bin hyprs --bin hyprs-paint
    echo "=== launching server (backend=drm, card={{card}}) ==="
    HYPRS_DRM_DEVICE={{card}} XDG_RUNTIME_DIR={{runtime}} \
        ./target/release/hyprs --backend=drm \
        > {{runtime}}/hyprs.log 2>&1 &
    server_pid=$!
    cleanup() {
        rc=$?
        kill "$server_pid" 2>/dev/null || true
        wait 2>/dev/null || true
        echo
        echo "=== server log (exit rc=$rc) ==="
        tail -200 {{runtime}}/hyprs.log 2>/dev/null || echo "<no log>"
        if [[ -f {{runtime}}/paint.log ]]; then
            echo "=== paint log ==="
            tail -40 {{runtime}}/paint.log
        fi
        exit "$rc"
    }
    trap cleanup EXIT
    # Wait up to 10s for the socket to appear.
    for _ in $(seq 1 100); do
        [[ -S {{runtime}}/{{socket}} ]] && break
        # also bail fast if the server already died
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "server exited before creating socket"
            exit 1
        fi
        sleep 0.1
    done
    if [[ ! -S {{runtime}}/{{socket}} ]]; then
        echo "server still alive but no socket after 10s"
        exit 1
    fi
    echo "socket up; launching paint"
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} \
        ./target/release/hyprs-paint > {{runtime}}/paint.log 2>&1 || true
    echo "stripes should now be on screen; holding for {{hold}}s..."
    sleep {{hold}}

# B3 demo: full DRM/GBM/GLES smoke test on a given card for N seconds.
# MUST be run from a free TTY (Ctrl+Alt+F2..F6) — Hyprland holds DRM master
# on the active VT. Prints a clear error if master can't be acquired.
demo-b3 card=drm_card seconds="8": build
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "${SKIP_TTY_CHECK:-}" && "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
        cat >&2 <<'EOF'
    Refusing to run: XDG_SESSION_TYPE=wayland suggests you're still inside
    a graphical session. Switch to a free TTY (Ctrl+Alt+F2..F6), log in,
    and run the same recipe there. Set SKIP_TTY_CHECK=1 to override.
    EOF
        exit 1
    fi
    cargo run --release --bin hyprs-drm-smoke -- run {{card}} {{seconds}}

# B2 verification: run demo-b2, then sample four pixels against expected colors
verify-b2: demo-b2
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v magick >/dev/null; then
        echo "need imagemagick (magick) for pixel verify" >&2
        exit 1
    fi
    declare -A want=(
        [100,100]="32,32,40"
        [900,440]="255,48,48"
        [900,520]="48,255,48"
        [900,630]="48,48,255"
    )
    rc=0
    for pos in "${!want[@]}"; do
        got=$(magick {{png}} -format "%[pixel:p{$pos}]" info: | grep -oE '[0-9]+,[0-9]+,[0-9]+' | head -1)
        if [[ "$got" == "${want[$pos]}" ]]; then
            echo "ok   $pos = $got"
        else
            echo "FAIL $pos = $got, want ${want[$pos]}" >&2
            rc=1
        fi
    done
    exit $rc
