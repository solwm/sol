set shell := ["bash", "-cu"]

runtime := "/tmp/voidptr-run"
socket  := "wayland-1"
png     := "/tmp/voidptr-headless.png"
# AMD card on this box has the monitor attached; card1 is the NVIDIA GPU with
# no connected outputs. Override with VOIDPTR_DRM_DEVICE=... if you moved things.
drm_card := env_var_or_default("VOIDPTR_DRM_DEVICE", "/dev/dri/card2")

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
    XDG_RUNTIME_DIR={{runtime}} cargo run --bin voidptr

# run the registry probe against an already-running server
probe:
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} cargo run --bin voidptr-probe

# run the paint client against an already-running server
paint:
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} cargo run --bin voidptr-paint

# open the most recent headless PNG
view:
    xdg-open {{png}}

# B1 demo: start server, enumerate globals, shut down
demo-b1: build _prep clean-socket
    #!/usr/bin/env bash
    set -euo pipefail
    XDG_RUNTIME_DIR={{runtime}} cargo run --quiet --bin voidptr > {{runtime}}/server.log 2>&1 &
    server_pid=$!
    trap "kill $server_pid 2>/dev/null; wait 2>/dev/null || true" EXIT
    sleep 1
    echo "=== probe ==="
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} cargo run --quiet --bin voidptr-probe
    echo "=== server log ==="
    cat {{runtime}}/server.log

# B2 demo: paint stripes, dump PNG
demo-b2: build _prep clean-socket
    #!/usr/bin/env bash
    set -euo pipefail
    rm -f {{png}}
    XDG_RUNTIME_DIR={{runtime}} cargo run --quiet --bin voidptr > {{runtime}}/server.log 2>&1 &
    server_pid=$!
    trap "kill $server_pid 2>/dev/null; wait 2>/dev/null || true" EXIT
    sleep 1
    XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} cargo run --quiet --bin voidptr-paint
    sleep 0.3
    echo "=== png: {{png}} ==="
    ls -la {{png}}
    file {{png}}
    echo "tip: 'just view' to open it"

# B3 info: list connectors/modes for the chosen DRM card (safe from inside any session)
drm-info card=drm_card:
    cargo run --bin voidptr-drm-smoke -- info {{card}}

# B3 gpu: render the checkerboard shader offscreen via GBM+EGL+GLES and write
# the result to a PNG. Works from inside a live Wayland session because it
# doesn't need DRM master — validates the whole GPU path that demo-b3 relies on.
demo-b3-gpu card=drm_card out="/tmp/voidptr-gpu.png": build
    cargo run --release --bin voidptr-drm-smoke -- gpu-render {{card}} {{out}}
    @file {{out}}
    @echo "tip: feh {{out}}  (or xdg-open)"

# B5 demo: compositor with a software cursor driven by libinput + an
# interactive voidptr-paint window. Click the window (or press any key) to
# cycle the stripe palette. Needs root for /dev/input/event* (same caveat
# as demo-b4's DRM path). Must run from a free TTY (Ctrl+Alt+F2..F6).
demo-b5 card=drm_card seconds="20": _prep clean-socket
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
    cargo build --release --bin voidptr --bin voidptr-paint
    echo
    echo "=== launching voidptr (sudo for /dev/input/event*) ==="
    sudo -E VOIDPTR_DRM_DEVICE={{card}} XDG_RUNTIME_DIR={{runtime}} \
        ./target/release/voidptr --backend=drm \
        > {{runtime}}/voidptr.log 2>&1 &
    server_pid=$!
    cleanup() {
        rc=$?
        sudo kill "$server_pid" 2>/dev/null || true
        wait 2>/dev/null || true
        echo
        echo "=== server log (rc=$rc) ==="
        tail -200 {{runtime}}/voidptr.log 2>/dev/null || echo "<no log>"
        if [[ -f {{runtime}}/paint.log ]]; then
            echo "=== paint log ==="
            tail -60 {{runtime}}/paint.log
        fi
        exit "$rc"
    }
    trap cleanup EXIT
    for _ in $(seq 1 100); do
        [[ -S {{runtime}}/{{socket}} ]] && break
        if ! sudo kill -0 "$server_pid" 2>/dev/null; then
            echo "server exited before creating socket"
            exit 1
        fi
        sleep 0.1
    done
    [[ -S {{runtime}}/{{socket}} ]] || { echo "no socket after 10s"; exit 1; }
    echo "=== launching voidptr-paint ==="
    sudo -E XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} \
        ./target/release/voidptr-paint > {{runtime}}/paint.log 2>&1 &
    paint_pid=$!
    echo "running for {{seconds}}s. move the mouse, click the stripes,"
    echo "press keys — each should cycle the palette. ctrl+c to end early."
    sleep {{seconds}}
    sudo kill "$paint_pid" 2>/dev/null || true

# B6.1 diagnostic: start voidptr, point alacritty at it, capture both sides
# of the wire. WAYLAND_DEBUG=1 on the client dumps every request/event from
# alacritty; RUST_LOG=debug on the server dumps what voidptr saw. The point
# is NOT to make alacritty draw — it's to read the logs and learn which
# globals and requests are missing. Must run from a free TTY.
demo-b6-1 card=drm_card seconds="8": _prep clean-socket
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
    if ! command -v alacritty >/dev/null; then
        echo "alacritty not found in PATH" >&2
        exit 1
    fi
    echo "=== building ==="
    cargo build --release --bin voidptr
    echo
    echo "=== launching voidptr (sudo for /dev/input/event*) ==="
    sudo -E VOIDPTR_DRM_DEVICE={{card}} XDG_RUNTIME_DIR={{runtime}} \
        RUST_LOG="voidptr=debug,voidptr_wayland=debug,voidptr_backend_drm=debug,info" \
        WAYLAND_DEBUG=server \
        ./target/release/voidptr --backend=drm \
        > {{runtime}}/voidptr.log 2>&1 &
    server_pid=$!
    cleanup() {
        rc=$?
        sudo kill "$server_pid" 2>/dev/null || true
        wait 2>/dev/null || true
        echo
        echo "=== voidptr.log (rc=$rc, last 300 lines) ==="
        tail -300 {{runtime}}/voidptr.log 2>/dev/null || echo "<no log>"
        if [[ -f {{runtime}}/alacritty.log ]]; then
            echo
            echo "=== alacritty.log (last 300 lines) ==="
            tail -300 {{runtime}}/alacritty.log
        fi
        echo
        echo "=== full logs kept at: ==="
        echo "  {{runtime}}/voidptr.log"
        echo "  {{runtime}}/alacritty.log"
        exit "$rc"
    }
    trap cleanup EXIT
    for _ in $(seq 1 100); do
        [[ -S {{runtime}}/{{socket}} ]] && break
        if ! sudo kill -0 "$server_pid" 2>/dev/null; then
            echo "server exited before creating socket"
            exit 1
        fi
        sleep 0.1
    done
    [[ -S {{runtime}}/{{socket}} ]] || { echo "no socket after 10s"; exit 1; }
    echo "=== launching alacritty (WAYLAND_DEBUG=1) ==="
    sudo -E XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} \
        WAYLAND_DEBUG=1 \
        alacritty > {{runtime}}/alacritty.log 2>&1 &
    alacritty_pid=$!
    echo "letting it run for {{seconds}}s. ctrl+c to end early."
    sleep {{seconds}}
    sudo kill "$alacritty_pid" 2>/dev/null || true

# B6 demo: master-stack tiling. Start voidptr on DRM, then launch 4 alacritty
# instances ~2s apart so the layout visibly transitions:
#   1 window  = fullscreen
#   2 windows = master left half, second on right half
#   3 windows = master left, right split into two stacked rows
#   4 windows = master left, right split into three stacked rows
# The master is always the FIRST alacritty; new instances push the bottom of
# the right stack. Close one with ctrl+d / exit — the remaining tiles snap
# to the new layout. Must run from a free TTY.
demo-b6 card=drm_card seconds="60": _prep clean-socket
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
    if ! command -v alacritty >/dev/null; then
        echo "alacritty not found in PATH" >&2
        exit 1
    fi
    echo "=== building ==="
    cargo build --release --bin voidptr
    echo
    echo "=== launching voidptr (sudo for /dev/input/event*) ==="
    sudo -E VOIDPTR_DRM_DEVICE={{card}} XDG_RUNTIME_DIR={{runtime}} \
        RUST_LOG="voidptr=info,voidptr_wayland=info,voidptr_backend_drm=info" \
        ./target/release/voidptr --backend=drm \
        > {{runtime}}/voidptr.log 2>&1 &
    server_pid=$!
    declare -a client_pids=()
    cleanup() {
        rc=$?
        for pid in "${client_pids[@]}"; do
            sudo kill "$pid" 2>/dev/null || true
        done
        sudo kill "$server_pid" 2>/dev/null || true
        wait 2>/dev/null || true
        echo
        echo "=== voidptr.log (rc=$rc, last 200 lines) ==="
        tail -200 {{runtime}}/voidptr.log 2>/dev/null || echo "<no log>"
        exit "$rc"
    }
    trap cleanup EXIT
    for _ in $(seq 1 100); do
        [[ -S {{runtime}}/{{socket}} ]] && break
        if ! sudo kill -0 "$server_pid" 2>/dev/null; then
            echo "server exited before creating socket"
            exit 1
        fi
        sleep 0.1
    done
    [[ -S {{runtime}}/{{socket}} ]] || { echo "no socket after 10s"; exit 1; }
    for n in 1 2 3 4; do
        echo "=== launching alacritty #$n (you should see layout reflow) ==="
        sudo -E XDG_RUNTIME_DIR={{runtime}} WAYLAND_DISPLAY={{socket}} \
            alacritty -T "voidptr-$n" > {{runtime}}/alacritty-$n.log 2>&1 &
        client_pids+=($!)
        sleep 2
    done
    echo "running for {{seconds}}s. click between windows to change focus."
    echo "close any window (ctrl+d) to see the tiles rebalance."
    sleep {{seconds}}

# B4 demo: run voidptr as a real compositor on the DRM backend, connect
# voidptr-paint, show its stripes on the physical panel. MUST be run from a
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
    cargo build --release --bin voidptr --bin voidptr-paint
    echo "=== launching server (backend=drm, card={{card}}) ==="
    VOIDPTR_DRM_DEVICE={{card}} XDG_RUNTIME_DIR={{runtime}} \
        ./target/release/voidptr --backend=drm \
        > {{runtime}}/voidptr.log 2>&1 &
    server_pid=$!
    cleanup() {
        rc=$?
        kill "$server_pid" 2>/dev/null || true
        wait 2>/dev/null || true
        echo
        echo "=== server log (exit rc=$rc) ==="
        tail -200 {{runtime}}/voidptr.log 2>/dev/null || echo "<no log>"
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
        ./target/release/voidptr-paint > {{runtime}}/paint.log 2>&1 || true
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
    cargo run --release --bin voidptr-drm-smoke -- run {{card}} {{seconds}}

# B2 verification: run demo-b2, then sample four pixels against expected colors.
# Post-B6 the layout is master-stack, so with one paint window (256x256 at the
# top-left tile of a 1920x1080 screen) each stripe band is ~85 rows tall and
# the rest of the canvas is background. Positions chosen to hit each stripe
# plus the outside-window background.
verify-b2: demo-b2
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v magick >/dev/null; then
        echo "need imagemagick (magick) for pixel verify" >&2
        exit 1
    fi
    declare -A want=(
        [100,40]="255,48,48"
        [100,120]="48,255,48"
        [100,200]="48,48,255"
        [900,500]="32,32,40"
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
