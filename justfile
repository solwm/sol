set shell := ["bash", "-cu"]

runtime := "/tmp/hyprs-run"
socket  := "wayland-1"
png     := "/tmp/hyprs-headless.png"

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
