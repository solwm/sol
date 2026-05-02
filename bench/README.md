# sol-bench

Scriptable benchmark harness for sol. Runs against a sol process built with the `debug-ctl` feature; drives it through a list of commands; reads the metrics dump back over the same socket.

## Build

```sh
cargo build --release --features debug-ctl
```

The feature is off by default — production builds carry no extra dependencies and no extra socket.

## Run

From a free VT (DRM backend exercises the real GPU pipeline):

```sh
./target/release/sol --backend=drm
```

`sol` will log a line like `debug-ctl listening path=/run/user/1000/sol-ctl-wayland-1.sock` at startup.

From any TTY (or another VT, or SSH'd in via `ssh -t`), in the project root:

```sh
./bench/sol-bench --script bench/scripts/spawn_and_churn.json
```

`sol-bench` finds the control socket automatically via `XDG_RUNTIME_DIR/sol-ctl-*.sock`. Override with `--socket /path` or `SOL_CTL_SOCKET=...` if multiple sols are running.

For a longer correctness/stability run that doesn't kill sol at the
end (so visual artifacts can be inspected after):

```sh
./bench/sol-bench --script bench/scripts/torture_10min.json --no-shutdown > /tmp/torture-metrics.json
```

This churns spawn / move / focus / workspace / zoom / fullscreen /
close for ~10 minutes, snapshots metrics every iteration, then
exits leaving sol running.

The script is JSON: a flat list of commands. Each command is a dict. sol-known commands (`focus`, `move`, `workspace`, `move_to_workspace`, `toggle_zoom`, `toggle_fullscreen`, `close`, `spawn`, `snapshot`, `shutdown`) get forwarded to the compositor; harness-local primitives:

- `{"sleep_ms": N}` — pause N ms (lets sol settle / animations complete).
- `{"repeat": N, "do": [...]}` — run a block N times.
- `{"comment": "..."}` — silently skipped, harmless documentation.

The harness sends a final `shutdown` command so sol exits cleanly and dumps its metrics. Use `--no-shutdown` to leave it running and just capture a `snapshot`.

## Output

One JSON object on stdout — the cumulative metrics over sol's lifetime up to shutdown:

```json
{
  "uptime_ms": 14207,
  "frames_rendered": 312,
  "page_flips": 311,
  "render_tick_total_ns": 184_000_000,
  "render_tick_max_ns": 4_500_000,
  "frame_time_ms_buckets": [40, 120, 110, 28, 12, 2, 0, 0],
  "spring_ticks": 312,
  "input_events": 0,
  "ctl_commands": 372,
  "clients_connected": 4,
  "mapped_toplevels": 4,
  "mapped_dialogs": 0
}
```

Buckets are frame-time histogram in ms: `[<2, <3, <4, <5, <8, <16, <33, >=33]`. Tuned for 240Hz (4.16ms) and 144Hz (6.94ms) — buckets 0-3 fit the 240Hz budget, bucket 4 fits 144Hz, buckets 5+ are budget overruns.

## What's exercised, what isn't

- **Real path:** every command goes through the same internal entry point as a keybind (`focus_direction`, `switch_workspace`, …). Layout, springs, scene collection, GL pipeline, page flip — all real.
- **Spawn:** real Wayland clients (alacritty, weston-simple-shm, anything you put in the script). Captures end-to-end startup latency via the metrics gap.
- **Not exercised:** libinput latency. Synthetic moves bypass the evdev → libinput → calloop hop. Numbers from this harness are a lower bound on input-to-pixel time, not the user-perceived figure.

## When to use which backend

- DRM — real numbers. Run from a free VT. Resource cost: blocks the GPU until shutdown.
- Headless — repeatable, but skips the GPU pipeline (no texture upload, no blur, no page flip). Useful for layout / spring / focus regression tests; less so for performance.
