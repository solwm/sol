# sol

A small Wayland tiling compositor written in Rust. Master-stack layout, vim-like keybinds, animated tile transitions, and a frosted-glass effect for inactive windows.

Status: a working daily driver for the author's setup (DRM/GBM/GLES on Linux). Not packaged, not tested broadly. Expect rough edges.

## Backends

- **DRM** — real hardware via libseat + libinput + GBM/EGL/GLES2. Run from a free TTY.
- **Headless** — software canvas that dumps each frame as a PNG. Useful for development and sanity-checking the protocol layer without a graphics stack.

## Build

```sh
cargo build --release
```

System libraries needed: `libdrm`, `libinput`, `libxkbcommon`, `libseat`, `libegl1`, `libgbm`. seatd or systemd-logind must be running so the binary can take a session without root.

## Run

DRM, from a free VT (Ctrl+Alt+F3 for example):

```sh
./target/release/sol --backend=drm
```

Headless (PNG dump per frame):

```sh
SOL_PNG_PATH=/tmp/sol.png ./target/release/sol --backend=headless
```

VT switching: `Ctrl+Alt+F1`–`F12` is hardcoded and intentionally not overridable.

## Config

`$XDG_CONFIG_HOME/sol/sol.conf` (or `~/.config/sol/sol.conf`). Format and full key list are documented at the top of [`crates/sol-wayland/src/config.rs`](crates/sol-wayland/src/config.rs). The file is watched at runtime — saves re-apply gaps, borders, idle timeout, keybinds, and remaps without a restart.

## Default keybinds

| Binding | Action |
|---|---|
| `Alt+Return` | spawn alacritty |
| `Alt+D` | spawn rofi (drun) |
| `Alt+H/J/K/L` | move keyboard focus |
| `Alt+Ctrl+H/J/K/L` | swap the focused tile with its neighbour |
| `Alt+Tab` | toggle zoom (focused tile fills usable area) |
| `Alt+Q` | close focused window (`xdg_toplevel.close`) |
| `Alt+1`…`9` | switch to workspace N |
| `Alt+Shift+1`…`9` | move focused window to workspace N |

CapsLock is remapped to Escape by default. All of the above is overridable in `sol.conf`.

## Implemented protocols

`wl_compositor`, `wl_subcompositor`, `wl_shm`, `zwp_linux_dmabuf_v1` (single-plane), `wl_seat` (pointer + keyboard), `wl_output`, `wl_data_device_manager` (clipboard), `zwp_primary_selection_device_manager_v1` (middle-click paste), `xdg_wm_base` (toplevel + popup), `zxdg_decoration_manager_v1`, `zwlr_layer_shell_v1`, `wp_viewporter`, `wp_fractional_scale_v1`, `wp_presentation`, `zxdg_output_manager_v1`, `ext_workspace_v1`, `zwp_idle_inhibit_manager_v1`.

## Crates

- `sol` — binary entry point and CLI.
- `sol-core` — backend-agnostic scene types (`Scene`, `SceneElement`).
- `sol-wayland` — protocol dispatch, layout, animation, input, focus.
- `sol-backend-drm` — DRM/GBM/EGL/GLES2 presenter (textured-quad pipeline, dual-Kawase blur for inactive-window backdrops).

## License

BSD-3-Clause.
