# sol

A small Wayland tiling compositor written in Rust. Master-stack layout, vim-like keybinds, spring-physics tile animations, frosted-glass dim for inactive windows, and rounded corners.

Status: working daily driver for the author's setup (DRM/GBM/Vulkan on Linux). Not packaged, not broadly tested. Expect rough edges.

## Backends

- **DRM** — real hardware via libseat + libinput + GBM/Vulkan. Run from a free TTY.
- **Headless** — software canvas that dumps each frame as a PNG. For development and sanity-checking the protocol layer without a graphics stack.

## Build

```sh
cargo build --release
```

System libraries at runtime: `libdrm`, `libinput`, `libxkbcommon`, `libseat`, `libgbm`, `libvulkan`, plus a Vulkan ICD for your GPU (Mesa's `vulkan-radeon` / `vulkan-intel`, or `nvidia-utils`). At build time you also need `glslc` from the `shaderc` package (Arch: `pacman -S shaderc`) — `build.rs` invokes it to compile the GLSL shaders to SPIR-V. seatd or systemd-logind must be running so the binary can take a session without root.

## Run

DRM, from a free VT (e.g. Ctrl+Alt+F3):

```sh
./target/release/sol --backend=drm
```

Headless (PNG dump per frame):

```sh
SOL_PNG_PATH=/tmp/sol.png ./target/release/sol --backend=headless
```

VT switching: `Ctrl+Alt+F1`–`F12` is hardcoded and intentionally not overridable.

## Config

`$XDG_CONFIG_HOME/sol/sol.conf` (or `~/.config/sol/sol.conf`). Format and full key list are documented at the top of [`crates/sol-wayland/src/config.rs`](crates/sol-wayland/src/config.rs). The file is watched at runtime — saves re-apply gaps, borders, idle timeout, keybinds, remaps, spring tuning, and the inactive-window blur live, without a restart.

## Default keybinds

| Binding | Action |
|---|---|
| `Alt+Return` | spawn alacritty |
| `Alt+D` | spawn rofi (drun) |
| `Alt+H/J/K/L` | move keyboard focus |
| `Ctrl+Alt+H/J/K/L` | move the focused tile through the layout |
| `Alt+Tab` | toggle zoom (focused tile fills the usable area, gaps respected) |
| `Ctrl+Tab` | toggle fullscreen (covers waybar, no gaps, no border, no rounded corners) |
| `Alt+R` | modal resize: `H`/`L` adjust the master split, `Esc` exits |
| `Alt+Q` | close focused window |
| `Alt+Y/U/I/O/P` | switch to workspace 1..5 |
| `Ctrl+Alt+Y/U/I/O/P` | move focused window to workspace 1..5 |
| `Super+Left-drag` | move floating windows that don't implement `xdg_toplevel.move` |

Zoom and fullscreen are remembered per workspace — leaving and returning restores the same view. CapsLock is remapped to Escape by default. All bindings are overridable in `sol.conf`.

## Animations

Per-tile motion is spring-driven (no time-based tweens): each of `x`, `y`, `w`, `h`, plus an alpha and scale spring for open / close fades, and a border-alpha spring for focus changes. Knobs in `sol.conf`:

- `spring_stiffness` / `spring_damping` — base layout spring.
- `spring_stiffness_vertical` / `spring_damping_vertical` — override for the y/h axes (right-stack reorders).
- `spring_stiffness_swap` / `spring_damping_swap` — dedicated spring while two stack tiles trade places.
- `spring_stiffness_fade` / `spring_damping_fade` — open / close fade-in-out, plus the maximize crossfade.

## Implemented protocols

`wl_compositor`, `wl_subcompositor`, `wl_shm`, `zwp_linux_dmabuf_v1` (single-plane), `wl_seat` (pointer + keyboard), `wl_output`, `wl_data_device_manager` (clipboard), `zwp_primary_selection_device_manager_v1` (middle-click paste), `xdg_wm_base` (toplevel + popup), `zxdg_decoration_manager_v1`, `zwlr_layer_shell_v1`, `wp_viewporter`, `wp_fractional_scale_v1`, `wp_presentation`, `zxdg_output_manager_v1`, `ext_workspace_v1`, `zwp_idle_inhibit_manager_v1`.

## Crates

- `sol` — binary entry point and CLI.
- `sol-core` — backend-agnostic scene types (`Scene`, `SceneElement`).
- `sol-wayland` — protocol dispatch, layout, animation, input, focus.
- `sol-backend-drm` — DRM/GBM/Vulkan presenter (textured-quad pipeline, dual-Kawase blur for inactive-window backdrops, SDF rounded-corner masking). Scan-out BOs are GBM-allocated and imported into Vulkan as `VkImage`s via `VK_EXT_image_drm_format_modifier`; GPU-completion is exported as a `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD` so the page flip defers until rendering's done.

## License

MIT.
