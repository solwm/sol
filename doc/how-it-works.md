# How sol works

A walk down the stack, from the wire protocol to the pixel that lands on screen. The aim isn't a reference manual — it's the trail you'd follow yourself if you opened the repo cold and asked "what actually happens when I press a key, when a window opens, when a frame is drawn?" Each section opens with the question and lands on the file/function that answers it.

The numbers in code references are line numbers as of writing — they'll drift; the function names won't. `crates/sol-wayland/src/lib.rs` is the central file; everything else is either a protocol module that hangs off it or the rendering backend.

## 1. What is a Wayland compositor, in one paragraph?

A Wayland compositor is two things glued together. It's a **server** that speaks the Wayland wire protocol over a Unix socket, accepting connections from client processes (Firefox, alacritty, waybar) and tracking their objects: surfaces, buffers, seats, outputs. And it's a **renderer** that takes the surfaces those clients hand it and composes them into one image per frame, which it then hands to the kernel via DRM/KMS for scanout. The interesting design choice is where the two halves meet — what does the protocol layer hand to the renderer? sol's answer is a flat `Scene` value (`crates/sol-core/src/lib.rs`), produced fresh on every render tick from the protocol-tracked state. The renderer never reaches back into protocol state; the protocol layer never knows about textures or framebuffers. That seam is what lets the same Wayland implementation drive both a software canvas (headless PNG dump) and a real DRM presenter.

## 2. The boot sequence

`crates/sol/src/main.rs` is a thin shim that picks a backend (`headless` or `drm`) and calls into `sol-wayland`. The substantive entry points are `run_drm` and `run_headless` (`crates/sol-wayland/src/lib.rs`, near the bottom). Both end up in `setup_event_loop`, which is where the compositor actually comes alive:

1. **Create the calloop event loop and a Wayland `Display`.** calloop is a single-threaded event loop; the `Display` is `wayland-server`'s representation of a server-side Wayland connection.
2. **Advertise globals.** Each protocol the compositor implements becomes a `wl_registry` global — `wl_compositor`, `wl_shm`, `xdg_wm_base`, `wl_seat`, `zwlr_layer_shell_v1`, `zwp_linux_dmabuf_v1`, and a dozen others (see `Globals` near the top of `setup_event_loop`). When a client connects, it walks the registry and binds to the ones it needs.
3. **Bind the listening socket.** `wayland-server`'s `ListeningSocket::bind_auto` finds a free `wayland-N` name in `$XDG_RUNTIME_DIR`. The socket name is exported as `WAYLAND_DISPLAY` in our own process env so every spawned client inherits it.
4. **Insert event sources into calloop.** This is where it stops being a sequence and starts being a reactor. The sources:
   - The Wayland listening socket — incoming client connections.
   - The display fd — pollable indication that some client has spoken.
   - The libinput fd (DRM mode only) — keyboard / pointer / touch events.
   - The DRM fd (DRM mode only) — page-flip-completion events from the kernel.
   - An inotify watch on `~/.config/sol/` for live config reload.
   - A timer for the idle-blank threshold.
5. **Run.** `event_loop.run` blocks; everything from here is event-driven.

The headless path is the same minus the libinput and DRM sources. There's no input, no scanout, just protocol traffic and a periodic PNG dump.

## 3. A client connects: the protocol dispatch shape

When a client connects, calloop's socket-source callback accepts the stream and hands it to `display.insert_client`. From that moment, every byte the client sends is parsed by `wayland-server` into a `Request` for some object the client has bound to, and dispatched to a `Dispatch<Object, UserData> for State` impl somewhere in our code. Each protocol gets its own module:

- `compositor.rs` — `wl_compositor`, `wl_surface`, the role-tracking `SurfaceData` (this one is load-bearing: every surface has an `Arc<Mutex<SurfaceData>>` in its user-data, and most of the rest of the codebase reads from it).
- `xdg_shell.rs` — `xdg_wm_base`, `xdg_surface`, `xdg_toplevel`, `xdg_popup`. The protocol that defines what we'd colloquially call "windows" and "menus".
- `layer_shell.rs` — `zwlr_layer_surface_v1`. The protocol waybar / rofi / wallpaper apps speak — they need to anchor to a screen edge with a known z-layer (Background, Bottom, Top, Overlay), not participate in tile layout.
- `seat.rs`, `input.rs` — pointer + keyboard input plumbing.
- `data_device.rs`, `primary_selection.rs` — clipboard (Ctrl+C/V) and middle-click paste, which are separate protocols.
- `linux_dmabuf.rs` — the protocol GPU clients (Chrome, mpv) use to hand us `dma-buf` fds instead of CPU-mapped pixels. The Vulkan backend imports them as `VkImage`s via `VK_EXT_external_memory_dma_buf`.
- `subcompositor.rs`, `viewporter.rs`, `xdg_decoration.rs`, `fractional_scale.rs`, `presentation_time.rs`, `xdg_output.rs`, `idle_inhibit.rs`, `ext_workspace.rs` — smaller protocols, one file each.

The pattern in all of them is the same: a `Dispatch` impl receives a `Request`, mutates `&mut State`, possibly sends a response event back via the resource handle. The state mutation is mostly bookkeeping — recording the new `wl_buffer` attached to a surface, recording a popup's positioner, recording that the client has acked a configure. The actual visual change comes a tick later.

## 4. The central data model

`State` (`lib.rs`, around line 331) is the compositor's brain. It's a single struct, single-threaded, mutated by every callback. The fields that matter for the bigger picture:

- `display_handle: DisplayHandle` — the wayland-server handle, used to send events back to clients.
- `mapped_toplevels: Vec<Window>` — the tiles, in z-order. The order matters: it's both the master-stack layout order and the focus-cycle order.
- `mapped_dialogs: Vec<DialogWindow>` — floating windows (anything with `xdg_toplevel.set_parent`, plus heuristic detection of fixed-size windows like splash screens).
- `mapped_popups: Vec<Weak<WlSurface>>` — `xdg_popup`s currently open. Drawn last in the scene so they sit above tiles and dialogs.
- `pending_layer_surfaces: Vec<Weak<WlSurface>>` — every `zwlr_layer_surface_v1` the server knows about. Background wallpaper, waybar, rofi all live here.
- `keyboard_focus: Option<WlSurface>`, plus the per-workspace `last_focus_per_workspace` for round-trip restoration.
- `active_ws: u32`, plus per-workspace memory maps (`zoomed_per_workspace`, `fullscreened_per_workspace`, `last_stack_focus_per_workspace`, `last_master_swap_partner_per_workspace`) — workspace-scoped state that wouldn't otherwise survive a switch.
- `cursor: Cursor` — software cursor with hot-spot.
- `config: Config` — parsed `sol.conf`. Hot-reloaded via inotify.
- `needs_render: bool` — the "should we draw a frame?" bit. Set by every dirty-making event, consumed by `render_tick`.

`Window` (line ~180) is the per-tile struct. It holds the `Weak<WlSurface>`, the **target** rect (`rect: Rect`, integer pixel-aligned, what master-stack computes), and the **animated** rect (`render_rect: RectF`, sub-pixel, what the GPU is asked to draw). Plus a velocity for each spring component, plus alpha and scale springs for open / close fades, plus a border-alpha spring for focus changes. The split is the whole reason animation is invisible to the protocol layer: layout decides `rect`, the animation tick interpolates `render_rect` toward it, the renderer reads `render_rect`.

## 5. The render tick — the heartbeat

Every observable thing the compositor does happens because `render_tick` ran (`lib.rs`, around line 3454). It's the one place where state and backend meet. The shape:

```
render_tick:
  if a page flip is already in flight:
    re-arm needs_render and bail   # don't pile up flip requests
  evict GPU resources for buffers clients destroyed since last tick
  prune dead surfaces from mapped_toplevels / mapped_popups / mapped_dialogs
  rebalance keyboard focus (if focused surface died)
  apply_layout — recompute every tile's target rect
  send_pending_configures — tell clients about size changes
  tick_animations — step every spring one dt
  tick_workspace_animation — advance the workspace-switch crossfade
  collect_scene — turn protocol state into a Vec<Placed>
  scene_from_buffers — turn that into a Scene<'a> with borrowed buffer refs
  add focus borders
  hand the Scene to the backend (DrmPresenter::render_scene OR the headless canvas)
  fire frame callbacks (headless path; DRM path waits for page-flip-complete)
```

That sequence is the contract. Nothing rendered isn't in the `Scene`; nothing in `Scene` came from anywhere but `collect_scene`. The whole compositor's behavior is a function of "what's in `State` right now?".

`render_tick` is called from two places: the calloop "idle" callback, set whenever `state.needs_render` is true and gated by `has_active_animation`, and the DRM page-flip-complete handler, which schedules another tick after each successful flip if anything is still animating. The idle-skip — not rendering when nothing has changed — is what keeps a quiescent desktop at 0% CPU instead of the per-vblank-tick CPU you'd see in a less-disciplined main loop.

## 6. Layout — where do tiles go?

`apply_layout` (line ~1507) takes the active workspace's tiles and assigns each a target `rect`. Three cases:

1. **Fullscreen:** if `state.fullscreened` is set and points to a tile on the active workspace, that tile gets the full output rect (no gaps, no border, no rounded corners — handled in the renderer). All other tiles' rects are left untouched; `collect_scene` skips them. Returns early.
2. **Zoom:** if `state.zoomed` is set, that tile gets the inner usable area (gaps respected). Others left as-is; again returns early.
3. **Master-stack:** the default. `master_stack_layout` (line ~1462) splits the inner usable area into a master pane on the left (`master_ratio` of the width) and a vertical stack of the remaining tiles on the right. `n=0` returns empty; `n=1` gives the one tile the whole area; `n>=2` does the split.

"Inner usable area" is the full output rect minus any layer-shell exclusive zones (a top-anchored waybar with `exclusive_zone=30` shrinks the available area by 30px at the top), minus `gaps_out` from each edge. Adjacent tiles get `gaps_in` between them, computed via `shrink_interior_edges` (line ~1633).

When `apply_layout` changes a tile's `rect`, it sets `pending_layout = true`. That's a flag the animation tick honors: don't start interpolating yet, the client hasn't committed a buffer at the new size. Once `settle_pending_layout` runs (called from the commit dispatch when the buffer's dimensions match `rect`), the flag clears and the spring kicks off. Without this, you get either a stretched buffer mid-animation or a snap from old size to new without a smooth tween.

## 7. Animation — springs, not tweens

`tick_animations` (line ~2477) steps a per-window damped spring system. There's no "duration" — each component (x, y, w, h of the rect, plus alpha, scale, border-alpha) is a damped spring with a stiffness and a damping ratio. Each tick:

```
force = (target - current) * stiffness
damp  = -velocity * damping
velocity += (force + damp) * dt
current  += velocity * dt
```

When `(target - current).abs()` is under half a pixel and velocity is sub-pixel-per-second, the spring is declared "settled" — the value is snapped to the target and velocity zeroed, so we don't accumulate floating-point noise.

The configurable knobs in `sol.conf`:

- `spring_stiffness` / `spring_damping` — base layout spring.
- `spring_stiffness_vertical` / `spring_damping_vertical` — y/h overrides for right-stack reorders.
- `spring_stiffness_swap` / `spring_damping_swap` — dedicated spring while two stack tiles trade places (the `swap_active` flag).
- `spring_stiffness_fade` / `spring_damping_fade` — open / close fade and zoom / fullscreen crossfade.

The damping ratio `damping / (2 * sqrt(stiffness))` decides feel: 1.0 is critically damped (no overshoot), <1.0 underdamped (one bounce), >1.0 overdamped (slow). sol's defaults sit slightly underdamped.

The trick worth noting: zoom and fullscreen don't have a special render path for "all the other tiles disappear". They just set those tiles' `alpha_target` to 0 and let the fade spring carry them out. When zoom is dismissed, the target flips back to 1 and they fade in. Same code path for normal map / unmap.

## 8. Building the scene

`collect_scene` (line ~1773) is the function that traverses State and produces a flat `Vec<Placed>` — an internal enum that says "here's a buffer to draw at this rect with this alpha and uv source", or "here's a frosted-glass backdrop to compute". The order it emits in IS the back-to-front draw order. Roughly:

1. Background + Bottom layer surfaces (wallpaper first, then anything else anchored to the lowest layers). The number of these is recorded as `background_count` so the DRM presenter can pre-render them into a separate FBO for the blur pipeline to sample from.
2. Tiled toplevels for the active workspace, in `mapped_toplevels` order. During a workspace crossfade, both workspaces' tiles get emitted with their respective `ws_alpha`.
3. Closing windows — frozen-buffer copies of tiles that were just unmapped, fading and shrinking to zero.
4. **Border anchor** — a marker index. The focused-tile border draws here (between tiles and floating elements), not at the very top of the stack.
5. Top layer surfaces (waybar etc).
6. Fullscreen tile, if any. Drawn above Top layers so it covers waybar.
7. Floating dialogs, popups.
8. Overlay layer surfaces (lockscreen, OSD).
9. Cursor.

`scene_from_buffers` (line ~2279) then walks `Vec<Placed>` and produces the actual `Scene<'a>` — the borrow-the-buffer-references step, so the renderer reads pixels without us needing to clone them.

`SceneElement`s are sub-pixel-positioned (`x: f32, y: f32`) so spring-driven motion doesn't quantize to integer pixels and produce visible step-judder near the curve's tail. The GPU's bilinear filter absorbs the fractional offset.

## 9. The renderer — `DrmPresenter::render_scene`

`crates/sol-backend-drm/src/presenter.rs::render_scene` is where the `Scene` becomes pixels. It's a Vulkan 1.3 pipeline driving a GBM-backed scan-out ring — no `VkSwapchainKHR`, no EGL, no compositor-side window-system dance: we own the display via DRM/KMS and just render into images the kernel will scan out.

The crate splits into a few small modules so the per-frame logic stays readable:

- `vk_stack.rs` — load libvulkan, create instance + device + graphics queue. Picks a discrete GPU first (skip llvmpipe), requires the dmabuf-import / sync-FD extensions.
- `vk_swap.rs` — allocate three GBM BOs (`SCANOUT | RENDERING | LINEAR`, `XRGB8888`), import each as a `VkImage` via `VK_EXT_image_drm_format_modifier` + `VK_EXT_external_memory_dma_buf`. The `Slot` ring tracks which BO is on screen, which has a flip queued, and which is free for the next render.
- `vk_pipe.rs` — four graphics pipelines that all share `quad.vert`: `quad` (textured), `solid` (rounded ring/border), `blur` (5×5 box), `backdrop` (sample blurred FBO with rounded mask). All use `dynamic_rendering` (no `VkRenderPass`). Per-draw uniforms travel as push constants; samplers are one combined-image-sampler descriptor at `(set=0, binding=0)`.
- `vk_texture.rs` — per-`buffer_key` `TextureEntry` holding `VkImage` + view + descriptor set. SHM uploads memcpy host pixels into a persistent host-visible staging buffer, then the per-frame command buffer issues `vkCmdCopyBufferToImage` bracketed by layout barriers. Dmabuf imports happen once per `wl_buffer` via `VK_EXT_external_memory_dma_buf`. The cursor sentinel still gets the upload-skip fast path.
- `vk_blur.rs` — three offscreen images for the inactive-window frosted-glass backdrop: `capture` at full output resolution + `ping`/`pong` at half. Ping-pongs with the `blur` pipeline; final result lives in `ping` or `pong` depending on parity.

Each `render_scene` call walks this sequence:

1. **Busy check.** If a flip is queued (`pending_flip`) or a sync FD is still in flight (`pending_render`), bail and drop the frame — the next `render_tick` will retry.
2. **Cache check.** The background slice (wallpaper + bottom layers) is hashed into a signature; if it matches last frame and the blur params are unchanged, the blur ping/pong images from the previous frame still hold the right result, so we skip both the bg pre-pass *and* the blur passes. Hot path for "client churns its own buffer but the wallpaper is static" (`cmatrix` in alacritty while waybar's clock advances).
3. **SHM uploads + dmabuf imports.** `prepare_uploads` walks the scene; SHM elements memcpy into their per-entry staging buffer and register a `PendingUpload`, dmabuf elements import once on first sight. The `upload_seq` plumbing is wired through but the actual upload-skip is gated on `CURSOR_SCENE_KEY` only — three richer modes (universal seq-skip, per-role gate keyed on layer-shell vs xdg_toplevel, and Hyprland-style snapshot-at-commit) all glitched on the development NVIDIA setup. See the `project_shm_upload_skip_disabled` memory entry for the full investigation. Cost is ~32 MB / frame of wallpaper-class re-upload at 4K, which puts the compositor at 120-180 fps under shadertoy + Chrome load instead of 240; that's the headline open perf item.
4. **Command buffer record.** Reset the per-frame primary command buffer, begin recording.
5. **Upload copies.** `record_uploads` translates each `PendingUpload` into a layout barrier (`SHADER_READ_ONLY` ⇒ `TRANSFER_DST`) + `vkCmdCopyBufferToImage` + barrier (`TRANSFER_DST` ⇒ `SHADER_READ_ONLY`).
6. **Background pre-pass.** If the cache failed, render the background slice into the blur capture FBO — same `quad` pipeline as the main draw, just into the half-res capture image instead of the scan-out slot.
7. **Blur passes.** `BlurChain::run` issues N rounds of 5×5 box blur ping-ponging between `ping` and `pong`. Each pass is a fullscreen quad; layout barriers transition src into `SHADER_READ_ONLY` and dst into `COLOR_ATTACHMENT_OPTIMAL` between passes.
8. **Main draw.** Begin dynamic rendering against the slot's image. Walk scene elements back-to-front: quad pipeline for textured surfaces, backdrop pipeline for `BlurredBackdrop` entries (sampling the final blur image), solid pipeline injected at `border_anchor`. Pipeline switches are tracked so consecutive same-pipeline draws don't rebind. Per-draw, push the `QuadPC` / `BackdropPC` / `SolidPC` block and `vkCmdDraw(4, 1, 0, 0)`.
9. **End rendering, transition slot to `GENERAL`.** DRM doesn't track Vulkan layouts; `GENERAL` is the spec-blessed "any operation" state, fine for kernel scan-out.
10. **Submit + sync-FD export.** `vkQueueSubmit` signals an exportable binary semaphore (created with `VkExportSemaphoreCreateInfo` for `SYNC_FD`); `vkGetSemaphoreFdKHR` transfers the payload out as an `OwnedFd` that becomes readable when the GPU finishes. The presenter sets `pending_render = true` and returns the FD up to the wayland-side; the wayland-side registers it as a one-shot `Generic` calloop source. Nothing has called `add_framebuffer` / `page_flip` yet — that's the point. The render thread returns to the event loop while the GPU is still working.

When the sync FD signals (GPU done), the calloop source fires and `submit_flip_after_fence` runs: `add_framebuffer` (cached, keyed on the BO's GEM handle), `drmModePageFlip(EVENT)`. State transitions `pending_render → pending_flip`. The kernel queues the flip for the next vblank.

The fallback path (driver doesn't expose `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD`) submits with a fence, blocks via `vkWaitForFences`, then runs `add_framebuffer` + `page_flip` inline — same shape as the pre-rewrite synchronous fallback. A startup log line tells you which path you're on (Mesa always supports the fast path on Linux).

The page flip is **not** waited on synchronously. The kernel wakes us via a readable event on the DRM fd; calloop's source callback calls `drain_events`, which clears `pending_flip` and rotates the slot ring. If anything was animating, `needs_render = true` schedules the next tick.

`is_busy()` covers both `pending_flip` and `pending_render` so a re-entry into `render_tick` doesn't try to start a second frame on top of an in-flight one.

### Format / orientation choices

- All scan-out and intermediate images use `VK_FORMAT_B8G8R8A8_UNORM`, which matches Wayland's little-endian ARGB / XRGB and the Linux DRM `AR24` / `XR24` fourcc bytes. The shader sees `(R, G, B, A)` directly; no per-shader swizzle.
- Vulkan's NDC has +Y pointing down — the opposite of GL's. The presenter's CPU-side `ndc_rect` uses Vulkan-native math (`y / fb_h * 2.0 - 1.0` puts the top edge at NDC y=-1) and the vertex shader doesn't flip V on UVs. Top-row-first Wayland buffers and top-row-first dmabufs are sampled right-side-up without tricks.
- Scan-out modifier is fixed at `DRM_FORMAT_MOD_LINEAR` for now — `BufferObjectFlags::LINEAR` on the GBM allocation, matched by an explicit modifier in the Vulkan import. Modifier negotiation between GBM and the Vulkan device is a future-perf knob, not a correctness requirement.

### Shaders

GLSL sources live in `crates/sol-backend-drm/shaders/`. `build.rs` invokes the system `glslc` (from the `shaderc` package) at compile time to produce SPIR-V in `OUT_DIR`; the renderer `include_bytes!`'s the `.spv` outputs. Edit a `.frag` / `.vert`, `cargo build` re-runs `glslc`, no manual step. If `glslc` isn't on `PATH`, the build fails loudly with the install hint baked into the panic message — there's no silent fallback to stale SPIR-V.

## 10. Input

libinput delivers raw events via its own fd. `InputState::drain` (`input.rs`) translates each into a high-level `InputEvent` (motion, button, axis, key) and `apply_input` (`lib.rs`, around line 4170) dispatches them. The interesting bits:

- **Pointer focus** — every motion event ends with `update_pointer_focus_and_motion`, which hit-tests the cursor position via `surface_under_cursor` (a top-to-bottom walk: popups → overlay → dialogs → fullscreen → top layer → tiles → bottom/background), then sends `wl_pointer.leave` / `wl_pointer.enter` if the focus surface changed, and `wl_pointer.motion` with surface-local coordinates.
- **Keyboard input** runs through xkbcommon. `xkb.rs` is a thin wrapper that turns evdev keycodes into keysyms via the user's layout, runs them past the configured `bind` rules in `sol.conf` (which match on modifier mask + key), and either dispatches the bound action (focus_dir, exec, workspace, etc.) or forwards the key to the focused client via `wl_keyboard.key`. `Ctrl+Alt+F1..F12` is hardcoded for VT switching and intentionally cannot be bound in config.
- **Modal resize loop** — `Alt+R` sets `state.resize_mode = true`; subsequent H/L bumps `master_ratio`, all other keys are swallowed (not forwarded to the client) so the user can rapid-fire without holding chords. `Esc` exits.
- **Super+drag** — clicks on a floating window with Super held start a `DialogDrag`; pointer-motion events update the window's offset until release. This is the workaround for clients that don't implement `xdg_toplevel.move`.

## 11. Workspaces

Workspaces are not a Wayland concept; they're a compositor-side filter on `mapped_toplevels`. Every `Window` carries a `workspace: u32`. `apply_layout` only sees windows on `state.active_ws`. `collect_scene` only emits windows on `active_ws` (or both, briefly, during a workspace crossfade).

Switching workspaces (`switch_workspace`, line ~3260) does five things in order:

1. Save the leaving workspace's keyboard focus to `last_focus_per_workspace`.
2. Save the leaving workspace's zoom / fullscreen mode to the per-workspace maps (so the round-trip restores the spotlight view, not the tile layout).
3. Update `active_ws`.
4. Restore the arriving workspace's zoom / fullscreen from the maps. Drop dead weaks.
5. Reset `render_rect = (0,0,0,0)` for every tile on the new workspace, so they get the "first-frame snap" treatment from `tick_animations` (no spring interpolation from a stale rect — they just appear laid out).

Crossfade is gated by `workspace_animation_duration_ms`. While the timer is running, both workspaces render simultaneously with `from_alpha` and `to_alpha` produced by `workspace_anim_alphas`.

The protocol layer of this is `ext_workspace.rs` — we advertise an `ext_workspace_manager_v1` global so a status bar can list and activate workspaces. `Activate` on a `WorkspaceHandle` routes through `switch_workspace` so the keyboard path and the protocol path share the same code.

## 12. Hot-reload

`install_config_watcher` (line ~1302) sets up an inotify watch on `~/.config/sol/`. Editors that save via "write tmp, rename over original" (vim's default, neovim, kate, gedit) replace the inode, which would dump a file-level watch. Watching the directory and filtering by basename catches both that pattern and write-in-place editors.

When the file changes, `apply_config_reload` reparses, diffs against the live `config`, and propagates the changes that don't require a restart — gaps, borders, idle timeout, keybind table, key remaps, spring tuning, the inactive-blur params, the rounded-corner radius. A handful of fields (`mode`, mostly) are still parsed but require a restart to take effect; they're tagged with comments in `sol.conf`.

## 13. Where the abstraction holds

What's worth re-reading after that walkthrough is the `sol-core` crate. It's tiny — about 170 lines — and it's where the contract between the protocol layer and the renderer is written down. `Scene`, `SceneElement`, `SceneContent`, `SceneBorder`. That's the entire surface area between two ~5,000-line bodies of code. As long as `collect_scene` produces a valid `Scene` and the backend renders it, you can swap either half without touching the other. The headless backend exists in production specifically to keep that boundary honest — anything that leaks protocol state into the renderer breaks the PNG dump first.

The other place the abstraction holds is `Window::render_rect` vs `Window::rect`. `apply_layout` only writes `rect`. `tick_animations` only reads `rect` and writes `render_rect`. `collect_scene` only reads `render_rect`. The protocol layer never sees animated values, the renderer never sees layout targets, and the animation system can be ripped out and replaced (it has been — the original was time-based ease curves) without either side noticing.

That's the whole shape of it. Everything else is filling in the protocol modules one at a time.
