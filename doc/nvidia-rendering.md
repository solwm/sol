# NVIDIA rendering correctness: the "cursed region" post-mortem

A debugging trail through three separate NVIDIA-specific defects that
stacked into one user-visible symptom: a dark, cross-window region —
the clear color bleeding through both the wallpaper and the windows
above it — pinned to the lower part of the right stack column,
appearing during window movement and other scene updates, dramatically
worse with a continuously-rendering wallpaper daemon.

Each layer below was a real, independent bug. Fixing the first two was
necessary (they corrupted other clients in their own ways) but the
symptom only fully cleared when the third was found. Recorded here
because every one of these is invisible on Mesa/AMD and will bite again
the next time someone touches the dmabuf or blur paths while developing
on non-NVIDIA hardware.

## Layer 1: implicit modifiers — shredded tiling

**Symptom:** Vulkan-WSI clients (vkgears, wgpu apps like sol-wallpaper)
rendered as horizontal shredded lines. EGL clients (alacritty/glutin,
Chrome) looked fine.

**Cause:** sol only advertised `Invalid` + `Linear` in its
zwp_linux_dmabuf_v1 feedback. NVIDIA's Vulkan WSI then allocates
block-linear buffers and tags them `Modifier::Invalid` (implicit
tiling). The importer fed that `Invalid` straight into
`VkImageDrmFormatModifierExplicitCreateInfoEXT` — undefined behaviour;
the image was sampled with the wrong memory layout. NVIDIA's EGL
happens to pick an actually-linear layout in the same situation, which
is why GL clients survived.

**Fix:** query the driver's supported modifiers at startup
(`VkDrmFormatModifierPropertiesListEXT`, `vk_stack.rs`), advertise them
in the dmabuf feedback (`setup_event_loop`), validate client modifiers
against the same list at import time, and build one plane layout per
memory plane (`vk_texture.rs`). Clients then allocate with explicit,
supported tiling. Side benefit: NVIDIA's WSI stops reallocating
wl_buffers constantly — imports went from ~14/s to ~0 steady-state.

## Layer 2: no fences anywhere — sampling in-flight buffers

**Symptom:** content sampled from a client buffer while the client's
GPU was still writing it. Most visible through the blur capture pass
(the earliest sampling in the frame), which then *cached* the torn
result: the blur cache made a transient race look like a persistent
artifact.

**Cause:** NVIDIA attaches no implicit fences to dmabufs — not for
Vulkan import to inherit, and none on the dma_resv for the kernel to
export (`DMA_BUF_IOCTL_EXPORT_SYNC_FILE` returns already-signaled
fences; we verified this empirically — the export "worked" and changed
nothing). The in-tree comment claiming implicit sync held "on first
read after creation" was true but useless for re-used buffers, which is
what a continuously-rendering wallpaper produces every frame.

**Fix (both kept):**

- `linux-drm-syncobj-v1` explicit sync (`setup_event_loop`,
  `new_surface` pre-commit hook): clients attach an acquire timeline
  point per commit; a `DrmSyncPointBlocker` holds the surface
  transaction until the client's GPU signals it. By the time a buffer
  enters the scene its content is complete *by construction*. This is
  the protocol that ended the NVIDIA-Wayland flicker era; NVIDIA
  drivers ≥555 use it whenever the compositor offers it. Gated on
  `drmSyncobjEventfd` support.
- Producer-fence export (`vk_texture.rs::export_dmabuf_read_fence` →
  submit wait semaphores in `presenter.rs`): per dmabuf whose content
  advanced, export its pending-write sync file and make the render
  submit wait on it at `FRAGMENT_SHADER`. A no-op on NVIDIA (no resv
  fences, see above) but real protection on Mesa drivers and for
  clients that don't speak syncobj.

## Layer 3: the actual cursed region — blended RMW raster defect

**Symptom:** with both sync layers fixed and buffers provably complete,
the dark region persisted. User bisection: disabling `inactive_blur`
made it vanish.

**Cause:** an NVIDIA raster-order defect with **fractional-alpha
read-modify-write draws**: in the bottom-right region of the
framebuffer, the blend unit can read stale/cleared destination pixels
instead of what earlier draws in the same pass already wrote. The
frosted-glass path was built from exactly such draws — a blended
backdrop quad plus the inactive window blended at `inactive_alpha` —
so it blended against the frame's *clear color* instead of the
wallpaper underneath. This is not a sol bug; niri-sol hit the identical
defect with smithay's stock GL renderer and documented it in its
manual-blend-offscreen commit ("NVIDIA bottom-right L-shape glitch in
the BLEND-on read-modify-write path").

**Fix:** eliminate hardware blending from the frosted path entirely.
The `BlurredBackdrop` scene element no longer draws; it arms the
`frosted` pipeline (`vk_pipe.rs`, `shaders/frosted.frag`) for the
window element that follows. That shader samples the window texture
(set 0) and the blurred background FBO (set 1, at `gl_FragCoord` —
both top-left origin, no flip) and computes the "over" blend itself,
writing opaque pixels. The driver's blend unit never touches these
pixels.

**Accepted trade-offs:** corner pixels outside the window's rounding
show frosted rather than sharp wallpaper (~`corner_radius` px), and the
backdrop no longer fades independently during workspace crossfades.

**Known residual:** minor traces of the defect can still appear from
the remaining fractional-alpha blended draws — subsurfaces of inactive
windows, focus-border fades, open/close animations, the software
cursor. If they ever bother anyone, the same treatment applies:
composite in-shader against a sampled background, write opaque. The
ultimate version is niri-sol's: route every fractional-alpha tile
through an offscreen and never blend against the framebuffer at all.

## Debugging notes for next time

- The metrics socket (`--features debug-ctl` + `bench/sol-bench`) was
  the workhorse: `n_dmabuf_imports_new` exposes buffer churn,
  `n_dmabuf_fence_waits` proves the sync path is exercised,
  `n_shm_upload_skipped_bytes` proves the wallpaper skip, and snapshot
  deltas give true fps without touching the screen.
- "Dark = clear color" is a load-bearing observation: both the scanout
  clear and the blur-capture clear use `[0.02, 0.02, 0.04]`. If a dark
  artifact matches that color, something sampled or blended against a
  cleared-but-not-yet-drawn target. (The headless canvas clears to a
  different `#202028` precisely so the two backends are tellable apart
  in captures.)
- The blur cache turns transient races into persistent artifacts —
  when hunting visual bugs near the backdrop, set the wallpaper to
  something that commits every frame to defeat the cache, or
  `inactive_blur = off` to bisect blur out entirely.
- Geometry of the artifact is a fingerprint: *bottom rows* → producer
  still writing (raster order is top-down) or scan-out latched early;
  *bottom-right L* → the NVIDIA blended-RMW defect; *uniform shred
  across the whole surface* → tiling/modifier mismatch.
