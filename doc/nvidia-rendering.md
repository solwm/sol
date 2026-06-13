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

**This was a partial fix on a wrong diagnosis.** Removing blended draws
made the artifact rarer, which *looked* like progress and reinforced
the "blended-RMW hardware defect" theory. But it persisted: with blur
off entirely (plain transparent quad, no frosted path at all) the
clear-color bleed still appeared at the bottom of inactive windows
during motion. niri-sol's manual-blend-offscreen genuinely is the GL
analogue of this defect, but our renderer had a second, more basic bug
underneath — see Layer 4.

## Layer 4: the actual root cause — missing render-pass barrier

**Symptom:** identical to Layer 3, but reproducible with blur off and
any fractional `inactive_alpha`. The frame-capture dashcam
(`bench/triage.py`) was decisive here: it found the bleed pixels were
the **exact** clear color (`#05050A`, tolerance 2), not torn or stale
*content*. Exact clear value = the cleared framebuffer read back before
anything was drawn over it — which points at a pass-ordering bug, not a
buffer-sync or hardware-blend bug.

**Cause:** `draw_main_pass` doesn't render in one pass. The focus-border
interleave splits it into three sequential dynamic-rendering instances
on the *same* scan-out image — clear+wallpaper+early windows, then
borders (LOAD), then the remaining windows (LOAD) — with **no memory
dependency between them**. Vulkan render-pass instances do not
implicitly synchronize attachment access against each other; an
explicit barrier is required. Without it, the alpha blend in a later
instance read-modify-writes the attachment before the earlier
instance's writes are visible. On NVIDIA's tiler the blend reads the
cleared value for the last-flushed (bottom) tiles. Opaque draws hid it
(dst blend factor `ONE_MINUS_SRC_ALPHA` = 0 when alpha = 1); only
fractional alpha weighted the stale read in — exactly the
`inactive_alpha` bisection. Invisible on Mesa, whose tiler/ROP flush
masks the UB.

**Fix:** a `COLOR_ATTACHMENT_OUTPUT` self-dependency barrier
(`presenter.rs::color_attachment_rmw_barrier`) at each pass boundary in
`draw_main_pass`. Verified objectively: same provocation, same triage
parameters, 20 clear-bleed findings → 0. Costs ~nothing (three barriers
per frame; CPU tick 0.09 ms, GPU 0.60 ms, still 240 fps).

**Lesson:** "rarer after the change" is not "fixed," and a plausible
prior (the documented niri-sol hardware defect) delayed finding the
real bug. The exact-clear-color observation — only possible once the
capture tooling existed — is what finally discriminated a Vulkan
synchronization bug from a hardware blend quirk. Reach for objective
pixel evidence before committing to a hardware-quirk explanation.

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
