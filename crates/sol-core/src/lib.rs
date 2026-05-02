//! Shared types used across sol crates. Kept dependency-free so any
//! other crate can pull this in without dragging Wayland or DRM deps along.

pub const NAME: &str = "sol";

/// Sentinel `SceneElement::buffer_key` for the compositor's default
/// cursor sprite. The sprite is allocated once at startup and never
/// mutated, so backends can fast-path "skip the per-frame
/// glTexSubImage2D for this key" without worrying about staleness.
/// Picked to fall outside the monotonic range that
/// `next_buffer_cache_key` produces; backends that don't special-case
/// this just treat it as any other key and re-upload every frame
/// (correct, just wasteful).
pub const CURSOR_SCENE_KEY: u64 = 0xC0FFEE_C0FFEE;

/// Wayland SHM pixel formats we know how to sample. Both store 32-bit little
/// endian words; the difference is whether the alpha byte is meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Argb8888,
    Xrgb8888,
}

/// Where a scene element's pixel data actually lives. SHM buffers are
/// CPU-mapped and the server blits/uploads from the borrowed slice. Dmabuf
/// buffers live on the GPU; the server imports the fd as an EGLImage on
/// first sight and re-uses the resulting GL texture.
pub enum SceneContent<'a> {
    Shm {
        pixels: &'a [u8],
        stride: i32,
        format: PixelFormat,
        /// Monotonic counter bumped each time the source buffer is
        /// re-committed. Backends compare it against the version they
        /// last uploaded for `buffer_key`; if equal, the existing GPU
        /// texture is current and the per-frame `glTexSubImage2D` can
        /// be skipped. Was the dominant idle cost at 240Hz — cursor /
        /// waybar / static layer surfaces were re-uploading every
        /// vblank because the renderer had no way to tell their
        /// pixels hadn't changed.
        upload_seq: u64,
    },
    Dmabuf {
        /// Raw dmabuf fd. Borrowed from the wl_buffer's user-data for one
        /// frame; the fd stays open for the full wl_buffer lifetime.
        fd: std::os::fd::RawFd,
        fourcc: u32,
        modifier: u64,
        offset: u32,
        stride: u32,
    },
    /// Frosted-glass backdrop: the presenter samples the already-rendered
    /// framebuffer at this element's screen rect, runs `passes` rounds of
    /// blur, and draws the blurred result back. Used to put a softened
    /// version of the wallpaper / lower layers behind partially-transparent
    /// inactive toplevels. Has no associated wl_buffer; sampling is
    /// driven entirely by screen position. `radius` scales the per-pass
    /// kernel offsets — 1.0 is the shader's natural 2-texel reach.
    BlurredBackdrop { passes: u32, radius: f32 },
}

/// One thing to draw on screen: a rectangle of pixels at a position. Produced
/// by the Wayland server on each render tick and consumed by whichever
/// backend is mounted (software PNG or DRM+GL).
///
/// `width`/`height` are the draw-quad size on screen — the
/// compositor's chosen output rect (tile rect for toplevels,
/// anchor rect for layer surfaces). `uv_*` describe the sub-rect
/// of the texture to sample from, in normalized `[0, 1]` coords;
/// populated from `wp_viewport.set_source` when the client uses
/// it, else the full texture `(0, 0, 1, 1)`. `wp_viewport`'s
/// destination is the surface's *logical* size and is **not**
/// fed here — it never drives output dimensions.
pub struct SceneElement<'a> {
    /// Stable-across-frames identifier for this texture. Backends can use
    /// this as a cache key so a reused wl_buffer keeps its GPU texture.
    pub buffer_key: u64,
    pub width: i32,
    pub height: i32,
    /// Top-left in screen coordinates. Sub-pixel precision (f32)
    /// because layout-change animations interpolate continuously and
    /// rounding to int per frame produces visible micro-judder near
    /// the curve's tail. The presenter feeds these directly into
    /// the NDC transform; the GPU's bilinear filter absorbs the
    /// fractional offset.
    pub x: f32,
    pub y: f32,
    /// Output rect size (the compositor's chosen on-screen dimensions).
    /// When `<= 0.0`, fall back to `width`/`height` (source) — the
    /// sentinel for subsurfaces / cursor that always render 1:1
    /// at buffer size. Toplevels and layer-shell surfaces pass
    /// the tile / anchor rect so a buffer larger or smaller than
    /// the output rect (common when `wp_viewport.set_source` is
    /// used to crop a buffer down to a sub-rect) gets stretched
    /// or cropped via UV into the intended quad.
    pub dst_width: f32,
    pub dst_height: f32,
    pub uv_x: f32,
    pub uv_y: f32,
    pub uv_w: f32,
    pub uv_h: f32,
    /// Per-element alpha multiplier in `[0, 1]`. Multiplied into the
    /// fragment shader's output (both alpha channel and RGB) so the
    /// surface blends with the framebuffer over the configured blend
    /// func. 1.0 is full opacity (the default for every element);
    /// 0.0 invisible. Used today for crossfade during workspace
    /// switches; available to any future caller that wants a fade.
    pub alpha: f32,
    /// Corner radius in pixels for rounded-rect masking. `0.0` =
    /// rectangular (the default — layer surfaces, cursor). Toplevels
    /// and their inactive-window backdrops set this to the
    /// configured value so windows render with rounded corners.
    /// Implemented in the fragment shader as an SDF-based alpha
    /// mask, which keeps the rounded edges anti-aliased (1-pixel
    /// transition) and preserves correct blending against whatever
    /// is behind the window.
    pub corner_radius: f32,
    pub content: SceneContent<'a>,
}

/// Flat-colored rectangle overlay — drawn on top of all textured
/// elements but below the cursor. Used today for the focused-tile
/// border; the render path is generic enough to reuse later for
/// notifications, window highlights, debug overlays, etc.
///
/// The fragment shader treats this as one of three shapes depending
/// on the trailing fields:
/// - `corner_radius == 0 && border_width == 0` → solid filled rect.
/// - `corner_radius > 0  && border_width == 0` → solid rounded rect.
/// - `border_width > 0`                        → rounded ring with
///   `border_width` thickness (both outer and inner edges follow
///   `corner_radius`; inner radius auto-shrinks by `border_width`).
///
/// `x` / `y` / `w` / `h` are f32 so the border can subscribe to
/// the same sub-pixel render_rect the window content uses, keeping
/// the ring concentric with the window during a layout-tween
/// animation.
pub struct SceneBorder {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Straight RGBA in [0, 1]. The DRM presenter treats it as
    /// un-premultiplied; the solid shader writes it directly.
    pub rgba: [f32; 4],
    /// Outer corner radius in pixels. `0.0` = rectangular.
    pub corner_radius: f32,
    /// Ring thickness in pixels; `0.0` collapses to a filled
    /// (possibly-rounded) rect.
    pub border_width: f32,
}

pub struct Scene<'a> {
    pub elements: Vec<SceneElement<'a>>,
    pub borders: Vec<SceneBorder>,
    /// Number of leading entries in `elements` that are background
    /// (Background + Bottom layer surfaces and their subsurfaces).
    /// The DRM presenter renders these into its capture FBO first,
    /// then blits the result onto the screen — that gives the blur
    /// pipeline a clean, sampler-friendly source independent of how
    /// the default framebuffer is laid out by GBM/EGL. Everything
    /// from `elements[background_count..]` draws normally on top.
    pub background_count: usize,
    /// Z-order anchor for the border pass. The presenter draws
    /// `elements[0..border_anchor]`, then `borders`, then
    /// `elements[border_anchor..]` — so focus rings sit between
    /// the tile pass and the dialog/popup/cursor passes instead
    /// of riding on top of every other surface. `usize::MAX` is
    /// the "draw borders last" sentinel (matches pre-anchor
    /// behaviour for callers that don't bother to set it).
    pub border_anchor: usize,
}

impl<'a> Scene<'a> {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            borders: Vec::new(),
            background_count: 0,
            border_anchor: usize::MAX,
        }
    }
}

impl<'a> Default for Scene<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-frame wall-clock breakdown of the backend render pass. The
/// presenter populates the four sub-phase fields inside
/// `render_scene`; the Wayland side adds them into its cumulative
/// `Metrics` so a benchmark harness can attribute time *inside* the
/// render phase, not just total it.
///
/// Sub-phases:
/// - `textures_ns`: SHM uploads + dmabuf imports + cache-key checks.
/// - `blur_ns`: blur capture pre-pass + ping/pong passes (zero when
///   the cache is valid for this frame).
/// - `draw_ns`: the main back-to-front textured-quad loop + the
///   border pass.
/// - `present_ns`: framebuffer lookup + page-flip schedule (the
///   GBM `lock_front_buffer` + `add_fb` + `drmModePageFlip`
///   sequence).
///
/// Values are cumulative *for this single render call*; the Wayland
/// side accumulates across frames before exporting via debug-ctl.
#[derive(Default, Debug, Clone, Copy)]
pub struct RenderTiming {
    pub textures_ns: u64,
    pub blur_ns: u64,
    pub draw_ns: u64,
    pub present_ns: u64,
}
