//! Shared types used across sol crates. Kept dependency-free so any
//! other crate can pull this in without dragging Wayland or DRM deps along.

pub const NAME: &str = "sol";

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
    /// driven entirely by screen position.
    BlurredBackdrop { passes: u32 },
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
    pub content: SceneContent<'a>,
}

/// Flat-colored rectangle overlay — drawn on top of all textured
/// elements but below the cursor. Used today for the focused-tile
/// border; the render path is generic enough to reuse later for
/// notifications, window highlights, debug overlays, etc.
pub struct SceneBorder {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    /// Straight RGBA in [0, 1]. The DRM presenter treats it as
    /// un-premultiplied; the solid shader writes it directly.
    pub rgba: [f32; 4],
}

pub struct Scene<'a> {
    pub elements: Vec<SceneElement<'a>>,
    pub borders: Vec<SceneBorder>,
}

impl<'a> Scene<'a> {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            borders: Vec::new(),
        }
    }
}

impl<'a> Default for Scene<'a> {
    fn default() -> Self {
        Self::new()
    }
}
