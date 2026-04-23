//! Shared types used across voidptr crates. Kept dependency-free so any
//! other crate can pull this in without dragging Wayland or DRM deps along.

pub const NAME: &str = "voidptr";

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
}

/// One thing to draw on screen: a rectangle of pixels at a position. Produced
/// by the Wayland server on each render tick and consumed by whichever
/// backend is mounted (software PNG or DRM+GL). Dimensions are always
/// the buffer's intrinsic size — no stretching. `wp_viewport`'s
/// `set_destination` is accepted for protocol compatibility (awww
/// mandatorily unwraps the global) but doesn't drive output
/// dimensions, since that would override the compositor's own
/// placement (tile rect / layer-shell anchor) and was what made
/// browsers glitch during moves.
pub struct SceneElement<'a> {
    /// Stable-across-frames identifier for this texture. Backends can use
    /// this as a cache key so a reused wl_buffer keeps its GPU texture.
    pub buffer_key: u64,
    pub width: i32,
    pub height: i32,
    /// Top-left in screen coordinates.
    pub x: i32,
    pub y: i32,
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
