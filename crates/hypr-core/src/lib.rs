//! Shared types used across hyperland-rs crates. Kept dependency-free so any
//! other crate can pull this in without dragging Wayland or DRM deps along.

pub const NAME: &str = "hyperland-rs";

/// Wayland SHM pixel formats we know how to sample. Both store 32-bit little
/// endian words; the difference is whether the alpha byte is meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Argb8888,
    Xrgb8888,
}

/// One thing to draw on screen: a rectangle of pixels at a position. Produced
/// by the Wayland server on each render tick and consumed by whichever
/// backend is mounted (software PNG or DRM+GL).
pub struct SceneElement<'a> {
    /// Stable-across-frames identifier for this texture. Backends can use
    /// this as a cache key so a reused wl_buffer keeps its GPU texture.
    pub buffer_key: u64,
    pub pixels: &'a [u8],
    pub format: PixelFormat,
    pub width: i32,
    pub height: i32,
    pub stride: i32,
    /// Top-left in screen coordinates.
    pub x: i32,
    pub y: i32,
}

pub struct Scene<'a> {
    pub elements: Vec<SceneElement<'a>>,
}

impl<'a> Scene<'a> {
    pub fn new() -> Self {
        Self { elements: Vec::new() }
    }
}

impl<'a> Default for Scene<'a> {
    fn default() -> Self {
        Self::new()
    }
}
