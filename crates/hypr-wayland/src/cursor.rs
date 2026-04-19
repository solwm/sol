//! A tiny software cursor: a 24x24 ARGB sprite (red dot, white outline, alpha
//! corners). Hand-drawn at startup so we don't need a cursor theme.
//!
//! Pixel layout matches Wayland SHM ARGB8888 in little-endian memory: four
//! bytes per pixel in the order [B, G, R, A]. That's what the DRM backend's
//! textured-quad shader expects.

pub const CURSOR_W: i32 = 24;
pub const CURSOR_H: i32 = 24;
pub const CURSOR_HOT_X: i32 = 12;
pub const CURSOR_HOT_Y: i32 = 12;

pub fn pixels() -> Vec<u8> {
    let w = CURSOR_W as i32;
    let h = CURSOR_H as i32;
    let mut p = vec![0u8; (w * h * 4) as usize];
    let cx = CURSOR_HOT_X;
    let cy = CURSOR_HOT_Y;
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            let dx = x - cx;
            let dy = y - cy;
            let dist2 = dx * dx + dy * dy;
            let (r, g, b, a): (u8, u8, u8, u8) = if dist2 <= 64 {
                (0xff, 0x46, 0x50, 0xff) // red fill
            } else if dist2 <= 100 {
                (0xff, 0xff, 0xff, 0xff) // white outline
            } else {
                (0x00, 0x00, 0x00, 0x00) // transparent
            };
            p[idx] = b;
            p[idx + 1] = g;
            p[idx + 2] = r;
            p[idx + 3] = a;
        }
    }
    p
}
