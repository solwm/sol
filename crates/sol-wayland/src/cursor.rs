//! Cursor sprite loader.
//!
//! On startup we try to load the "default" (arrow) cursor from the
//! user's XCursor theme — resolved through `XCURSOR_THEME` /
//! `~/.icons` / `/usr/share/icons/<name>`. XCursor files are a list
//! of PNG-ish frames at various nominal sizes; we pick the frame
//! whose nominal size is closest to our desired pixel size. If
//! nothing resolves, we fall back to a hand-drawn red dot — ugly
//! but visible, and loud enough that broken theme discovery is
//! obvious.
//!
//! Pixel layout matches Wayland SHM ARGB8888 in little-endian
//! memory: four bytes per pixel as `[B, G, R, A]`, which is what
//! the DRM backend's textured-quad shader expects.

use xcursor::CursorTheme;
use xcursor::parser::{Image, parse_xcursor};

/// Fallback target cursor size when `XCURSOR_SIZE` is unset.
const DESIRED_SIZE: u32 = 24;

pub struct CursorSprite {
    pub pixels: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub hot_x: i32,
    pub hot_y: i32,
}

/// Load the default arrow cursor from the active XCursor theme,
/// or fall back to a hand-drawn red dot. Honors `XCURSOR_THEME`,
/// `XCURSOR_SIZE`, and the standard theme search path.
pub fn load() -> CursorSprite {
    match load_theme() {
        Some(sprite) => sprite,
        None => {
            tracing::warn!(
                "xcursor theme lookup failed for 'default'; falling back to hand-drawn dot"
            );
            dot_fallback()
        }
    }
}

fn load_theme() -> Option<CursorSprite> {
    let theme_name =
        std::env::var("XCURSOR_THEME").unwrap_or_else(|_| "default".into());
    let desired = std::env::var("XCURSOR_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DESIRED_SIZE);
    let theme = CursorTheme::load(&theme_name);

    // Different themes disagree about the canonical filename for
    // the default pointer. Try the common ones; first hit wins.
    let path = ["default", "left_ptr", "arrow"]
        .into_iter()
        .find_map(|name| theme.load_icon(name))?;
    let bytes = std::fs::read(&path).ok()?;
    let images = parse_xcursor(&bytes)?;
    let image = pick_closest(&images, desired)?;
    tracing::info!(
        theme = %theme_name,
        path = %path.display(),
        width = image.width,
        height = image.height,
        "cursor: loaded from xcursor theme"
    );
    Some(rgba_to_bgra(image))
}

/// Pick the `Image` whose nominal size is nearest `desired`. The
/// XCursor format stores `size` as the target visual size the
/// artist drew for; actual pixel dims may differ slightly, but
/// `size` is the right thing to match against.
fn pick_closest(images: &[Image], desired: u32) -> Option<&Image> {
    images
        .iter()
        .min_by_key(|img| img.size.abs_diff(desired))
}

/// XCursor pixel data is RGBA8888 with pre-multiplied alpha. Our
/// shader samples SHM buffers as ARGB8888 with a BGR swizzle (see
/// `sol-backend-drm/src/quad.rs`), so we store bytes as
/// `[B, G, R, A]`. Simple per-pixel reorder — no alpha re-multiply,
/// the source is already premultiplied.
fn rgba_to_bgra(img: &Image) -> CursorSprite {
    let w = img.width as i32;
    let h = img.height as i32;
    let mut out = vec![0u8; (w * h * 4) as usize];
    for i in 0..(w * h) as usize {
        let src = i * 4;
        let dst = i * 4;
        out[dst] = img.pixels_rgba[src + 2]; // B
        out[dst + 1] = img.pixels_rgba[src + 1]; // G
        out[dst + 2] = img.pixels_rgba[src]; // R
        out[dst + 3] = img.pixels_rgba[src + 3]; // A
    }
    CursorSprite {
        pixels: out,
        width: w,
        height: h,
        hot_x: img.xhot as i32,
        hot_y: img.yhot as i32,
    }
}

/// Last-resort cursor: a 24×24 red dot with a white ring. Shown
/// when no XCursor theme can be resolved, so there's at least a
/// visible pointer and broken theme discovery is self-evident.
fn dot_fallback() -> CursorSprite {
    const W: i32 = 24;
    const H: i32 = 24;
    let cx = 12;
    let cy = 12;
    let mut p = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let idx = ((y * W + x) * 4) as usize;
            let dx = x - cx;
            let dy = y - cy;
            let dist2 = dx * dx + dy * dy;
            let (r, g, b, a): (u8, u8, u8, u8) = if dist2 <= 64 {
                (0xff, 0x46, 0x50, 0xff)
            } else if dist2 <= 100 {
                (0xff, 0xff, 0xff, 0xff)
            } else {
                (0x00, 0x00, 0x00, 0x00)
            };
            p[idx] = b;
            p[idx + 1] = g;
            p[idx + 2] = r;
            p[idx + 3] = a;
        }
    }
    CursorSprite {
        pixels: p,
        width: W,
        height: H,
        hot_x: cx,
        hot_y: cy,
    }
}
