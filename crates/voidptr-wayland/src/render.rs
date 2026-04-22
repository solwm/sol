//! Software canvas and PNG dump used by the headless backend at B2.
//!
//! This is intentionally the dumbest possible renderer: a flat ARGB8888
//! buffer we memcpy pixels into, then encode as PNG. Real rendering lands in
//! B3/B4 when DRM+GLES arrive; the interface here (`Canvas::blit_argb`,
//! `Canvas::write_png`) is what the compositor will keep calling.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};

pub use voidptr_core::PixelFormat;

pub struct Canvas {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA bytes, matching what `png` wants directly.
    pixels: Vec<u8>,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        // Slate grey so we can tell the canvas from a genuinely black blit.
        for px in pixels.chunks_exact_mut(4) {
            px[0] = 0x20;
            px[1] = 0x20;
            px[2] = 0x28;
            px[3] = 0xff;
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    pub fn clear(&mut self) {
        for px in self.pixels.chunks_exact_mut(4) {
            px[0] = 0x20;
            px[1] = 0x20;
            px[2] = 0x28;
            px[3] = 0xff;
        }
    }

    /// Blit a source ARGB/XRGB buffer at (dst_x, dst_y). Src pixels are the
    /// raw bytes from a wl_shm buffer: little-endian 32-bit words where the
    /// top byte is A, then R, G, B. Clipped to the canvas bounds.
    pub fn blit_argb(
        &mut self,
        dst_x: i32,
        dst_y: i32,
        src: &[u8],
        src_width: i32,
        src_height: i32,
        src_stride: i32,
        format: PixelFormat,
    ) {
        if src_width <= 0 || src_height <= 0 || src_stride < src_width * 4 {
            return;
        }
        let (cw, ch) = (self.width as i32, self.height as i32);
        let xs = dst_x.max(0);
        let ys = dst_y.max(0);
        let xe = (dst_x + src_width).min(cw);
        let ye = (dst_y + src_height).min(ch);
        if xs >= xe || ys >= ye {
            return;
        }
        for y in ys..ye {
            let src_row = (y - dst_y) as usize * src_stride as usize;
            let dst_row = (y as usize * self.width as usize + xs as usize) * 4;
            for x in xs..xe {
                let sx = (x - dst_x) as usize * 4;
                let s = &src[src_row + sx..src_row + sx + 4];
                // Wayland ARGB8888 in memory (little endian u32) -> [B, G, R, A]
                let b = s[0];
                let g = s[1];
                let r = s[2];
                let a = match format {
                    PixelFormat::Argb8888 => s[3],
                    PixelFormat::Xrgb8888 => 0xff,
                };
                let d = dst_row + (x - xs) as usize * 4;
                self.pixels[d] = r;
                self.pixels[d + 1] = g;
                self.pixels[d + 2] = b;
                self.pixels[d + 3] = a;
            }
        }
    }

    /// Fill a rectangle with a flat RGBA color. RGBA components are in
    /// [0, 1]; alpha is written straight into the framebuffer's alpha
    /// channel (no blending — the headless canvas doesn't model
    /// translucent overlays).
    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, rgba: [f32; 4]) {
        if w <= 0 || h <= 0 {
            return;
        }
        let (cw, ch) = (self.width as i32, self.height as i32);
        let xs = x.max(0);
        let ys = y.max(0);
        let xe = (x + w).min(cw);
        let ye = (y + h).min(ch);
        if xs >= xe || ys >= ye {
            return;
        }
        let to_byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let [r, g, b, a] = [to_byte(rgba[0]), to_byte(rgba[1]), to_byte(rgba[2]), to_byte(rgba[3])];
        for py in ys..ye {
            let row = py as usize * self.width as usize * 4;
            for px in xs..xe {
                let d = row + px as usize * 4;
                self.pixels[d] = r;
                self.pixels[d + 1] = g;
                self.pixels[d + 2] = b;
                self.pixels[d + 3] = a;
            }
        }
    }

    pub fn write_png(&self, path: &Path) -> Result<()> {
        let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut encoder = png::Encoder::new(file, self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().context("png header")?;
        writer
            .write_image_data(&self.pixels)
            .context("png image data")?;
        Ok(())
    }
}
