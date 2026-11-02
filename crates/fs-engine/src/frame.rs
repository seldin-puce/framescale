//! Raw frames as they come off the decoder pipe.

use fs_types::worker::FrameFormat;

#[derive(Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub format: FrameFormat,
    /// Packed RGB, row-major; 8-bit or 16-bit little-endian samples.
    pub data: Vec<u8>,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Frame({}x{} {:?}, {} B)",
            self.width,
            self.height,
            self.format,
            self.data.len()
        )
    }
}

impl Frame {
    pub fn byte_len(width: u32, height: u32, format: FrameFormat) -> usize {
        format.frame_bytes(width, height)
    }

    pub fn new(width: u32, height: u32, format: FrameFormat, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), Self::byte_len(width, height, format));
        Self {
            width,
            height,
            format,
            data,
        }
    }

    /// Normalised sample at byte offset of a sample (0.0–1.0).
    #[inline]
    pub fn sample(&self, sample_index: usize) -> f32 {
        match self.format {
            FrameFormat::Rgb24 => f32::from(self.data[sample_index]) / 255.0,
            FrameFormat::Rgb48 => {
                let i = sample_index * 2;
                f32::from(u16::from_le_bytes([self.data[i], self.data[i + 1]])) / 65535.0
            }
        }
    }

    pub fn samples(&self) -> usize {
        self.width as usize * self.height as usize * 3
    }
}

/// Sampled content hash for resume alignment: xxh3 over every 8th pixel of
/// every 8th row. It's cheap at 8K and robust to everything except identical
/// content, which aligns correctly anyway.
pub fn hash(frame: &Frame) -> u64 {
    let bpp = frame.format.bytes_per_pixel();
    let stride = frame.width as usize * bpp;
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&frame.width.to_le_bytes());
    h.update(&frame.height.to_le_bytes());
    for row in (0..frame.height as usize).step_by(8) {
        let line = &frame.data[row * stride..(row + 1) * stride];
        for px in line.chunks_exact(bpp).step_by(8) {
            h.update(px);
        }
    }
    h.digest()
}

/// Mean absolute luma difference in 0–1 over a sparse grid. This is the
/// backstop for scene cuts that scdet missed (e.g. below its threshold).
pub fn difference(a: &Frame, b: &Frame) -> f32 {
    if a.width != b.width || a.height != b.height || a.format != b.format {
        return 1.0;
    }
    let luma = |f: &Frame, px: usize| {
        0.2126 * f.sample(px * 3) + 0.7152 * f.sample(px * 3 + 1) + 0.0722 * f.sample(px * 3 + 2)
    };
    let (w, h) = (a.width as usize, a.height as usize);
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for y in (0..h).step_by(4) {
        for x in (0..w).step_by(4) {
            let px = y * w + x;
            sum += (luma(a, px) - luma(b, px)).abs();
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn solid(w: u32, h: u32, fmt: FrameFormat, v: u8) -> Frame {
        Frame::new(w, h, fmt, vec![v; Frame::byte_len(w, h, fmt)])
    }

    #[test]
    fn hash_is_content_sensitive_and_deterministic() {
        let a = solid(64, 32, FrameFormat::Rgb24, 10);
        let mut b = a.clone();
        assert_eq!(hash(&a), hash(&b));
        b.data[0] = 11; // pixel (0,0) is sampled
        assert_ne!(hash(&a), hash(&b));
    }

    #[test]
    fn difference_bounds() {
        let a = solid(32, 32, FrameFormat::Rgb48, 0);
        let b = solid(32, 32, FrameFormat::Rgb48, 255);
        assert!(difference(&a, &a) < 1e-6);
        assert!((difference(&a, &b) - 1.0).abs() < 1e-3);
    }
}
