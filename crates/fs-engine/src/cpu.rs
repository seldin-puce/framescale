//! Deterministic CPU reference backend: nearest-neighbour ×N and linear blend.
//!
//! **Not AI.** It exists so that the full pipeline (decode → stages →
//! interpolation → segmented encode → concat → mux) runs and is tested on any
//! machine, including CI runners without a GPU.

use fs_types::pipeline::StageConfig;
use fs_types::worker::FrameFormat;

use crate::{EngineError, Frame, InferenceBackend};

#[derive(Debug, Clone)]
pub struct ReferenceBackend {
    scale: u32,
}

impl ReferenceBackend {
    pub fn new(stages: &[StageConfig]) -> Self {
        let scale = stages
            .iter()
            .find_map(|s| match s {
                StageConfig::Upscale { scale, .. } => Some(u32::from(*scale)),
                _ => None,
            })
            .unwrap_or(1);
        Self { scale }
    }
}

fn read(f: &Frame, i: usize) -> u32 {
    match f.format {
        FrameFormat::Rgb24 => u32::from(f.data[i]),
        FrameFormat::Rgb48 => u32::from(u16::from_le_bytes([f.data[i * 2], f.data[i * 2 + 1]])),
    }
}

fn write(out: &mut [u8], fmt: FrameFormat, i: usize, v: u32) {
    match fmt {
        FrameFormat::Rgb24 => out[i] = v as u8,
        FrameFormat::Rgb48 => out[i * 2..i * 2 + 2].copy_from_slice(&(v as u16).to_le_bytes()),
    }
}

impl InferenceBackend for ReferenceBackend {
    fn name(&self) -> &'static str {
        "cpu-reference"
    }

    fn device_name(&self) -> Option<String> {
        Some("CPU (reference, non-AI)".into())
    }

    fn process(&mut self, input: &Frame, _tile: u32) -> Result<Frame, EngineError> {
        if self.scale == 1 {
            return Ok(input.clone());
        }
        let s = self.scale as usize;
        let bpp = input.format.bytes_per_pixel();
        let (w, h) = (input.width as usize, input.height as usize);
        let (ow, oh) = (w * s, h * s);
        let mut out = vec![0u8; ow * oh * bpp];
        for y in 0..h {
            let src_row = &input.data[y * w * bpp..(y + 1) * w * bpp];
            // Build one widened row, then copy it `s` times.
            let mut row = Vec::with_capacity(ow * bpp);
            for px in src_row.chunks_exact(bpp) {
                for _ in 0..s {
                    row.extend_from_slice(px);
                }
            }
            for dy in 0..s {
                let o = (y * s + dy) * ow * bpp;
                out[o..o + ow * bpp].copy_from_slice(&row);
            }
        }
        Ok(Frame::new(ow as u32, oh as u32, input.format, out))
    }

    fn interpolate(
        &mut self,
        a: &Frame,
        b: &Frame,
        t: f32,
        _tile: u32,
    ) -> Result<Frame, EngineError> {
        if a.width != b.width || a.height != b.height || a.format != b.format {
            return Err(EngineError::Internal(
                "interpolate: frame shape mismatch".into(),
            ));
        }
        let mut out = vec![0u8; a.data.len()];
        let wt = (t.clamp(0.0, 1.0) * 256.0) as u32;
        for i in 0..a.samples() {
            let v = (read(a, i) * (256 - wt) + read(b, i) * wt + 128) >> 8;
            write(&mut out, a.format, i, v);
        }
        Ok(Frame::new(a.width, a.height, a.format, out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::tests::solid;
    use fs_types::pipeline::ModelRef;

    #[test]
    fn upscales_and_blends() {
        let m = ModelRef {
            id: "ref".into(),
            path: None,
        };
        let mut b = ReferenceBackend::new(&[StageConfig::Upscale { model: m, scale: 2 }]);
        let mut f = solid(3, 2, FrameFormat::Rgb48, 0);
        f.data[0] = 0xff; // pixel (0,0) red low byte
        let up = b.process(&f, 256).expect("up");
        assert_eq!((up.width, up.height), (6, 4));
        assert_eq!(up.data[0], 0xff);
        assert_eq!(up.data[6], 0xff, "(1,0) duplicates (0,0)");
        assert_eq!(up.data[12], 0, "(2,0) is the next source pixel");
        let row1 = 6 * 6;
        assert_eq!(up.data[row1], 0xff, "(0,1) duplicates row 0");

        let lo = solid(2, 2, FrameFormat::Rgb24, 0);
        let hi = solid(2, 2, FrameFormat::Rgb24, 200);
        let mid = b.interpolate(&lo, &hi, 0.5, 256).expect("blend");
        assert!(mid.data.iter().all(|v| *v == 100));
    }
}
