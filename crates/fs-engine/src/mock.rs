//! Test engine: the real tiler and scheduler with a fake network, no GPU.
//!
//! [`MockEngine`] is a [`NetEngine`] whose "network" is a nearest-neighbour
//! ×N. It finishes submissions **last in, first out** (out of order on
//! purpose), counts how many were in flight at once, and injects
//! failures: out of memory above a tile size, a lost device after N runs.
//! [`MockBackend`] wraps it as an [`InferenceBackend`], so `FS_ENGINE=mock`
//! runs the whole worker pipeline through [`crate::planar::run_tiled`]. Its
//! frames equal [`ReferenceBackend`]'s pixel for pixel: nearest-neighbour is
//! exact, and so is the u8/u16 ⇄ f32 round trip.

use fs_types::pipeline::StageConfig;

use crate::cpu::ReferenceBackend;
use crate::engine::{BackendKind, Completed, EngineStats, NetEngine, SideInputs, Tensor, Ticket};
use crate::planar::{self, Planar, TileMemory};
use crate::{EngineError, Frame, InferenceBackend};

/// Context the mock "network" is given around each tile.
const PAD: usize = 4;

#[derive(Debug, Clone)]
pub struct MockConfig {
    pub scale: usize,
    pub max_in_flight: usize,
    /// Tiles (padded) larger than this on either side fail with
    /// `OutOfDeviceMemory`.
    pub oom_above: Option<usize>,
    /// The n-th run (0-based) and every one after fail with `DeviceLost`.
    pub lost_after: Option<u64>,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            scale: 1,
            max_in_flight: 2,
            oom_above: None,
            lost_after: None,
        }
    }
}

impl MockConfig {
    /// `FS_MOCK_IN_FLIGHT`, `FS_MOCK_OOM_ABOVE`, `FS_MOCK_LOST_AFTER`.
    pub fn from_env(scale: usize) -> Self {
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok());
        Self {
            scale,
            max_in_flight: num("FS_MOCK_IN_FLIGHT").map_or(2, |n| n.max(1) as usize),
            oom_above: num("FS_MOCK_OOM_ABOVE").map(|n| n as usize),
            lost_after: num("FS_MOCK_LOST_AFTER"),
        }
    }
}

#[derive(Debug)]
pub struct MockEngine {
    cfg: MockConfig,
    queue: Vec<(Ticket, Planar)>,
    next: u64,
    runs: u64,
    peak: usize,
}

impl MockEngine {
    pub fn new(cfg: MockConfig) -> Self {
        Self {
            cfg,
            queue: Vec::new(),
            next: 0,
            runs: 0,
            peak: 0,
        }
    }

    /// Most submissions that were outstanding at once.
    pub fn peak_in_flight(&self) -> usize {
        self.peak
    }

    /// Runs so far (submissions that finished).
    pub fn runs(&self) -> u64 {
        self.runs
    }

    fn run(&mut self, t: &Planar) -> Result<Planar, EngineError> {
        let n = self.runs;
        self.runs += 1;
        if self.cfg.lost_after.is_some_and(|k| n >= k) {
            return Err(EngineError::DeviceLost("mock: device lost".into()));
        }
        if self.cfg.oom_above.is_some_and(|m| t.w.max(t.h) > m) {
            return Err(EngineError::OutOfDeviceMemory {
                tile: (t.w.max(t.h) - 2 * PAD) as u32,
            });
        }
        let s = self.cfg.scale;
        let mut o = Planar::zeros(t.w * s, t.h * s);
        for c in 0..3 {
            for y in 0..o.h {
                for x in 0..o.w {
                    o.set(c, x, y, t.at(c, x / s, y / s));
                }
            }
        }
        Ok(o)
    }
}

impl NetEngine for MockEngine {
    fn max_in_flight(&self) -> usize {
        self.cfg.max_in_flight
    }

    fn submit(&mut self, input: Tensor, _side: Option<SideInputs>) -> Result<Ticket, EngineError> {
        if self.queue.len() >= self.cfg.max_in_flight {
            return Err(EngineError::Internal(format!(
                "mock: more than {} tiles in flight",
                self.cfg.max_in_flight
            )));
        }
        let t = Ticket(self.next);
        self.next += 1;
        self.queue.push((t, input.into_host()?));
        self.peak = self.peak.max(self.queue.len());
        Ok(t)
    }

    fn wait_any(&mut self) -> Option<Completed> {
        let (ticket, input) = self.queue.pop()?; // last in, first out
        let output = self.run(&input).map(Tensor::Host);
        Some(Completed {
            ticket,
            input: Some(input),
            output,
        })
    }

    fn poll(&mut self) -> Option<Completed> {
        self.wait_any()
    }
}

/// [`MockEngine`] behind the frame-level trait: Upscale ×N through the shared
/// tiler (with tile reuse); interpolation is [`ReferenceBackend`]'s blend.
#[derive(Debug)]
pub struct MockBackend {
    engine: MockEngine,
    reference: ReferenceBackend,
    scratch: Planar,
    memory: TileMemory,
    reuse: bool,
    last_tile: Option<u32>,
}

impl MockBackend {
    pub fn new(stages: &[StageConfig], cfg: MockConfig) -> Self {
        let scale = stages
            .iter()
            .find_map(|s| match s {
                StageConfig::Upscale { scale, .. } => Some(usize::from(*scale)),
                _ => None,
            })
            .unwrap_or(1);
        Self {
            engine: MockEngine::new(MockConfig { scale, ..cfg }),
            reference: ReferenceBackend::new(stages),
            scratch: Planar::zeros(1, 1),
            memory: TileMemory::default(),
            reuse: true,
            last_tile: None,
        }
    }

    pub fn engine(&self) -> &MockEngine {
        &self.engine
    }
}

impl InferenceBackend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn device_name(&self) -> Option<String> {
        Some("Mock engine (tests, non-AI)".into())
    }

    fn process(&mut self, input: &Frame, tile: u32) -> Result<Frame, EngineError> {
        let img = planar::from_frame(input);
        let t = (tile as usize).max(16);
        let scale = self.engine.cfg.scale;
        let memory = self.reuse.then_some(&mut self.memory);
        planar::run_tiled(
            &mut self.engine,
            &img,
            (t, t),
            PAD,
            scale,
            &mut self.scratch,
            memory,
            None,
        )?;
        self.last_tile = Some(tile);
        Ok(planar::to_frame(&self.scratch, input.format))
    }

    fn interpolate(
        &mut self,
        a: &Frame,
        b: &Frame,
        t: f32,
        tile: u32,
    ) -> Result<Frame, EngineError> {
        self.reference.interpolate(a, b, t, tile)
    }

    /// Twice the minimum, so an injected OOM has room to halve it.
    fn max_tile(&self) -> u32 {
        128
    }

    fn set_reuse(&mut self, on: bool) {
        self.reuse = on;
        if !on {
            self.memory.clear();
        }
    }

    fn stats(&self) -> EngineStats {
        let m = &self.memory;
        EngineStats {
            backend: Some(BackendKind::Mock),
            tile: self.last_tile,
            max_in_flight: Some(self.engine.max_in_flight() as u32),
            reused_tiles_pct: (m.tiles > 0)
                .then(|| (100.0 * m.reused as f64 / m.tiles as f64) as f32),
            ..EngineStats::default()
        }
    }

    fn report(&self) -> Option<String> {
        Some(format!(
            "mock engine: {} runs, peak {} in flight",
            self.engine.runs(),
            self.engine.peak_in_flight()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planar::tests::ramp;
    use fs_types::pipeline::ModelRef;
    use fs_types::worker::FrameFormat;

    fn upscale(scale: u8) -> Vec<StageConfig> {
        vec![StageConfig::Upscale {
            model: ModelRef {
                id: "mock".into(),
                path: None,
            },
            scale,
            denoise: None,
        }]
    }

    fn frame(w: u32, h: u32, fmt: FrameFormat) -> Frame {
        let n = fmt.frame_bytes(w, h);
        Frame::new(w, h, fmt, (0..n).map(|i| (i * 31 % 253) as u8).collect())
    }

    #[test]
    fn equals_the_reference_backend_for_any_scale_and_format() {
        for scale in [1u8, 2, 4] {
            for fmt in [FrameFormat::Rgb24, FrameFormat::Rgb48] {
                let f = frame(53, 37, fmt); // edge tiles on both axes
                let want = ReferenceBackend::new(&upscale(scale))
                    .process(&f, 16)
                    .expect("ref");
                for in_flight in [1, 2, 5] {
                    let cfg = MockConfig {
                        max_in_flight: in_flight,
                        ..MockConfig::default()
                    };
                    let mut b = MockBackend::new(&upscale(scale), cfg);
                    let got = b.process(&f, 16).expect("mock");
                    assert_eq!(got, want, "×{scale} {fmt:?} {in_flight} in flight");
                    assert!(b.engine().peak_in_flight() <= in_flight);
                    // Second frame: unchanged, every tile reused, same output.
                    let runs = b.engine().runs();
                    assert_eq!(b.process(&f, 16).expect("again"), want);
                    assert_eq!(b.engine().runs(), runs, "nothing re-ran");
                }
            }
        }
    }

    #[test]
    fn run_tiled_fills_the_slots_and_finishes_out_of_order() {
        let img = ramp(70, 50);
        let mut e = MockEngine::new(MockConfig {
            scale: 2,
            max_in_flight: 3,
            ..MockConfig::default()
        });
        let mut out = Planar::zeros(1, 1);
        planar::run_tiled(&mut e, &img, (16, 16), 4, 2, &mut out, None, None).expect("run");
        assert_eq!(e.peak_in_flight(), 3);
        assert_eq!(e.runs(), 5 * 4);
        let want = planar::tiled(&img, 1024, 0, 2, |t| {
            MockEngine::new(MockConfig {
                scale: 2,
                ..MockConfig::default()
            })
            .run(t)
        })
        .expect("whole");
        assert_eq!(out, want);
    }

    #[test]
    fn failures_surface_as_engine_errors() {
        let img = ramp(40, 40);
        let mut out = Planar::zeros(1, 1);
        let mut e = MockEngine::new(MockConfig {
            scale: 2,
            oom_above: Some(20),
            ..MockConfig::default()
        });
        assert_eq!(
            planar::run_tiled(&mut e, &img, (16, 16), 4, 2, &mut out, None, None),
            Err(EngineError::OutOfDeviceMemory { tile: 16 })
        );
        assert!(e.queue.is_empty(), "the tiler drained every submission");
        let mut e = MockEngine::new(MockConfig {
            scale: 2,
            lost_after: Some(3),
            ..MockConfig::default()
        });
        let r = planar::run_tiled(&mut e, &img, (16, 16), 4, 2, &mut out, None, None);
        assert!(matches!(r, Err(EngineError::DeviceLost(_))), "{r:?}");
        assert!(e.queue.is_empty());
    }
}
