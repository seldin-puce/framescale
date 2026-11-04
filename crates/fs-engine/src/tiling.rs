//! Tile size: VRAM-OOM recovery ([`TileController`]: halve the tile and retry
//! the same frame) and per-device tuning ([`TileTuner`]: time a few sizes on
//! the first frames of a render and keep the fastest).

use crate::EngineError;

pub const MIN_TILE: u32 = 64;

/// Sizes the tuner chooses from.
const LADDER: [u32; 7] = [64, 96, 128, 192, 256, 384, 512];
/// Frames run at a new size before timing starts (allocations, first-use
/// costs), then frames timed per size (the fastest counts).
const WARMUP: u32 = 1;
const SAMPLES: usize = 2;
/// A size has to beat the best so far by this much to win (timing noise).
const MIN_GAIN: f64 = 0.04;
/// Upper bound on the estimated GPU time of one tile. Windows resets a GPU
/// that runs one submission for 2 s (TDR); a larger tile is only tried when
/// it should stay well under that.
const MAX_TILE_SECONDS: f64 = 0.4;

#[derive(Debug, Clone)]
pub struct TileController {
    tile: u32,
}

/// What happened while running one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileOutcome {
    /// Set when the tile shrank during this call (emit a `TileReduced` warning
    /// and persist the new size for this device/model).
    pub reduced_to: Option<u32>,
}

impl TileController {
    pub fn new(initial: u32) -> Self {
        Self {
            tile: initial.max(MIN_TILE),
        }
    }

    pub fn tile(&self) -> u32 {
        self.tile
    }

    /// Use `tile` from the next call on (the tuner's choice).
    pub fn set(&mut self, tile: u32) {
        self.tile = tile.max(MIN_TILE);
    }

    /// Run `f(tile)`. On OOM, halve the tile (down to [`MIN_TILE`]) and retry.
    /// Any other error, or OOM at the minimum tile, is returned to the caller.
    pub fn run<T>(
        &mut self,
        mut f: impl FnMut(u32) -> Result<T, EngineError>,
    ) -> Result<(T, TileOutcome), EngineError> {
        let mut reduced = None;
        loop {
            match f(self.tile) {
                Ok(v) => {
                    return Ok((
                        v,
                        TileOutcome {
                            reduced_to: reduced,
                        },
                    ));
                }
                Err(EngineError::OutOfDeviceMemory { .. }) if self.tile > MIN_TILE => {
                    self.tile = (self.tile / 2).max(MIN_TILE);
                    reduced = Some(self.tile);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// Finds the fastest tile size on this device: times the backend's default
/// on real frames, then its neighbours on the ladder, and keeps moving while
/// it gets faster (hill climbing). Every frame is still real output, only the
/// speed differs, so tuning costs a few slower frames rather than a pass.
///
/// Larger is not always faster: on RDNA 4, 256 px beat 512 px by half
/// because two small tiles in flight stay in cache. Smaller costs more
/// overlap. Where the balance lies depends on the GPU, so it's measured.
#[derive(Debug, Clone)]
pub struct TileTuner {
    ladder: Vec<u32>,
    /// Seconds per frame for each ladder size: `None` untried, infinite when
    /// ruled out (out of memory).
    time: Vec<Option<f64>>,
    /// The size being timed, and the fastest so far.
    cur: usize,
    best: usize,
    seen: u32,
    samples: Vec<f64>,
    /// Pixels per input frame, to estimate one tile's time.
    frame_px: f64,
    done: Option<u32>,
}

impl TileTuner {
    /// Start at `start` (the backend's default) for `frame_px`-pixel frames.
    pub fn new(start: u32, frame_px: u64) -> Self {
        let start = start.max(MIN_TILE);
        let mut ladder = LADDER.to_vec();
        if !ladder.contains(&start) {
            ladder.push(start);
            ladder.sort_unstable();
        }
        let cur = ladder.iter().position(|&t| t == start).unwrap_or(0);
        Self {
            time: vec![None; ladder.len()],
            ladder,
            cur,
            best: cur,
            seen: 0,
            samples: Vec::new(),
            frame_px: frame_px.max(1) as f64,
            done: None,
        }
    }

    /// The size for the next frame.
    pub fn tile(&self) -> u32 {
        self.done.unwrap_or(self.ladder[self.cur])
    }

    /// The chosen size, once tuning is over.
    pub fn result(&self) -> Option<u32> {
        self.done
    }

    /// `(tile, seconds per frame)` for every size timed so far.
    pub fn timings(&self) -> Vec<(u32, f64)> {
        self.ladder
            .iter()
            .zip(&self.time)
            .filter_map(|(&t, s)| s.filter(|s| s.is_finite()).map(|s| (t, s)))
            .collect()
    }

    /// Record one frame run at `tile` that took `seconds`. `oom`: it ran out
    /// of GPU memory, so neither this size nor any larger one is used.
    /// Returns the chosen size once tuning is over.
    pub fn record(&mut self, tile: u32, seconds: f64, oom: bool) -> Option<u32> {
        if self.done.is_some() || tile != self.ladder[self.cur] {
            return self.done;
        }
        if oom {
            for t in &mut self.time[self.cur..] {
                *t = Some(f64::INFINITY);
            }
            if self.best >= self.cur {
                if self.cur == 0 {
                    self.done = Some(self.ladder[0]);
                    return self.done;
                }
                // The next size down becomes the one to beat (timed next).
                self.best = self.cur - 1;
            }
        } else {
            self.seen += 1;
            if self.seen <= WARMUP {
                return None;
            }
            self.samples.push(seconds);
            if self.samples.len() < SAMPLES {
                return None;
            }
            let t = self.samples.iter().copied().fold(f64::INFINITY, f64::min);
            self.time[self.cur] = Some(t);
            let best = self.time[self.best].unwrap_or(f64::INFINITY);
            if self.cur != self.best && t < best * (1.0 - MIN_GAIN) {
                self.best = self.cur;
            }
        }
        self.advance();
        self.done
    }

    /// Whether ladder entry `i` is worth timing next.
    fn allowed(&self, i: usize) -> bool {
        if i >= self.ladder.len() || self.time[i].is_some() {
            return false;
        }
        if self.ladder[i] <= self.ladder[self.best] {
            return true;
        }
        // Larger than the best: only while one tile stays far from the GPU
        // watchdog. The whole frame's time per pixel overstates the GPU's,
        // which keeps the estimate on the safe side.
        match self.time[self.best] {
            Some(t) if t.is_finite() => {
                let side = f64::from(self.ladder[i]);
                t / self.frame_px * side * side <= MAX_TILE_SECONDS
            }
            _ => false,
        }
    }

    fn advance(&mut self) {
        let b = self.best;
        let next = if self.time[b].is_none() {
            Some(b) // after an OOM: time the new size to beat first
        } else {
            [b.checked_sub(1), Some(b + 1)]
                .into_iter()
                .flatten()
                .find(|&i| self.allowed(i))
        };
        match next {
            Some(i) => {
                self.cur = i;
                self.seen = 0;
                self.samples.clear();
            }
            None => self.done = Some(self.ladder[b]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the tuner against a cost model: `Ok(seconds per frame)` or
    /// `Err(())` for out of memory. Returns the choice and the sizes tried.
    fn tune(start: u32, frame_px: u64, cost: impl Fn(u32) -> Result<f64, ()>) -> (u32, Vec<u32>) {
        let mut t = TileTuner::new(start, frame_px);
        let mut tried = Vec::new();
        for _ in 0..200 {
            let tile = t.tile();
            if tried.last() != Some(&tile) {
                tried.push(tile);
            }
            let r = match cost(tile) {
                Ok(s) => t.record(tile, s, false),
                Err(()) => t.record(tile, 0.0, true),
            };
            if let Some(best) = r {
                assert_eq!(t.tile(), best, "keeps the choice");
                return (best, tried);
            }
        }
        panic!("tuning never finished: {tried:?}");
    }

    const HD: u64 = 1920 * 1080;

    #[test]
    fn walks_down_to_the_fastest_size() {
        // Fastest at 128; the default 256 is slower, and so are 192's
        // neighbours on the far side.
        let cost = |t: u32| {
            Ok(match t {
                64 => 1.6,
                96 => 1.3,
                128 => 1.0,
                192 => 1.1,
                256 => 1.2,
                _ => 1.5,
            })
        };
        let (best, tried) = tune(256, HD, cost);
        assert_eq!(best, 128);
        assert_eq!(tried, vec![256, 192, 128, 96]);
    }

    #[test]
    fn walks_up_when_larger_is_faster() {
        let cost = |t: u32| Ok(1.0 + 100.0 / f64::from(t));
        let (best, tried) = tune(192, HD, cost);
        assert_eq!(best, 512);
        assert_eq!(tried, vec![192, 128, 256, 384, 512]);
    }

    #[test]
    fn keeps_the_default_when_the_difference_is_noise() {
        let cost = |t: u32| Ok(if t == 256 { 1.0 } else { 0.98 });
        let (best, tried) = tune(256, HD, cost);
        assert_eq!(best, 256);
        assert_eq!(tried, vec![256, 192, 384]);
    }

    #[test]
    fn never_tries_tiles_slow_enough_to_trip_the_watchdog() {
        // A slow GPU, 13–20 s per 1080p frame, faster with larger tiles: at
        // 192 px a 256 px tile would take ~0.42 s, so it's never tried.
        let cost = |t: u32| Ok(20.0 * 128.0 / f64::from(t));
        let (best, tried) = tune(128, HD, cost);
        assert_eq!(best, 192);
        assert!(tried.iter().all(|&t| t <= 192), "{tried:?}");
    }

    #[test]
    fn out_of_memory_rules_out_that_size_and_larger() {
        let cost = |t: u32| {
            if t >= 384 {
                Err(())
            } else {
                Ok(10.0 / f64::from(t))
            }
        };
        let (best, tried) = tune(256, HD, cost);
        assert_eq!(best, 256);
        assert_eq!(tried, vec![256, 192, 384]);

        // The default itself doesn't fit: step down until one does.
        let cost = |t: u32| {
            if t > 128 {
                Err(())
            } else {
                Ok(10.0 / f64::from(t))
            }
        };
        let (best, tried) = tune(256, HD, cost);
        assert_eq!(best, 128);
        assert_eq!(tried, vec![256, 192, 128, 96]);
    }

    #[test]
    fn off_ladder_start_and_stale_records() {
        let mut t = TileTuner::new(200, HD);
        assert_eq!(t.tile(), 200);
        assert_eq!(t.record(999, 1.0, false), None, "a size it didn't ask for");
        assert!(t.timings().is_empty());
    }

    #[test]
    fn halves_until_it_fits_and_remembers() {
        let mut tc = TileController::new(512);
        let mut tried = Vec::new();
        let (v, out) = tc
            .run(|t| {
                tried.push(t);
                if t > 128 {
                    Err(EngineError::OutOfDeviceMemory { tile: t })
                } else {
                    Ok(t)
                }
            })
            .expect("fits");
        assert_eq!(tried, vec![512, 256, 128]);
        assert_eq!((v, out.reduced_to), (128, Some(128)));
        // Next frame starts at the learned size, with no repeated OOMs.
        let (_, out) = tc.run(Ok).expect("ok");
        assert_eq!(out.reduced_to, None);
        assert_eq!(tc.tile(), 128);
    }

    #[test]
    fn gives_up_at_minimum_and_passes_other_errors() {
        let mut tc = TileController::new(128);
        let r: Result<((), _), _> = tc.run(|t| Err(EngineError::OutOfDeviceMemory { tile: t }));
        assert_eq!(r, Err(EngineError::OutOfDeviceMemory { tile: MIN_TILE }));
        let mut tc = TileController::new(512);
        let r: Result<((), _), _> = tc.run(|_| Err(EngineError::DeviceLost("x".into())));
        assert!(matches!(r, Err(EngineError::DeviceLost(_))));
        assert_eq!(tc.tile(), 512, "device loss isn't a tile problem");
    }
}
