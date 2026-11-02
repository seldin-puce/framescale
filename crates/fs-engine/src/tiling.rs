//! VRAM-OOM recovery: halve the tile and retry the same frame.

use crate::EngineError;

pub const MIN_TILE: u32 = 64;

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

#[cfg(test)]
mod tests {
    use super::*;

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
