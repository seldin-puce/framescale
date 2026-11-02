//! Frame-rate multiplication that never morphs across hard cuts.
//!
//! For each input pair (i, i+1) the scheduler emits frame i followed by
//! `factor − 1` intermediates. At a cut, the intermediates are *duplicates of
//! frame i*: RIFE would otherwise hallucinate a cross-dissolve "morph" between
//! two unrelated shots. The final frame is repeated `factor − 1` times, so the
//! output is exactly `factor × N` frames and A/V sync holds.

use std::collections::BTreeSet;

/// Mean-luma-difference threshold for the fallback cut detector.
pub const FALLBACK_CUT_THRESHOLD: f32 = 0.30;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Synth {
    /// Interpolate at `t ∈ (0, 1)`.
    Blend(f32),
    /// Repeat the earlier frame (scene cut, or end of stream).
    Hold,
}

#[derive(Debug, Clone)]
pub struct Scheduler {
    factor: u32,
    cuts: BTreeSet<u64>,
}

impl Scheduler {
    /// `cuts`: frame indices `j` where a cut lies between `j − 1` and `j`.
    pub fn new(factor: u32, cuts: impl IntoIterator<Item = u64>) -> Self {
        Self {
            factor: factor.max(1),
            cuts: cuts.into_iter().collect(),
        }
    }

    pub fn factor(&self) -> u32 {
        self.factor
    }

    pub fn is_cut(&self, next_index: u64) -> bool {
        self.cuts.contains(&next_index)
    }

    /// Intermediates between input frame `i` and `i + 1`. `pixel_cut` is the
    /// fallback detector's verdict for this pair.
    pub fn between(&self, i: u64, pixel_cut: bool) -> Vec<Synth> {
        let n = self.factor - 1;
        if self.is_cut(i + 1) || pixel_cut {
            return vec![Synth::Hold; n as usize];
        }
        (1..=n)
            .map(|k| Synth::Blend(k as f32 / self.factor as f32))
            .collect()
    }

    /// Trailing copies after the last frame.
    pub fn tail(&self) -> Vec<Synth> {
        vec![Synth::Hold; (self.factor - 1) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_blends_across_a_cut() {
        let s = Scheduler::new(4, [48]);
        assert_eq!(
            s.between(10, false),
            vec![Synth::Blend(0.25), Synth::Blend(0.5), Synth::Blend(0.75)]
        );
        assert_eq!(
            s.between(47, false),
            vec![Synth::Hold; 3],
            "cut between 47 and 48"
        );
        assert_eq!(s.between(48, false).len(), 3);
        assert_eq!(
            s.between(20, true),
            vec![Synth::Hold; 3],
            "pixel-difference backstop"
        );
        assert_eq!(s.tail().len(), 3);
    }

    #[test]
    fn factor_one_is_identity() {
        let s = Scheduler::new(1, []);
        assert!(s.between(0, false).is_empty() && s.tail().is_empty());
    }
}
