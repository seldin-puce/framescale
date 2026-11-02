//! Recovery policy: what to do when a worker exits. Pure logic, no I/O, so
//! every branch is unit-tested.

use std::time::Duration;

use fs_types::FailureKind;
use fs_types::worker::exit;

/// How the worker run ended.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerExit {
    /// `None` = killed by a signal / crashed.
    pub code: Option<i32>,
    /// The last fatal `Error` event, if the worker managed to send one.
    pub error: Option<(FailureKind, String)>,
    /// Segments committed during this run (progress resets the strike count).
    pub committed: u32,
    /// Segment the run started at.
    pub start_segment: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Done,
    /// User-requested stop (hard pause / cancel) completed.
    Stopped,
    Restart(Restart),
    /// Out of disk: wait for space, then resume (never fail a 10-hour batch for this).
    PauseForDisk,
    Fail(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Restart {
    pub delay: Duration,
    /// Multiply the tile size by this (device lost / TDR / OOM at start).
    pub tile_scale: Option<f32>,
    /// Retry with software decoding.
    pub disable_hw_decode: bool,
    /// Re-plan with a software encoder.
    pub software_encoder: bool,
}

const BACKOFF: [Duration; 3] = [
    Duration::from_secs(1),
    Duration::from_secs(4),
    Duration::from_secs(15),
];
const MAX_STRIKES: u32 = 3;

#[derive(Debug, Clone, Default)]
pub struct RecoveryPolicy {
    strikes: u32,
    strike_segment: Option<u32>,
    hw_decode_disabled: bool,
    encoder_fallback_used: bool,
}

impl RecoveryPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hw_decode_disabled(&self) -> bool {
        self.hw_decode_disabled
    }

    pub fn decide(&mut self, e: &WorkerExit) -> Action {
        let kind = e.error.as_ref().map(|(k, _)| *k);
        let msg = e
            .error
            .as_ref()
            .map_or_else(String::new, |(_, m)| m.clone());
        match e.code {
            Some(exit::OK) => return Action::Done,
            Some(exit::STOPPED) => return Action::Stopped,
            Some(exit::DISK_FULL) => return Action::PauseForDisk,
            Some(exit::LICENSE) => return Action::Fail(msg),
            Some(exit::USAGE) => return Action::Fail(format!("worker protocol error: {msg}")),
            _ => {}
        }
        if e.committed > 0 {
            // Progress was made: whatever went wrong isn't stuck on one segment.
            self.strikes = 0;
            self.strike_segment = None;
        }
        match kind {
            Some(FailureKind::DiskFull) => return Action::PauseForDisk,
            Some(FailureKind::LicenseRequired | FailureKind::InvalidInput) => {
                return Action::Fail(msg);
            }
            Some(FailureKind::VramOom) => {
                return Action::Fail(format!(
                    "not enough GPU memory even at the minimum tile size. Choose a lighter model, \
                     a smaller scale, or another GPU ({msg})"
                ));
            }
            Some(FailureKind::DecodeFailed) if !self.hw_decode_disabled => {
                self.hw_decode_disabled = true;
                return Action::Restart(Restart {
                    disable_hw_decode: true,
                    ..Default::default()
                });
            }
            Some(FailureKind::EncoderUnavailable) => {
                if self.encoder_fallback_used {
                    return Action::Fail(format!("no usable encoder: {msg}"));
                }
                self.encoder_fallback_used = true;
                return Action::Restart(Restart {
                    software_encoder: true,
                    ..Default::default()
                });
            }
            _ => {}
        }
        // Device lost, crash, or internal error: retry the same segment with
        // backoff, smaller tiles (shorter dispatches avoid TDR), then give up.
        let failing = e.start_segment + e.committed;
        if self.strike_segment != Some(failing) {
            self.strike_segment = Some(failing);
            self.strikes = 0;
        }
        self.strikes += 1;
        if self.strikes > MAX_STRIKES {
            let what = if msg.is_empty() {
                "worker crashed".to_owned()
            } else {
                msg
            };
            return Action::Fail(format!(
                "segment {failing} failed {MAX_STRIKES} times: {what}"
            ));
        }
        let device_trouble = matches!(kind, Some(FailureKind::DeviceLost))
            || e.code == Some(exit::DEVICE_LOST)
            || e.code.is_none();
        Action::Restart(Restart {
            delay: BACKOFF[(self.strikes - 1) as usize],
            tile_scale: device_trouble.then_some(0.5),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(code: Option<i32>, kind: Option<FailureKind>, committed: u32, start: u32) -> WorkerExit {
        WorkerExit {
            code,
            error: kind.map(|k| (k, "boom".into())),
            committed,
            start_segment: start,
        }
    }

    #[test]
    fn clean_outcomes() {
        let mut p = RecoveryPolicy::new();
        assert_eq!(p.decide(&ex(Some(exit::OK), None, 3, 0)), Action::Done);
        assert_eq!(
            p.decide(&ex(Some(exit::STOPPED), None, 1, 0)),
            Action::Stopped
        );
        assert_eq!(
            p.decide(&ex(
                Some(exit::DISK_FULL),
                Some(FailureKind::DiskFull),
                0,
                0
            )),
            Action::PauseForDisk
        );
        assert!(matches!(
            p.decide(&ex(
                Some(exit::LICENSE),
                Some(FailureKind::LicenseRequired),
                0,
                0
            )),
            Action::Fail(_)
        ));
    }

    #[test]
    fn device_lost_backs_off_shrinks_tiles_then_poisons() {
        let mut p = RecoveryPolicy::new();
        let lost = ex(Some(exit::DEVICE_LOST), Some(FailureKind::DeviceLost), 0, 7);
        for want in BACKOFF {
            match p.decide(&lost) {
                Action::Restart(r) => {
                    assert_eq!(r.delay, want);
                    assert_eq!(r.tile_scale, Some(0.5));
                }
                a => panic!("expected restart, got {a:?}"),
            }
        }
        match p.decide(&lost) {
            Action::Fail(m) => assert!(m.contains("segment 7 failed 3 times"), "{m}"),
            a => panic!("expected poison, got {a:?}"),
        }
    }

    #[test]
    fn progress_resets_strikes() {
        let mut p = RecoveryPolicy::new();
        let crash = |committed, start| ex(None, None, committed, start);
        assert!(matches!(p.decide(&crash(0, 0)), Action::Restart(_)));
        assert!(matches!(p.decide(&crash(0, 0)), Action::Restart(_)));
        // Third run gets 5 segments further before crashing: fresh strike budget.
        match p.decide(&crash(5, 0)) {
            Action::Restart(r) => assert_eq!(r.delay, BACKOFF[0]),
            a => panic!("{a:?}"),
        }
    }

    #[test]
    fn decode_and_encoder_fallbacks_happen_once() {
        let mut p = RecoveryPolicy::new();
        let dec = ex(Some(exit::FATAL), Some(FailureKind::DecodeFailed), 0, 0);
        assert!(matches!(
            p.decide(&dec),
            Action::Restart(Restart {
                disable_hw_decode: true,
                ..
            })
        ));
        assert!(p.hw_decode_disabled());
        // Software decode failing too is a regular strike, not another fallback.
        assert!(matches!(
            p.decide(&dec),
            Action::Restart(Restart {
                disable_hw_decode: false,
                ..
            })
        ));

        let enc = ex(
            Some(exit::FATAL),
            Some(FailureKind::EncoderUnavailable),
            0,
            0,
        );
        assert!(matches!(
            p.decide(&enc),
            Action::Restart(Restart {
                software_encoder: true,
                ..
            })
        ));
        assert!(matches!(p.decide(&enc), Action::Fail(_)));
    }

    #[test]
    fn oom_at_min_tile_fails_with_guidance() {
        let mut p = RecoveryPolicy::new();
        match p.decide(&ex(Some(exit::FATAL), Some(FailureKind::VramOom), 0, 0)) {
            Action::Fail(m) => assert!(m.contains("lighter model")),
            a => panic!("{a:?}"),
        }
    }
}
