//! The ANALYZE pass: one fast hardware decode that detects interlacing,
//! variable frame rate and scene cuts, and counts conformed frames exactly.

use std::path::Path;

use fs_types::Rational;
use fs_types::media::{AnalysisReport, FieldOrder};
use fs_types::worker::HwDecode;

use crate::argv::{Argv, hwaccel_args};

/// `analysis_height` is the downscale before scdet (270 lines is plenty for
/// cut detection, and it keeps the pass memory-bound rather than compute-bound).
pub fn argv(
    input: &Path,
    stream_index: u32,
    hw: &HwDecode,
    conform_rate: Rational,
    scene_threshold: Option<f32>,
) -> Argv {
    let mut a = Argv::new();
    // `info` level: idet/vfrdet/scdet report at info. `-progress` on stdout gives
    // the exact conformed frame count.
    a.args(["-hide_banner", "-nostdin", "-nostats", "-loglevel", "info"]);
    a.args(["-progress", "pipe:1"]);
    a.args(hwaccel_args(hw));
    a.arg("-i").arg(input);
    a.args(["-map", &format!("0:{stream_index}")]);
    let mut vf = vec![
        "idet".to_owned(),
        "vfrdet".to_owned(),
        "setpts=PTS-STARTPTS".to_owned(),
        format!("fps={conform_rate}:round=near"),
    ];
    if let Some(t) = scene_threshold {
        vf.push("scale=-2:270:flags=fast_bilinear".into());
        vf.push(format!("scdet=threshold={t}"));
    }
    a.args(["-vf", &vf.join(",")]);
    a.args(["-an", "-sn", "-dn", "-f", "null", "-"]);
    a
}

/// Accumulates analysis results from the pass's stderr (and stdout progress).
#[derive(Debug, Default)]
pub struct AnalyzeParser {
    /// (ratio, vfr_count, min_duration, max_duration) from vfrdet.
    vfr: Option<(f64, u64, u64, u64)>,
    idet: Option<[u64; 4]>,
    cut_times: Vec<f64>,
    frames: u64,
}

impl AnalyzeParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a stderr line. FFmpeg prints each filter summary twice (a probe
    /// instance with zeros first), so later values always win.
    pub fn feed_stderr(&mut self, line: &str) {
        if let Some(rest) = after(line, "lavfi.scd.time:") {
            if let Ok(t) = rest.trim().parse::<f64>() {
                self.cut_times.push(t);
            }
        } else if let Some(rest) = after(line, "VFR:") {
            // "VFR:0.826087 (95/20) min: 41 max: 84 avg: 53"  |  "VFR:-nan (0/0)"
            let ratio = rest
                .split_whitespace()
                .next()
                .and_then(|r| r.parse::<f64>().ok());
            let counts = rest
                .split_once('(')
                .and_then(|(_, r)| r.split_once(')'))
                .and_then(|(inner, _)| inner.split_once('/'))
                .and_then(|(v, c)| {
                    Some((v.trim().parse::<u64>().ok()?, c.trim().parse::<u64>().ok()?))
                });
            let field = |k: &str| -> u64 {
                after(rest, k)
                    .and_then(|r| r.split_whitespace().next())
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0)
            };
            if let Some((vfr, _cfr)) = counts {
                let ratio: f64 = ratio.filter(|r| r.is_finite()).unwrap_or(0.0);
                self.vfr = Some((ratio, vfr, field("min:"), field("max:")));
            }
        } else if let Some(rest) = after(line, "Multi frame detection:") {
            // "TFF:    34 BFF:     0 Progressive:    82 Undetermined:     0"
            let nums: Vec<u64> = rest
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            if let [tff, bff, prog, und] = nums[..] {
                self.idet = Some([tff, bff, prog, und]);
            }
        }
    }

    /// Feed a stdout `-progress` line (only `frame=` matters here).
    pub fn feed_progress(&mut self, line: &str) {
        if let Some(v) = line.trim().strip_prefix("frame=") {
            self.frames = v.trim().parse().unwrap_or(self.frames);
        }
    }

    pub fn finish(mut self, conform_rate: Rational) -> AnalysisReport {
        let (vfr_ratio, is_vfr) = match self.vfr {
            // vfrdet flags *any* duration change, including the ±1 tick jitter
            // of millisecond timebases (Matroska: 24 fps → 41/42 ms). Real VFR
            // also needs a >10 % spread between the shortest and longest frame.
            Some((ratio, vfr, min, max)) => {
                let spread = min > 0 && max * 10 > min * 11;
                (ratio, vfr > 0 && ratio >= 0.01 && spread)
            }
            None => (0.0, false),
        };
        let field_order = match self.idet {
            // Interlaced only when field-based frames clearly dominate:
            // idet throws false positives on synthetic or very sharp content.
            Some([tff, bff, prog, _]) if tff + bff > 2 * prog && tff + bff >= 10 => {
                if tff >= bff {
                    FieldOrder::TopFirst
                } else {
                    FieldOrder::BottomFirst
                }
            }
            Some(_) => FieldOrder::Progressive,
            None => FieldOrder::Unknown,
        };
        self.cut_times.sort_by(f64::total_cmp);
        let mut cuts: Vec<u64> = self
            .cut_times
            .iter()
            .map(|t| conform_rate.frame_at(*t))
            .filter(|&i| i > 0)
            .collect();
        cuts.dedup();
        AnalysisReport {
            conform_rate,
            vfr_ratio,
            is_vfr,
            field_order,
            scene_cuts: cuts,
            total_frames: self.frames,
        }
    }
}

fn after<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    line.find(marker).map(|i| &line[i + marker.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;

    // Captured from the pinned FFmpeg 8.1.3 on a 3×2 s, 24 fps clip with
    // every 5th frame dropped (VFR).
    const LOG: &str = "\
[Parsed_idet_0 @ 0x55] Multi frame detection: TFF:     0 BFF:     0 Progressive:     0 Undetermined:     0
[Parsed_vfrdet_1 @ 0x55] VFR:-nan (0/0)
[Parsed_scdet_5 @ 0x7f] lavfi.scd.score: 18.639, lavfi.scd.time: 2
[Parsed_scdet_5 @ 0x7f] lavfi.scd.score: 23.619, lavfi.scd.time: 4
[Parsed_idet_0 @ 0x7f] Multi frame detection: TFF:    34 BFF:     0 Progressive:    82 Undetermined:     0
[Parsed_vfrdet_1 @ 0x7f] VFR:0.826087 (95/20) min: 41 max: 84 avg: 53";

    #[test]
    fn parses_captured_log() {
        let mut p = AnalyzeParser::new();
        LOG.lines().for_each(|l| p.feed_stderr(l));
        p.feed_progress("frame=144");
        let r = p.finish(Rational::new(24, 1));
        assert_eq!(r.scene_cuts, vec![48, 96]);
        assert!(r.is_vfr);
        assert!((r.vfr_ratio - 0.826087).abs() < 1e-6);
        assert_eq!(
            r.field_order,
            FieldOrder::Progressive,
            "idet false positives ignored"
        );
        assert_eq!(r.total_frames, 144);
    }

    #[test]
    fn millisecond_timestamp_jitter_is_not_vfr() {
        let mut p = AnalyzeParser::new();
        p.feed_stderr("[Parsed_vfrdet_0 @ 0x7f] VFR:0.664336 (95/48) min: 41 max: 42 avg: 41");
        assert!(!p.finish(Rational::new(24, 1)).is_vfr);
    }

    #[test]
    fn cfr_nan_is_not_vfr_and_interlace_detected() {
        let mut p = AnalyzeParser::new();
        p.feed_stderr("[Parsed_vfrdet_1 @ 0x1] VFR:-nan (0/0)");
        p.feed_stderr("[Parsed_idet_0 @ 0x1] Multi frame detection: TFF:   900 BFF:     2 Progressive:    40 Undetermined:   10");
        let r = p.finish(Rational::new(25, 1));
        assert!(!r.is_vfr);
        assert_eq!(r.field_order, FieldOrder::TopFirst);
    }
}
