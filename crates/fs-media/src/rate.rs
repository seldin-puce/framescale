//! Conform-rate selection: the constant rate every source is resampled to
//! before inference. Frame index is our clock, so this rate defines the clock.

use fs_types::Rational;

/// Broadcast/cinema rates we snap to.
pub const STANDARD_RATES: [Rational; 13] = [
    Rational::new(24000, 1001),
    Rational::new(24, 1),
    Rational::new(25, 1),
    Rational::new(30000, 1001),
    Rational::new(30, 1),
    Rational::new(48, 1),
    Rational::new(50, 1),
    Rational::new(60000, 1001),
    Rational::new(60, 1),
    Rational::new(100, 1),
    Rational::new(120000, 1001),
    Rational::new(120, 1),
    Rational::new(240, 1),
];

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / b
}

fn nearest_standard(fps: f64, tolerance: f64) -> Option<Rational> {
    STANDARD_RATES
        .iter()
        .copied()
        .map(|r| (r, rel_diff(fps, r.as_f64())))
        .filter(|(_, d)| *d <= tolerance)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(r, _)| r)
}

/// Choose the conform rate from ffprobe's nominal (`r_frame_rate`) and
/// average (`avg_frame_rate`) rates.
///
/// 1. If the nominal rate is standard and the average agrees within 5 %
///    (typical phone VFR: nominal 30, average 29.7), use the nominal rate.
/// 2. Otherwise snap the average to a standard rate within 2 %.
/// 3. Otherwise keep the average, expressed over 1 or 1001.
pub fn conform_rate(nominal: Option<Rational>, average: Option<Rational>) -> Rational {
    let nominal_std = nominal.and_then(|r| nearest_standard(r.as_f64(), 0.0005));
    match (nominal_std, average) {
        (Some(n), Some(a)) if rel_diff(a.as_f64(), n.as_f64()) <= 0.05 => return n,
        (Some(n), None) => return n,
        _ => {}
    }
    let Some(a) = average.or(nominal) else {
        return Rational::new(30, 1); // unreachable for real video; a safe default
    };
    if let Some(s) = nearest_standard(a.as_f64(), 0.02) {
        return s;
    }
    let exact = a.reduced();
    if exact.den <= 1001 {
        return exact;
    }
    let fps = a.as_f64();
    if (fps - fps.round()).abs() < 0.01 {
        Rational::new(fps.round() as i64, 1)
    } else {
        Rational::new((fps * 1001.0).round() as i64, 1001).reduced()
    }
}

/// Frames per segment for a target segment duration.
pub fn segment_frames(rate: Rational, seconds: f64) -> u64 {
    (rate.as_f64() * seconds).round().max(1.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Option<Rational> {
        Some(Rational::new(n, d))
    }

    #[test]
    fn cfr_film() {
        assert_eq!(
            conform_rate(r(24000, 1001), r(24000, 1001)),
            Rational::new(24000, 1001)
        );
    }

    #[test]
    fn phone_vfr_uses_nominal() {
        assert_eq!(
            conform_rate(r(30, 1), r(2_970_000, 100_000)),
            Rational::new(30, 1)
        );
    }

    #[test]
    fn odd_nominal_snaps_average() {
        // Some muxers report a timebase-ish nominal rate (e.g. 90000/1).
        assert_eq!(
            conform_rate(r(90000, 1), r(5994, 100)),
            Rational::new(60000, 1001)
        );
    }

    #[test]
    fn nonstandard_kept() {
        assert_eq!(conform_rate(r(15, 1), r(15, 1)), Rational::new(15, 1));
        assert_eq!(conform_rate(None, r(1250, 100)), Rational::new(25, 2));
    }

    #[test]
    fn segment_sizes() {
        assert_eq!(segment_frames(Rational::new(24000, 1001), 4.0), 96);
        assert_eq!(segment_frames(Rational::new(60, 1), 4.0), 240);
    }
}
