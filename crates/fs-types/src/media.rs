//! Media description: what ffprobe and the analysis pass tell us about a source.

use std::fmt;

use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// Exact rational (frame rates, aspect ratios). Kept exact so that
/// frame index ⇄ time conversions never drift over long renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct Rational {
    #[ts(type = "number")]
    pub num: i64,
    #[ts(type = "number")]
    pub den: i64,
}

impl Rational {
    pub const fn new(num: i64, den: i64) -> Self {
        Self { num, den }
    }

    pub fn is_valid(self) -> bool {
        self.num > 0 && self.den > 0
    }

    pub fn as_f64(self) -> f64 {
        if self.den == 0 {
            0.0
        } else {
            self.num as f64 / self.den as f64
        }
    }

    pub fn reduced(self) -> Self {
        let g = gcd(self.num.unsigned_abs(), self.den.unsigned_abs()).max(1) as i64;
        Self::new(self.num / g, self.den / g)
    }

    /// Multiply by an integer factor (e.g. interpolation ×2).
    pub fn times(self, k: i64) -> Self {
        Self::new(self.num * k, self.den).reduced()
    }

    /// Parse ffprobe's `"30000/1001"` form. `"0/0"` → `None`.
    pub fn parse(s: &str) -> Option<Self> {
        let (n, d) = s.split_once('/').unwrap_or((s, "1"));
        let r = Self::new(n.trim().parse().ok()?, d.trim().parse().ok()?);
        r.is_valid().then_some(r)
    }

    /// Seconds at which frame `index` starts, for a stream at this rate.
    pub fn frame_time(self, index: u64) -> f64 {
        index as f64 * self.den as f64 / self.num as f64
    }

    /// Frame index nearest to `seconds`, for a stream at this rate.
    pub fn frame_at(self, seconds: f64) -> u64 {
        (seconds * self.as_f64()).round().max(0.0) as u64
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

macro_rules! ffmpeg_enum {
    ($(#[$m:meta])* $name:ident { $($variant:ident => [$($alias:literal),+]),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
        #[serde(rename_all = "camelCase")]
        #[ts(export)]
        pub enum $name { $($variant,)+ Unknown }

        impl $name {
            /// Parse an ffprobe/ffmpeg name (first alias is canonical).
            pub fn from_ffmpeg(s: &str) -> Self {
                match s { $($($alias)|+ => Self::$variant,)+ _ => Self::Unknown }
            }
            /// Canonical FFmpeg option value, `None` for `Unknown`.
            pub fn ffmpeg_name(self) -> Option<&'static str> {
                match self { $(Self::$variant => Some(ffmpeg_enum!(@first $($alias),+)),)+ Self::Unknown => None }
            }
        }
    };
    (@first $first:literal $(, $rest:literal)*) => { $first };
}

ffmpeg_enum!(
    /// Colour primaries (`-color_primaries`).
    ColorPrimaries {
        Bt709 => ["bt709"],
        Bt470bg => ["bt470bg"],
        Smpte170m => ["smpte170m"],
        Bt2020 => ["bt2020"],
        DisplayP3 => ["smpte432"],
    }
);

ffmpeg_enum!(
    /// Transfer characteristics (`-color_trc`).
    TransferCharacteristic {
        Bt709 => ["bt709"],
        Smpte170m => ["smpte170m", "bt601"],
        Srgb => ["iec61966-2-1"],
        Pq => ["smpte2084"],
        Hlg => ["arib-std-b67"],
        Linear => ["linear"],
        Bt2020_10 => ["bt2020-10"],
        Bt2020_12 => ["bt2020-12"],
    }
);

ffmpeg_enum!(
    /// Matrix coefficients (`-colorspace`).
    MatrixCoefficients {
        Bt709 => ["bt709"],
        Bt470bg => ["bt470bg"],
        Smpte170m => ["smpte170m"],
        Bt2020Nc => ["bt2020nc", "bt2020_ncl"],
        Bt2020C => ["bt2020c", "bt2020_cl"],
        Rgb => ["gbr", "rgb"],
    }
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum ColorRange {
    /// "tv" / MPEG / 16–235.
    Limited,
    /// "pc" / JPEG / 0–255.
    Full,
}

impl ColorRange {
    pub fn from_ffmpeg(s: &str) -> Option<Self> {
        match s {
            "tv" | "mpeg" | "limited" => Some(Self::Limited),
            "pc" | "jpeg" | "full" => Some(Self::Full),
            _ => None,
        }
    }
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::Limited => "tv",
            Self::Full => "pc",
        }
    }
    /// zimg/zscale naming.
    pub fn zscale_name(self) -> &'static str {
        match self {
            Self::Limited => "limited",
            Self::Full => "full",
        }
    }
}

/// SMPTE ST 2086 mastering display colour volume, in the units used by the
/// HEVC/AV1 SEI/OBU: chromaticity × 50 000, luminance × 10 000 (cd/m²).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct MasteringDisplay {
    pub red: [u32; 2],
    pub green: [u32; 2],
    pub blue: [u32; 2],
    pub white: [u32; 2],
    pub max_luminance: u32,
    pub min_luminance: u32,
}

/// CTA-861.3 content light level (cd/m²).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ContentLightLevel {
    pub max_cll: u32,
    pub max_fall: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ColorProfile {
    pub primaries: ColorPrimaries,
    pub transfer: TransferCharacteristic,
    pub matrix: MatrixCoefficients,
    pub range: ColorRange,
    pub bit_depth: u8,
    pub mastering: Option<MasteringDisplay>,
    pub content_light: Option<ContentLightLevel>,
    /// True when one or more fields were missing from the source and inferred.
    pub inferred: bool,
}

impl ColorProfile {
    pub fn is_hdr(&self) -> bool {
        matches!(
            self.transfer,
            TransferCharacteristic::Pq | TransferCharacteristic::Hlg
        )
    }

    /// Whether the AI path must carry 16-bit samples (rgb48) to avoid banding.
    pub fn needs_high_precision(&self) -> bool {
        self.bit_depth > 8 || self.is_hdr()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum FieldOrder {
    Progressive,
    TopFirst,
    BottomFirst,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct VideoStream {
    pub index: u32,
    pub codec: String,
    pub profile: Option<String>,
    pub width: u32,
    pub height: u32,
    pub sample_aspect: Rational,
    pub r_frame_rate: Rational,
    pub avg_frame_rate: Rational,
    #[ts(type = "number | null")]
    pub frame_count: Option<u64>,
    pub start_time: f64,
    pub pix_fmt: String,
    pub color: ColorProfile,
    /// Display-matrix rotation in degrees (FFmpeg autorotates on decode).
    pub rotation: i32,
    pub field_order: FieldOrder,
    pub dolby_vision: bool,
    pub hdr10_plus: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct AudioStream {
    pub index: u32,
    pub codec: String,
    pub channels: u32,
    pub sample_rate: u32,
    pub start_time: f64,
    pub language: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct SubtitleStream {
    pub index: u32,
    pub codec: String,
    pub language: Option<String>,
    /// Image-based (PGS/VobSub/DVB): cannot be converted to text formats.
    pub bitmap: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct MediaInfo {
    pub path: String,
    pub format: String,
    pub duration_s: f64,
    #[ts(type = "number")]
    pub size_bytes: u64,
    #[ts(type = "number | null")]
    pub bit_rate: Option<u64>,
    pub start_time: f64,
    pub video: VideoStream,
    pub audio: Vec<AudioStream>,
    pub subtitles: Vec<SubtitleStream>,
    pub chapters: u32,
}

/// Result of the ANALYZE pass (one fast decode with idet/vfrdet/scdet).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct AnalysisReport {
    pub conform_rate: Rational,
    /// Fraction of frames vfrdet flagged as variable-duration.
    pub vfr_ratio: f64,
    pub is_vfr: bool,
    pub field_order: FieldOrder,
    /// Scene-cut frame indices in the CFR (conformed) timeline: a cut lies
    /// between frame `i - 1` and frame `i`.
    #[ts(type = "number[]")]
    pub scene_cuts: Vec<u64>,
    /// Frames in the conformed timeline.
    #[ts(type = "number")]
    pub total_frames: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_parse_and_time() {
        let r = Rational::parse("30000/1001").expect("ntsc");
        assert!((r.as_f64() - 29.97).abs() < 0.001);
        assert_eq!(Rational::parse("0/0"), None);
        assert_eq!(Rational::parse("25"), Some(Rational::new(25, 1)));
        assert_eq!(r.frame_at(r.frame_time(123_456)), 123_456);
        assert_eq!(
            Rational::new(48000, 1001).times(2),
            Rational::new(96000, 1001)
        );
        assert_eq!(Rational::new(60, 2).reduced(), Rational::new(30, 1));
    }

    #[test]
    fn color_enums_roundtrip() {
        assert_eq!(
            TransferCharacteristic::from_ffmpeg("smpte2084"),
            TransferCharacteristic::Pq
        );
        assert_eq!(TransferCharacteristic::Pq.ffmpeg_name(), Some("smpte2084"));
        assert_eq!(
            MatrixCoefficients::from_ffmpeg("bt2020_ncl"),
            MatrixCoefficients::Bt2020Nc
        );
        assert_eq!(
            ColorPrimaries::from_ffmpeg("weird"),
            ColorPrimaries::Unknown
        );
        assert_eq!(ColorPrimaries::Unknown.ffmpeg_name(), None);
    }
}
