//! Classify FFmpeg stderr lines into actionable issues.

use fs_types::FailureKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfmpegIssue {
    NoSpace,
    PermissionDenied,
    /// Encoder can't be opened on this machine (missing driver, unsupported GPU).
    EncoderUnavailable,
    /// NVENC concurrent session cap reached, or similar transient limits.
    EncoderSessionLimit,
    HwDecodeFailed,
    /// Damaged packets mid-stream. FFmpeg conceals them; we count and warn.
    CorruptInput,
    /// Input can't be read at all.
    InvalidInput,
    BrokenPipe,
    OutOfMemory,
}

impl FfmpegIssue {
    pub fn failure_kind(self) -> Option<FailureKind> {
        Some(match self {
            Self::NoSpace => FailureKind::DiskFull,
            Self::PermissionDenied => FailureKind::Io,
            Self::EncoderUnavailable | Self::EncoderSessionLimit => FailureKind::EncoderUnavailable,
            Self::HwDecodeFailed => FailureKind::DecodeFailed,
            Self::InvalidInput => FailureKind::InvalidInput,
            Self::OutOfMemory => FailureKind::Internal,
            Self::CorruptInput | Self::BrokenPipe => return None,
        })
    }
}

const RULES: &[(&str, FfmpegIssue)] = &[
    ("No space left on device", FfmpegIssue::NoSpace),
    ("Disk quota exceeded", FfmpegIssue::NoSpace),
    ("Permission denied", FfmpegIssue::PermissionDenied),
    (
        "OpenEncodeSessionEx failed: out of memory",
        FfmpegIssue::EncoderSessionLimit,
    ),
    ("incompatible client key", FfmpegIssue::EncoderSessionLimit),
    ("No capable devices found", FfmpegIssue::EncoderUnavailable),
    (
        "Cannot load libnvidia-encode",
        FfmpegIssue::EncoderUnavailable,
    ),
    ("Cannot load nvEncodeAPI", FfmpegIssue::EncoderUnavailable),
    (
        "Driver does not support the required nvenc API version",
        FfmpegIssue::EncoderUnavailable,
    ),
    (
        "Error initializing an MFX session",
        FfmpegIssue::EncoderUnavailable,
    ),
    (
        "Error initializing an internal MFX session",
        FfmpegIssue::EncoderUnavailable,
    ),
    ("AMF failed to initialise", FfmpegIssue::EncoderUnavailable),
    (
        "DLL amfrt64.dll failed to open",
        FfmpegIssue::EncoderUnavailable,
    ),
    (
        "No usable encoding profile found",
        FfmpegIssue::EncoderUnavailable,
    ),
    (
        "No usable encoding entrypoint found",
        FfmpegIssue::EncoderUnavailable,
    ),
    (
        "Error while opening encoder",
        FfmpegIssue::EncoderUnavailable,
    ),
    (
        "cannot create compression session",
        FfmpegIssue::EncoderUnavailable,
    ),
    ("Failed setup for format", FfmpegIssue::HwDecodeFailed),
    (
        "hwaccel initialisation returned error",
        FfmpegIssue::HwDecodeFailed,
    ),
    ("Device creation failed", FfmpegIssue::HwDecodeFailed),
    (
        "Failed to initialise VAAPI connection",
        FfmpegIssue::HwDecodeFailed,
    ),
    (
        "No device available for decoder",
        FfmpegIssue::HwDecodeFailed,
    ),
    (
        "Invalid data found when processing input",
        FfmpegIssue::InvalidInput,
    ),
    ("moov atom not found", FfmpegIssue::InvalidInput),
    ("Invalid NAL unit size", FfmpegIssue::CorruptInput),
    ("error while decoding MB", FfmpegIssue::CorruptInput),
    ("corrupt decoded frame", FfmpegIssue::CorruptInput),
    ("concealing", FfmpegIssue::CorruptInput),
    ("missing picture in access unit", FfmpegIssue::CorruptInput),
    ("Broken pipe", FfmpegIssue::BrokenPipe),
    ("Cannot allocate memory", FfmpegIssue::OutOfMemory),
];

/// Classify one stderr line. The first matching rule wins.
pub fn classify(line: &str) -> Option<FfmpegIssue> {
    RULES
        .iter()
        .find(|(pat, _)| line.contains(pat))
        .map(|(_, i)| *i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_world_lines() {
        let cases = [
            (
                "[out#0/matroska @ 0x5] Error writing trailer: No space left on device",
                FfmpegIssue::NoSpace,
            ),
            (
                "[hevc_nvenc @ 0x1] OpenEncodeSessionEx failed: out of memory (10): (no details)",
                FfmpegIssue::EncoderSessionLimit,
            ),
            (
                "[av1_nvenc @ 0x1] No capable devices found",
                FfmpegIssue::EncoderUnavailable,
            ),
            (
                "[hevc @ 0x2] Invalid NAL unit size (1234 > 99).",
                FfmpegIssue::CorruptInput,
            ),
            (
                "[h264 @ 0x3] error while decoding MB 12 34, bytestream -5",
                FfmpegIssue::CorruptInput,
            ),
            (
                "[AVHWDeviceContext @ 0x4] Failed to initialise VAAPI connection: -1 (unknown libva error).",
                FfmpegIssue::HwDecodeFailed,
            ),
            (
                "in.mp4: Invalid data found when processing input",
                FfmpegIssue::InvalidInput,
            ),
            (
                "av_interleaved_write_frame(): Broken pipe",
                FfmpegIssue::BrokenPipe,
            ),
        ];
        for (line, want) in cases {
            assert_eq!(classify(line), Some(want), "{line}");
        }
        assert_eq!(classify("frame=  100 fps= 25"), None);
        assert_eq!(
            FfmpegIssue::NoSpace.failure_kind(),
            Some(FailureKind::DiskFull)
        );
        assert_eq!(FfmpegIssue::CorruptInput.failure_kind(), None);
    }
}
