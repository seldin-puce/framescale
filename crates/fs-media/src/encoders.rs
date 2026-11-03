//! Encoder selection per codec, OS and GPU vendor, plus rate-control and HDR
//! arguments.
//!
//! Selection is based on encoders *verified usable on this machine*
//! ([`probe_usable`]). A build listing `av1_nvenc` doesn't mean the GPU can
//! encode AV1 (RTX 40+ only).

use std::collections::BTreeSet;
use std::path::Path;
use std::process::{Command, Stdio};

use fs_types::media::ColorProfile;
use fs_types::pipeline::{EncoderPreference, VideoCodec};
use fs_types::worker::HdrEmbedding;
use fs_types::{Rational, Vendor};

use crate::argv;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Windows,
    MacOs,
    Linux,
}

impl Os {
    pub const fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::MacOs
        } else {
            Self::Linux
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncoderEnv {
    pub os: Os,
    pub vendor: Option<Vendor>,
    /// Encoders that opened successfully in a probe encode.
    pub usable: BTreeSet<String>,
    /// `/dev/dri/renderD*` for VAAPI encoders.
    pub vaapi_device: Option<String>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EncoderError {
    #[error("{codec:?} is not available: {reason}")]
    Unavailable { codec: VideoCodec, reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncoderSelection {
    pub encoder: String,
    pub args: Vec<String>,
    pub pix_fmt: String,
    pub vaapi_device: Option<String>,
    pub hdr: HdrEmbedding,
    pub out_bit_depth: u8,
    /// A software encoder was chosen although hardware was preferred.
    pub fell_back: bool,
}

/// Rate-control inputs.
#[derive(Debug, Clone, Copy)]
pub struct EncodeTarget {
    /// CRF-like quality on 0–51 (lower is better).
    pub quality: u8,
    pub width: u32,
    pub height: u32,
    pub rate: Rational,
}

fn hw_candidates(codec: VideoCodec, os: Os, vendor: Option<Vendor>) -> Vec<&'static str> {
    use Vendor::*;
    let fam = match codec {
        VideoCodec::Hevc10 => "hevc",
        VideoCodec::Av1 => "av1",
        VideoCodec::H264 => "h264",
        VideoCodec::ProRes422Hq | VideoCodec::ProRes4444 => {
            return if os == Os::MacOs {
                vec!["prores_videotoolbox"]
            } else {
                vec![]
            };
        }
    };
    let by_api = |api: &str| -> Option<&'static str> {
        const ALL: [&str; 14] = [
            "hevc_nvenc",
            "av1_nvenc",
            "h264_nvenc",
            "hevc_amf",
            "av1_amf",
            "h264_amf",
            "hevc_qsv",
            "av1_qsv",
            "h264_qsv",
            "hevc_vaapi",
            "av1_vaapi",
            "h264_vaapi",
            "hevc_videotoolbox",
            "h264_videotoolbox",
        ];
        ALL.into_iter()
            .find(|e| e.starts_with(fam) && e.ends_with(api))
    };
    let apis: Vec<&str> = match (os, vendor) {
        (Os::MacOs, _) => vec!["videotoolbox"],
        (_, Some(Nvidia)) => vec!["nvenc"],
        (Os::Windows, Some(Amd)) => vec!["amf"],
        (Os::Windows, Some(Intel)) => vec!["qsv"],
        (Os::Linux, Some(Amd)) => vec!["vaapi"],
        (Os::Linux, Some(Intel)) => vec!["vaapi", "qsv"],
        (Os::Windows, _) => vec!["nvenc", "amf", "qsv"],
        (Os::Linux, _) => vec!["nvenc", "vaapi", "qsv"],
    };
    apis.into_iter().filter_map(by_api).collect()
}

fn sw_candidate(codec: VideoCodec, color: &ColorProfile) -> Option<&'static str> {
    match codec {
        VideoCodec::Av1 => Some("libsvtav1"),
        // kvazaar (BSD) is 8-bit only: acceptable for SDR 8-bit sources.
        VideoCodec::Hevc10 if !color.needs_high_precision() => Some("libkvazaar"),
        VideoCodec::H264 if !color.is_hdr() => Some("libopenh264"),
        _ => None,
    }
}

/// Every encoder worth probing on this machine (for [`probe_usable`]).
pub fn all_candidates(os: Os, vendor: Option<Vendor>) -> Vec<&'static str> {
    let mut v = Vec::new();
    for c in [
        VideoCodec::Hevc10,
        VideoCodec::Av1,
        VideoCodec::H264,
        VideoCodec::ProRes422Hq,
    ] {
        v.extend(hw_candidates(c, os, vendor));
    }
    v.extend(["libsvtav1", "libkvazaar", "libopenh264"]);
    v
}

/// Run a tiny test encode for each candidate in parallel and keep the ones that work.
pub fn probe_usable(
    ffmpeg: &Path,
    candidates: &[&str],
    vaapi_device: Option<&str>,
) -> BTreeSet<String> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = candidates
            .iter()
            .map(|enc| {
                scope.spawn(move || {
                    let pix = if enc.starts_with("lib") {
                        "yuv420p"
                    } else {
                        "nv12"
                    };
                    let a = argv::encoder_probe(enc, pix, vaapi_device);
                    let ok = Command::new(ffmpeg)
                        .args(a.as_slice())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .is_ok_and(|s| s.success());
                    ok.then(|| (*enc).to_owned())
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().ok().flatten())
            .collect()
    })
}

pub fn select(
    codec: VideoCodec,
    pref: EncoderPreference,
    color: &ColorProfile,
    target: EncodeTarget,
    env: &EncoderEnv,
) -> Result<EncoderSelection, EncoderError> {
    if codec == VideoCodec::H264 && color.is_hdr() {
        return Err(EncoderError::Unavailable {
            codec,
            reason: "H.264 cannot carry HDR (PQ/HLG); choose HEVC or AV1".into(),
        });
    }
    let hw = hw_candidates(codec, env.os, env.vendor);
    let sw = sw_candidate(codec, color);
    let first_hw = hw.iter().copied().find(|e| env.usable.contains(*e));
    let sw_ok = sw.filter(|e| env.usable.contains(*e));
    let (encoder, fell_back) = match pref {
        EncoderPreference::Hardware => (first_hw, false),
        EncoderPreference::Software => (sw_ok, false),
        EncoderPreference::Auto => match first_hw {
            Some(e) => (Some(e), false),
            None => (sw_ok, !hw.is_empty()),
        },
    };
    let Some(encoder) = encoder else {
        let reason = match (codec, sw) {
            (VideoCodec::ProRes422Hq | VideoCodec::ProRes4444, _) => {
                "ProRes is only available on macOS (VideoToolbox)".to_owned()
            }
            (VideoCodec::Hevc10, None) => {
                "no hardware HEVC encoder is usable, and the only software HEVC encoder \
                 (kvazaar) is 8-bit SDR only. Choose AV1 (SVT-AV1) instead"
                    .to_owned()
            }
            _ => format!("no usable encoder among {hw:?} / {sw:?}"),
        };
        return Err(EncoderError::Unavailable { codec, reason });
    };
    Ok(build(encoder, codec, color, target, env, fell_back))
}

fn build(
    enc: &str,
    codec: VideoCodec,
    color: &ColorProfile,
    t: EncodeTarget,
    env: &EncoderEnv,
    fell_back: bool,
) -> EncoderSelection {
    let ten_bit = match codec {
        VideoCodec::H264 => false,
        VideoCodec::Hevc10 => enc != "libkvazaar",
        _ => true,
    };
    let q = t.quality.min(51);
    let gop = ((t.rate.as_f64() * 2.0).round() as u32).max(1).to_string();
    let s = |xs: &[&str]| xs.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
    let qs = q.to_string();

    let mut args: Vec<String>;
    let mut pix_fmt = if ten_bit { "p010le" } else { "nv12" }.to_owned();
    let mut vaapi_device = None;
    let mut hdr = if color.is_hdr() {
        HdrEmbedding::TagsOnly
    } else {
        HdrEmbedding::NotApplicable
    };

    match enc {
        e if e.ends_with("_nvenc") => {
            args = s(&[
                "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", &qs, "-b:v", "0",
            ]);
            args.extend(s(&[
                "-spatial_aq",
                "1",
                "-temporal_aq",
                "1",
                "-g",
                &gop,
                "-bf",
                "3",
            ]));
            if e == "hevc_nvenc" && ten_bit {
                args.extend(s(&["-profile:v", "main10"]));
            }
        }
        e if e.ends_with("_amf") => {
            args = s(&[
                "-quality", "quality", "-rc", "cqp", "-qp_i", &qs, "-qp_p", &qs, "-g", &gop,
            ]);
            if e == "hevc_amf" && ten_bit {
                args.extend(s(&["-profile:v", "main10"]));
            }
        }
        e if e.ends_with("_qsv") => {
            args = s(&["-preset", "slower", "-global_quality", &qs, "-g", &gop]);
            if e == "hevc_qsv" && ten_bit {
                args.extend(s(&["-profile:v", "main10"]));
            }
        }
        e if e.ends_with("_vaapi") => {
            args = s(&["-rc_mode", "CQP", "-qp", &qs, "-g", &gop]);
            if e == "hevc_vaapi" && ten_bit {
                args.extend(s(&["-profile:v", "main10"]));
            }
            pix_fmt = if ten_bit { "p010le" } else { "nv12" }.into();
            vaapi_device = env
                .vaapi_device
                .clone()
                .or_else(|| Some("/dev/dri/renderD128".into()));
        }
        "prores_videotoolbox" => {
            let (profile, pf) = match codec {
                VideoCodec::ProRes4444 => ("4444", "ayuv64le"),
                _ => ("hq", "p210le"),
            };
            args = s(&["-profile:v", profile]);
            pix_fmt = pf.into();
        }
        e if e.ends_with("_videotoolbox") => {
            // VideoToolbox constant quality is 1–100 (higher is better); ~65 ≈ CRF 20.
            let vq = (100.0 - f32::from(q) * 1.75)
                .clamp(1.0, 100.0)
                .round()
                .to_string();
            args = s(&["-q:v", &vq, "-allow_sw", "0", "-realtime", "0", "-g", &gop]);
            if e == "hevc_videotoolbox" && ten_bit {
                args.extend(s(&["-profile:v", "main10"]));
            }
        }
        "libsvtav1" => {
            let crf = (u32::from(q) + 8).clamp(1, 63).to_string();
            let mut params = String::from("tune=0");
            if color.is_hdr() {
                params.push_str(":enable-hdr=1");
                if let Some(md) = color.mastering {
                    let c = |v: u32| format!("{:.4}", f64::from(v) / 50_000.0);
                    params.push_str(&format!(
                        ":mastering-display=G({},{})B({},{})R({},{})WP({},{})L({:.4},{:.4})",
                        c(md.green[0]),
                        c(md.green[1]),
                        c(md.blue[0]),
                        c(md.blue[1]),
                        c(md.red[0]),
                        c(md.red[1]),
                        c(md.white[0]),
                        c(md.white[1]),
                        f64::from(md.max_luminance) / 10_000.0,
                        f64::from(md.min_luminance) / 10_000.0,
                    ));
                }
                if let Some(cll) = color.content_light {
                    params.push_str(&format!(":content-light={},{}", cll.max_cll, cll.max_fall));
                }
                hdr = HdrEmbedding::Bitstream;
            }
            args = s(&[
                "-preset",
                "6",
                "-crf",
                &crf,
                "-g",
                &gop,
                "-svtav1-params",
                &params,
            ]);
            pix_fmt = "yuv420p10le".into();
        }
        "libkvazaar" => {
            args = s(&["-kvazaar-params", &format!("preset=medium,qp={q}")]);
            pix_fmt = "yuv420p".into();
        }
        "libopenh264" => {
            // No constant-quality mode; target ≈0.12 bits/pixel/frame, scaled by quality.
            let bpp = 0.12 * (1.0 + (20.0 - f64::from(q)) / 20.0).max(0.25);
            let bits = f64::from(t.width) * f64::from(t.height) * t.rate.as_f64() * bpp;
            args = s(&[
                "-rc_mode",
                "quality",
                "-b:v",
                &format!("{}", bits.round() as u64),
                "-g",
                &gop,
            ]);
            pix_fmt = "yuv420p".into();
        }
        _ => args = Vec::new(),
    }

    EncoderSelection {
        encoder: enc.to_owned(),
        args,
        pix_fmt,
        vaapi_device,
        hdr,
        out_bit_depth: if ten_bit { 10 } else { 8 },
        fell_back,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_types::media::*;

    fn color(hdr: bool) -> ColorProfile {
        ColorProfile {
            primaries: if hdr {
                ColorPrimaries::Bt2020
            } else {
                ColorPrimaries::Bt709
            },
            transfer: if hdr {
                TransferCharacteristic::Pq
            } else {
                TransferCharacteristic::Bt709
            },
            matrix: if hdr {
                MatrixCoefficients::Bt2020Nc
            } else {
                MatrixCoefficients::Bt709
            },
            range: ColorRange::Limited,
            bit_depth: if hdr { 10 } else { 8 },
            mastering: hdr.then_some(MasteringDisplay {
                red: [34000, 16000],
                green: [13250, 34500],
                blue: [7500, 3000],
                white: [15635, 16450],
                max_luminance: 10_000_000,
                min_luminance: 50,
            }),
            content_light: hdr.then_some(ContentLightLevel {
                max_cll: 1000,
                max_fall: 400,
            }),
            inferred: false,
        }
    }
    fn env(os: Os, vendor: Option<Vendor>, usable: &[&str]) -> EncoderEnv {
        EncoderEnv {
            os,
            vendor,
            usable: usable.iter().map(|s| (*s).to_owned()).collect(),
            vaapi_device: Some("/dev/dri/renderD129".into()),
        }
    }
    const T: EncodeTarget = EncodeTarget {
        quality: 20,
        width: 3840,
        height: 2160,
        rate: Rational::new(24, 1),
    };

    #[test]
    fn nvidia_hevc_main10() {
        let e = env(
            Os::Windows,
            Some(Vendor::Nvidia),
            &["hevc_nvenc", "libsvtav1"],
        );
        let s = select(
            VideoCodec::Hevc10,
            EncoderPreference::Auto,
            &color(true),
            T,
            &e,
        )
        .expect("ok");
        assert_eq!(s.encoder, "hevc_nvenc");
        assert_eq!(s.pix_fmt, "p010le");
        assert!(s.args.windows(2).any(|w| w == ["-profile:v", "main10"]));
        assert_eq!(s.hdr, HdrEmbedding::TagsOnly);
    }

    #[test]
    fn av1_falls_back_to_svt_with_hdr_in_bitstream() {
        // RTX 30-series: av1_nvenc compiled in but not usable.
        let e = env(
            Os::Linux,
            Some(Vendor::Nvidia),
            &["hevc_nvenc", "libsvtav1"],
        );
        let s = select(
            VideoCodec::Av1,
            EncoderPreference::Auto,
            &color(true),
            T,
            &e,
        )
        .expect("ok");
        assert_eq!(s.encoder, "libsvtav1");
        assert!(s.fell_back);
        assert_eq!(s.hdr, HdrEmbedding::Bitstream);
        let p = &s.args[s
            .args
            .iter()
            .position(|a| a == "-svtav1-params")
            .expect("p")
            + 1];
        assert!(p.contains("mastering-display=G(0.2650,0.6900)B(0.1500,0.0600)R(0.6800,0.3200)WP(0.3127,0.3290)L(1000.0000,0.0050)"), "{p}");
        assert!(p.contains("content-light=1000,400"));
    }

    #[test]
    fn hevc10_hdr_without_hw_is_a_clear_error() {
        let e = env(Os::Linux, Some(Vendor::Amd), &["libsvtav1", "libkvazaar"]);
        let err = select(
            VideoCodec::Hevc10,
            EncoderPreference::Auto,
            &color(true),
            T,
            &e,
        )
        .expect_err("no hevc");
        assert!(err.to_string().contains("Choose AV1"), "{err}");
        // SDR 8-bit can fall back to kvazaar.
        let s = select(
            VideoCodec::Hevc10,
            EncoderPreference::Auto,
            &color(false),
            T,
            &e,
        )
        .expect("sdr");
        assert_eq!((s.encoder.as_str(), s.out_bit_depth), ("libkvazaar", 8));
    }

    #[test]
    fn vaapi_uses_render_node_and_prores_is_mac_only() {
        let e = env(Os::Linux, Some(Vendor::Amd), &["hevc_vaapi"]);
        let s = select(
            VideoCodec::Hevc10,
            EncoderPreference::Hardware,
            &color(false),
            T,
            &e,
        )
        .expect("ok");
        assert_eq!(s.vaapi_device.as_deref(), Some("/dev/dri/renderD129"));
        assert!(
            select(
                VideoCodec::ProRes422Hq,
                EncoderPreference::Auto,
                &color(false),
                T,
                &e
            )
            .is_err()
        );
        let mac = env(Os::MacOs, Some(Vendor::Apple), &["prores_videotoolbox"]);
        let p = select(
            VideoCodec::ProRes422Hq,
            EncoderPreference::Auto,
            &color(false),
            T,
            &mac,
        )
        .expect("mac");
        assert_eq!(p.pix_fmt, "p210le");
    }

    #[test]
    fn h264_refuses_hdr() {
        let e = env(Os::Windows, Some(Vendor::Nvidia), &["h264_nvenc"]);
        assert!(
            select(
                VideoCodec::H264,
                EncoderPreference::Auto,
                &color(true),
                T,
                &e
            )
            .is_err()
        );
    }
}
