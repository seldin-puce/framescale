//! ffprobe → [`MediaInfo`].
//!
//! Two probes:
//! 1. Container and streams (`-show_format -show_streams -show_chapters`).
//! 2. For PQ/HLG sources only: the first frame's side data, because HEVC
//!    carries mastering-display and content-light metadata in SEI messages,
//!    not in the stream headers.

use std::path::Path;
use std::process::Command;

use fs_types::media::{
    AudioStream, ContentLightLevel, FieldOrder, MasteringDisplay, MediaInfo, Rational,
    SubtitleStream, VideoStream,
};
use serde_json::Value;

use crate::argv::Argv;
use crate::color::{self, RawColorTags};

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("could not run ffprobe: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("ffprobe failed: {0}")]
    Failed(String),
    #[error("ffprobe output is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no video stream found")]
    NoVideo,
}

pub fn streams_argv(input: &Path) -> Argv {
    let mut a = Argv::new();
    a.args(["-v", "error", "-print_format", "json"]);
    a.args(["-show_format", "-show_streams", "-show_chapters"]);
    a.arg(input);
    a
}

pub fn frame_side_data_argv(input: &Path, stream_index: u32) -> Argv {
    let mut a = Argv::new();
    a.args(["-v", "error", "-print_format", "json"]);
    a.args([
        "-select_streams",
        &stream_index.to_string(),
        "-read_intervals",
        "%+#1",
    ]);
    a.args(["-show_frames", "-show_entries", "frame=side_data_list"]);
    a.arg(input);
    a
}

/// Run both probes (the second only when needed).
pub fn probe(ffprobe: &Path, input: &Path) -> Result<MediaInfo, ProbeError> {
    let run = |a: Argv| -> Result<String, ProbeError> {
        let out = Command::new(ffprobe).args(a.as_slice()).output()?;
        if !out.status.success() {
            return Err(ProbeError::Failed(
                String::from_utf8_lossy(&out.stderr).trim().to_owned(),
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let mut info = parse_streams(&run(streams_argv(input))?, &input.to_string_lossy())?;
    if info.video.color.is_hdr() {
        let json = run(frame_side_data_argv(input, info.video.index))?;
        apply_frame_side_data(&mut info.video, &json)?;
    }
    Ok(info)
}

fn s<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k)?.as_str()
}
fn num<T: std::str::FromStr>(v: &Value, k: &str) -> Option<T> {
    match v.get(k)? {
        Value::String(s) => s.parse().ok(),
        Value::Number(n) => n.to_string().parse().ok(),
        _ => None,
    }
}
fn language(v: &Value) -> Option<String> {
    v.get("tags")?
        .get("language")?
        .as_str()
        .filter(|l| *l != "und")
        .map(str::to_owned)
}
fn side_data(v: &Value) -> impl Iterator<Item = &Value> {
    v.get("side_data_list")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

/// Parse probe 1.
pub fn parse_streams(json: &str, path: &str) -> Result<MediaInfo, ProbeError> {
    let root: Value = serde_json::from_str(json)?;
    let streams = root
        .get("streams")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let format = root.get("format").cloned().unwrap_or(Value::Null);

    // First real video stream: skip cover art / attached pictures.
    let v = streams
        .iter()
        .find(|st| {
            s(st, "codec_type") == Some("video")
                && st
                    .get("disposition")
                    .and_then(|d| d.get("attached_pic"))
                    .and_then(Value::as_i64)
                    != Some(1)
        })
        .ok_or(ProbeError::NoVideo)?;

    let video = parse_video(v);
    let audio = streams
        .iter()
        .filter(|st| s(st, "codec_type") == Some("audio"))
        .map(|st| AudioStream {
            index: num(st, "index").unwrap_or(0),
            codec: s(st, "codec_name").unwrap_or("unknown").to_owned(),
            channels: num(st, "channels").unwrap_or(0),
            sample_rate: num(st, "sample_rate").unwrap_or(0),
            start_time: num(st, "start_time").unwrap_or(0.0),
            language: language(st),
        })
        .collect();
    let subtitles = streams
        .iter()
        .filter(|st| s(st, "codec_type") == Some("subtitle"))
        .map(|st| {
            let codec = s(st, "codec_name").unwrap_or("unknown").to_owned();
            SubtitleStream {
                index: num(st, "index").unwrap_or(0),
                bitmap: matches!(
                    codec.as_str(),
                    "hdmv_pgs_subtitle" | "dvd_subtitle" | "dvb_subtitle" | "xsub"
                ),
                codec,
                language: language(st),
            }
        })
        .collect();

    Ok(MediaInfo {
        path: path.to_owned(),
        format: s(&format, "format_name").unwrap_or("unknown").to_owned(),
        duration_s: num(&format, "duration").unwrap_or(0.0),
        size_bytes: num(&format, "size").unwrap_or(0),
        bit_rate: num(&format, "bit_rate"),
        start_time: num(&format, "start_time").unwrap_or(0.0),
        video,
        audio,
        subtitles,
        chapters: root
            .get("chapters")
            .and_then(Value::as_array)
            .map_or(0, |c| c.len() as u32),
    })
}

fn parse_video(v: &Value) -> VideoStream {
    let width = num(v, "width").unwrap_or(0);
    let height = num(v, "height").unwrap_or(0);
    let pix_fmt = s(v, "pix_fmt").unwrap_or("yuv420p").to_owned();
    let bit_depth = num::<u8>(v, "bits_per_raw_sample")
        .filter(|d| *d > 8)
        .unwrap_or_else(|| color::bit_depth_of(&pix_fmt));
    let tags = RawColorTags {
        primaries: s(v, "color_primaries"),
        transfer: s(v, "color_transfer"),
        matrix: s(v, "color_space"),
        range: s(v, "color_range"),
    };
    let color = color::resolve(
        &tags,
        width,
        height,
        bit_depth,
        color::is_rgb_pix_fmt(&pix_fmt),
    );

    let rotation = side_data(v)
        .find(|d| s(d, "side_data_type") == Some("Display Matrix"))
        .and_then(|d| num::<f64>(d, "rotation"))
        .or_else(|| v.get("tags").and_then(|t| num::<f64>(t, "rotate")))
        .map_or(0, |r| (r.round() as i32).rem_euclid(360));

    let dolby_vision = side_data(v)
        .any(|d| s(d, "side_data_type").is_some_and(|t| t.starts_with("DOVI configuration")));

    let field_order = match s(v, "field_order") {
        Some("progressive") => FieldOrder::Progressive,
        Some("tt" | "tb") => FieldOrder::TopFirst,
        Some("bb" | "bt") => FieldOrder::BottomFirst,
        _ => FieldOrder::Unknown,
    };

    VideoStream {
        index: num(v, "index").unwrap_or(0),
        codec: s(v, "codec_name").unwrap_or("unknown").to_owned(),
        profile: s(v, "profile").map(str::to_owned),
        width,
        height,
        sample_aspect: s(v, "sample_aspect_ratio")
            .and_then(|r| Rational::parse(&r.replace(':', "/")))
            .unwrap_or(Rational::new(1, 1)),
        r_frame_rate: s(v, "r_frame_rate")
            .and_then(Rational::parse)
            .unwrap_or(Rational::new(0, 1)),
        avg_frame_rate: s(v, "avg_frame_rate")
            .and_then(Rational::parse)
            .unwrap_or(Rational::new(0, 1)),
        frame_count: num(v, "nb_frames"),
        start_time: num(v, "start_time").unwrap_or(0.0),
        pix_fmt,
        color,
        rotation,
        field_order,
        dolby_vision,
        hdr10_plus: false,
    }
}

/// `"34000/50000"` → value scaled by `unit` (e.g. 50 000 for chromaticity).
fn scaled(d: &Value, k: &str, unit: f64) -> Option<u32> {
    let r = s(d, k)?;
    let (n, den) = r.split_once('/').unwrap_or((r, "1"));
    let x = n.parse::<f64>().ok()? / den.parse::<f64>().ok()?;
    Some((x * unit).round() as u32)
}

/// Parse probe 2: mastering display, content light, HDR10+, DV RPU.
pub fn apply_frame_side_data(video: &mut VideoStream, json: &str) -> Result<(), ProbeError> {
    let root: Value = serde_json::from_str(json)?;
    let Some(frame) = root
        .get("frames")
        .and_then(Value::as_array)
        .and_then(|f| f.first())
    else {
        return Ok(());
    };
    for d in side_data(frame) {
        match s(d, "side_data_type").unwrap_or_default() {
            "Mastering display metadata" => {
                let xy = |c: &str| -> Option<[u32; 2]> {
                    Some([
                        scaled(d, &format!("{c}_x"), 50_000.0)?,
                        scaled(d, &format!("{c}_y"), 50_000.0)?,
                    ])
                };
                if let (Some(red), Some(green), Some(blue), Some(white), Some(max), Some(min)) = (
                    xy("red"),
                    xy("green"),
                    xy("blue"),
                    xy("white_point"),
                    scaled(d, "max_luminance", 10_000.0),
                    scaled(d, "min_luminance", 10_000.0),
                ) {
                    video.color.mastering = Some(MasteringDisplay {
                        red,
                        green,
                        blue,
                        white,
                        max_luminance: max,
                        min_luminance: min,
                    });
                }
            }
            "Content light level metadata" => {
                video.color.content_light = Some(ContentLightLevel {
                    max_cll: num(d, "max_content").unwrap_or(0),
                    max_fall: num(d, "max_average").unwrap_or(0),
                });
            }
            t if t.starts_with("HDR Dynamic Metadata SMPTE2094-40") => video.hdr10_plus = true,
            t if t.starts_with("Dolby Vision") => video.dolby_vision = true,
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_types::media::{MatrixCoefficients, TransferCharacteristic};

    const HDR_STREAMS: &str = r#"{
      "streams": [
        { "index": 0, "codec_type": "video", "codec_name": "mjpeg", "width": 600, "height": 600,
          "pix_fmt": "yuvj420p", "disposition": { "attached_pic": 1 } },
        { "index": 1, "codec_type": "video", "codec_name": "hevc", "profile": "Main 10",
          "width": 3840, "height": 2160, "pix_fmt": "yuv420p10le", "sample_aspect_ratio": "1:1",
          "r_frame_rate": "24000/1001", "avg_frame_rate": "24000/1001", "start_time": "0.042000",
          "color_range": "tv", "color_space": "bt2020nc", "color_transfer": "smpte2084",
          "color_primaries": "bt2020", "field_order": "progressive", "nb_frames": "1000",
          "side_data_list": [
            { "side_data_type": "Display Matrix", "rotation": -90 },
            { "side_data_type": "DOVI configuration record", "dv_profile": 8 }
          ] },
        { "index": 2, "codec_type": "audio", "codec_name": "eac3", "channels": 6,
          "sample_rate": "48000", "start_time": "0.000000", "tags": { "language": "eng" } },
        { "index": 3, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": { "language": "und" } }
      ],
      "chapters": [ {}, {} ],
      "format": { "format_name": "matroska,webm", "duration": "41.708000", "size": "123456789",
                  "bit_rate": "23680000", "start_time": "0.000000" }
    }"#;

    const HDR_FRAME: &str = r#"{ "frames": [ { "side_data_list": [
        { "side_data_type": "Mastering display metadata",
          "red_x": "34000/50000", "red_y": "16000/50000", "green_x": "13250/50000", "green_y": "34500/50000",
          "blue_x": "7500/50000", "blue_y": "3000/50000", "white_point_x": "15635/50000", "white_point_y": "16450/50000",
          "min_luminance": "50/10000", "max_luminance": "10000000/10000" },
        { "side_data_type": "Content light level metadata", "max_content": 1000, "max_average": 400 },
        { "side_data_type": "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)" }
    ] } ] }"#;

    #[test]
    fn parses_hdr_source() {
        let mut info = parse_streams(HDR_STREAMS, "/x.mkv").expect("parse");
        let v = &info.video;
        assert_eq!(v.index, 1, "cover art skipped");
        assert_eq!(v.color.transfer, TransferCharacteristic::Pq);
        assert_eq!(v.color.matrix, MatrixCoefficients::Bt2020Nc);
        assert_eq!(v.color.bit_depth, 10);
        assert_eq!(v.rotation, 270);
        assert!(v.dolby_vision);
        assert_eq!(v.frame_count, Some(1000));
        assert_eq!(info.audio[0].language.as_deref(), Some("eng"));
        assert!(info.subtitles[0].bitmap && info.subtitles[0].language.is_none());
        assert_eq!(info.chapters, 2);

        apply_frame_side_data(&mut info.video, HDR_FRAME).expect("frame");
        let md = info.video.color.mastering.expect("mdcv");
        assert_eq!(md.red, [34000, 16000]);
        assert_eq!(md.white, [15635, 16450]);
        assert_eq!((md.max_luminance, md.min_luminance), (10_000_000, 50));
        assert_eq!(
            info.video.color.content_light.map(|c| c.max_cll),
            Some(1000)
        );
        assert!(info.video.hdr10_plus);
    }

    #[test]
    fn no_video_is_an_error() {
        let j = r#"{"streams":[{"index":0,"codec_type":"audio"}],"format":{}}"#;
        assert!(matches!(
            parse_streams(j, "a.flac"),
            Err(ProbeError::NoVideo)
        ));
    }
}
