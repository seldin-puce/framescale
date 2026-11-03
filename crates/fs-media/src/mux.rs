//! Final mux: AI video + original audio/subtitles/chapters/metadata.
//!
//! Audio is never decoded (`-c:a copy`) when the container accepts the codec,
//! so samples and timestamps stay bit-exact. The only timing correction is the
//! source's video-vs-container start offset, applied with `-itsoffset`.

use std::path::Path;

use fs_types::WarningCode;
use fs_types::media::MediaInfo;
use fs_types::pipeline::{Container, VideoCodec};

use crate::argv::Argv;

#[derive(Debug, Clone, PartialEq)]
pub struct MuxPlan {
    pub argv: Argv,
    pub warnings: Vec<(WarningCode, String)>,
}

fn audio_copy_ok(container: Container, codec: &str) -> bool {
    match container {
        Container::Mkv => true,
        Container::Mp4 => matches!(
            codec,
            "aac" | "mp3" | "ac3" | "eac3" | "opus" | "flac" | "alac"
        ),
        Container::Mov => {
            matches!(codec, "aac" | "alac" | "mp3" | "ac3" | "eac3") || codec.starts_with("pcm_")
        }
    }
}

fn is_text_sub(codec: &str) -> bool {
    matches!(
        codec,
        "subrip" | "srt" | "ass" | "ssa" | "mov_text" | "webvtt" | "text"
    )
}

/// Offset of the first video frame relative to the container start. Our
/// rendered video starts at 0, so it has to be shifted by this to keep sync.
pub fn video_offset(source: &MediaInfo) -> f64 {
    (source.video.start_time - source.start_time).max(0.0)
}

pub fn plan(
    video: &Path,
    source: &MediaInfo,
    container: Container,
    codec: VideoCodec,
    out: &Path,
) -> MuxPlan {
    let mut warnings = Vec::new();
    let mut a = Argv::new();
    a.args([
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-y", // the output is our own `.part`; a crashed attempt may have left one
        "-loglevel",
        "warning",
        "-progress",
        "pipe:1",
    ]);
    let offset = video_offset(source);
    if offset > 0.0005 {
        a.args(["-itsoffset", &format!("{offset:.6}")]);
    }
    a.arg("-i").arg(video);
    a.arg("-i").arg(&source.path);
    a.args(["-map", "0:v:0", "-c:v", "copy"]);

    for (n, au) in source.audio.iter().enumerate() {
        a.args(["-map", &format!("1:{}", au.index)]);
        if audio_copy_ok(container, &au.codec) {
            a.args([format!("-c:a:{n}"), "copy".into()]);
        } else {
            a.args([
                format!("-c:a:{n}"),
                "aac".into(),
                format!("-b:a:{n}"),
                "320k".into(),
            ]);
            warnings.push((
                WarningCode::AudioTranscoded,
                format!(
                    "audio track {n} ({}) re-encoded to AAC for {container:?}",
                    au.codec
                ),
            ));
        }
    }

    let mut sn = 0usize;
    for sub in &source.subtitles {
        let action = match container {
            Container::Mkv => Some("copy"),
            _ if !sub.bitmap && is_text_sub(&sub.codec) => Some("mov_text"),
            _ => None,
        };
        match action {
            Some(c) => {
                a.args(["-map", &format!("1:{}", sub.index)]);
                a.args([format!("-c:s:{sn}"), c.to_owned()]);
                sn += 1;
            }
            None => warnings.push((
                WarningCode::SubtitlesDropped,
                format!(
                    "{} subtitle track can't be stored in {container:?}; use MKV to keep it",
                    sub.codec
                ),
            )),
        }
    }

    a.args(["-map_chapters", "1", "-map_metadata", "1"]);
    // Rotation was applied at decode time; don't carry the source's display matrix.
    a.args(["-metadata:s:v:0", "rotate=0"]);
    if matches!(container, Container::Mp4 | Container::Mov) {
        a.args(["-movflags", "+faststart"]);
        if codec == VideoCodec::Hevc10 {
            // Apple players only accept the `hvc1` sample entry.
            a.args(["-tag:v", "hvc1"]);
        }
    }
    a.args(["-f", container.muxer()]).arg(out);
    MuxPlan { argv: a, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> MediaInfo {
        let json = r#"{"streams":[
          {"index":0,"codec_type":"video","codec_name":"h264","width":1920,"height":1080,"pix_fmt":"yuv420p","start_time":"0.100000"},
          {"index":1,"codec_type":"audio","codec_name":"pcm_s24le","channels":2,"sample_rate":"48000","start_time":"0.000000"},
          {"index":2,"codec_type":"audio","codec_name":"aac","channels":2,"sample_rate":"48000","start_time":"0.000000"},
          {"index":3,"codec_type":"subtitle","codec_name":"subrip"},
          {"index":4,"codec_type":"subtitle","codec_name":"hdmv_pgs_subtitle"}],
          "format":{"format_name":"matroska","start_time":"0.000000","duration":"10"}}"#;
        crate::probe::parse_streams(json, "/in.mkv").expect("parse")
    }

    #[test]
    fn mp4_mux_maps_transcodes_and_drops_correctly() {
        let p = plan(
            Path::new("/v.mkv"),
            &source(),
            Container::Mp4,
            VideoCodec::Hevc10,
            Path::new("/o.mp4.part"),
        );
        let s = p.argv.to_string();
        assert_eq!(p.argv.value_of("-itsoffset").as_deref(), Some("0.100000"));
        assert!(s.contains("-map 1:1 -c:a:0 aac -b:a:0 320k"), "{s}");
        assert!(s.contains("-map 1:2 -c:a:1 copy"), "{s}");
        assert!(s.contains("-map 1:3 -c:s:0 mov_text"), "{s}");
        assert!(!s.contains("1:4"), "PGS can't go into MP4: {s}");
        assert_eq!(p.argv.value_of("-tag:v").as_deref(), Some("hvc1"));
        let codes: Vec<_> = p.warnings.iter().map(|w| w.0).collect();
        assert_eq!(
            codes,
            vec![WarningCode::AudioTranscoded, WarningCode::SubtitlesDropped]
        );
    }

    #[test]
    fn mkv_copies_everything() {
        let p = plan(
            Path::new("/v.mkv"),
            &source(),
            Container::Mkv,
            VideoCodec::Av1,
            Path::new("/o.mkv"),
        );
        assert!(p.warnings.is_empty());
        assert!(p.argv.to_string().contains("-map 1:4 -c:s:1 copy"));
        assert!(!p.argv.contains("-movflags"));
    }
}
