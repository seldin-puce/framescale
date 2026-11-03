//! Compile (probe + analysis + user request + device) into a [`RenderPlan`].
//! This is the single place where pipeline policy is decided. The worker
//! only executes the plan.

use fs_types::media::{AnalysisReport, ColorProfile, MediaInfo};
use fs_types::pipeline::{JobSpec, PipelineError, StageConfig};
use fs_types::worker::{DecodePlan, EncodePlan, EnginePlan, FrameFormat, HwDecode, RenderPlan};
use fs_types::{GpuDevice, Rational, Vendor, WarningCode};

use crate::encoders::{self, EncodeTarget, EncoderEnv, EncoderError, Os};
use crate::rate;

/// Decision D4: 4-second closed-GOP segments.
pub const SEGMENT_SECONDS: f64 = 4.0;

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("invalid pipeline: {0:?}")]
    Pipeline(Vec<PipelineError>),
    #[error(transparent)]
    Encoder(#[from] EncoderError),
    #[error("source has invalid dimensions {0}×{1}")]
    BadDimensions(u32, u32),
    #[error("{0}")]
    Output(String),
}

#[derive(Debug, Clone)]
pub struct PlanInputs<'a> {
    pub info: &'a MediaInfo,
    pub analysis: &'a AnalysisReport,
    pub spec: &'a JobSpec,
    pub device: Option<&'a GpuDevice>,
    pub encoders: &'a EncoderEnv,
    pub ffmpeg: &'a str,
    pub segment_dir: &'a str,
    pub models_dir: &'a str,
}

#[derive(Debug, Clone)]
pub struct Planned {
    pub plan: RenderPlan,
    pub warnings: Vec<(WarningCode, String)>,
    pub encoder_fell_back: bool,
}

/// Hardware decode API for the selected device.
pub fn hw_decode_for(device: Option<&GpuDevice>, os: Os) -> HwDecode {
    match (os, device.map(|d| d.vendor)) {
        (Os::MacOs, _) => HwDecode::VideoToolbox,
        (_, Some(Vendor::Nvidia)) => HwDecode::Cuda,
        (Os::Windows, Some(_)) => HwDecode::D3d11va,
        (Os::Linux, Some(Vendor::Amd | Vendor::Intel)) => {
            match device.and_then(|d| d.drm_render_minor) {
                Some(minor) => HwDecode::Vaapi {
                    device: format!("/dev/dri/renderD{minor}"),
                },
                None => HwDecode::None,
            }
        }
        _ => HwDecode::None,
    }
}

pub fn compile(i: &PlanInputs<'_>) -> Result<Planned, PlanError> {
    i.spec.pipeline.validate().map_err(PlanError::Pipeline)?;
    if let Some(why) = i.spec.output.problem() {
        return Err(PlanError::Output(why));
    }
    let v = &i.info.video;
    if v.width == 0 || v.height == 0 {
        return Err(PlanError::BadDimensions(v.width, v.height));
    }
    let mut warnings = Vec::new();
    let pipe = &i.spec.pipeline;

    // FFmpeg autorotates on decode, so the engine sees display orientation.
    let (in_w, in_h) = if v.rotation % 180 == 90 {
        (v.height, v.width)
    } else {
        (v.width, v.height)
    };

    let deinterlace = pipe.deinterlace();
    let base_rate = i.analysis.conform_rate;
    // bwdif send_field emits one frame per field → twice the frame rate.
    let field_mul = if deinterlace == Some(true) { 2 } else { 1 };
    let conform = base_rate.times(field_mul);
    let out_rate = conform.times(i64::from(pipe.interpolation_factor()));
    let scale = pipe.scale_factor();
    let (out_w, out_h) = (in_w * scale, in_h * scale);

    if i.analysis.is_vfr {
        warnings.push((
            WarningCode::VfrConformed,
            format!(
                "variable frame rate source conformed to {:.3} fps",
                base_rate.as_f64()
            ),
        ));
    }
    if v.dolby_vision || v.hdr10_plus {
        warnings.push((
            WarningCode::DynamicHdrStripped,
            "Dolby Vision / HDR10+ dynamic metadata can't survive resizing; output is HDR10".into(),
        ));
    }

    let frame_format = if v.color.needs_high_precision() {
        FrameFormat::Rgb48
    } else {
        FrameFormat::Rgb24
    };
    let os = i.encoders.os;
    let hw = hw_decode_for(i.device, os);

    let target = EncodeTarget {
        quality: i.spec.output.quality,
        width: out_w,
        height: out_h,
        rate: out_rate,
    };
    let sel = encoders::select(
        i.spec.output.codec,
        i.spec.output.encoder,
        &v.color,
        target,
        i.encoders,
    )?;
    if sel.fell_back {
        warnings.push((
            WarningCode::EncoderFallback,
            format!("no usable hardware encoder; using {}", sel.encoder),
        ));
    }
    if sel.hdr == fs_types::worker::HdrEmbedding::TagsOnly && v.color.mastering.is_some() {
        warnings.push((
            WarningCode::HdrStaticMetadataTagsOnly,
            format!(
                "{} will tag the output PQ/HLG + BT.2020, but mastering-display and \
                 content-light values are not embedded; choose AV1 to keep them",
                sel.encoder
            ),
        ));
    }
    let out_color = ColorProfile {
        bit_depth: sel.out_bit_depth,
        ..v.color.clone()
    };

    let scene_cuts = if field_mul == 2 {
        i.analysis.scene_cuts.iter().map(|c| c * 2).collect()
    } else {
        i.analysis.scene_cuts.clone()
    };
    let total_frames = i.analysis.total_frames * field_mul as u64;
    let models_dir = i.models_dir.to_owned();

    let plan = RenderPlan {
        ffmpeg: i.ffmpeg.to_owned(),
        segment_dir: i.segment_dir.to_owned(),
        decode: DecodePlan {
            input: i.info.path.clone(),
            stream_index: v.index,
            hw,
            conform_rate: conform,
            deinterlace,
            width: in_w,
            height: in_h,
            frame_format,
            color: v.color.clone(),
            seek_base_s: crate::mux::video_offset(i.info),
        },
        encode: EncodePlan {
            encoder: sel.encoder,
            args: sel.args,
            pix_fmt: sel.pix_fmt,
            vaapi_device: sel.vaapi_device,
            width: out_w,
            height: out_h,
            rate: out_rate,
            color: out_color,
            hdr: sel.hdr,
            sample_aspect: v.sample_aspect,
        },
        engine: EnginePlan {
            stages: pipe
                .stages
                .iter()
                .filter(|s| !matches!(s, StageConfig::Deinterlace { .. }))
                .cloned()
                .collect(),
            scene_cuts,
            in_width: in_w,
            in_height: in_h,
            out_width: out_w,
            out_height: out_h,
            frame_format,
            device: i.device.map(|d| d.uuid),
            tile: None,
            models_dir,
        },
        total_frames,
        segment_frames: rate::segment_frames(conform, SEGMENT_SECONDS),
    };
    Ok(Planned {
        plan,
        warnings,
        encoder_fell_back: sel.fell_back,
    })
}

/// Rough output size for the storage preflight: bits-per-pixel heuristics per
/// codec family, intentionally pessimistic (×1.5).
pub fn estimate_output_bytes(plan: &RenderPlan) -> u64 {
    let e = &plan.encode;
    let bpp = match e.encoder.as_str() {
        x if x.starts_with("prores") => 1.6,
        x if x.contains("av1") => 0.05,
        x if x.contains("hevc") || x == "libkvazaar" => 0.07,
        _ => 0.12,
    };
    let interp = e.rate.as_f64() / plan.decode.conform_rate.as_f64();
    let frames_out = plan.total_frames as f64 * interp;
    let bytes = f64::from(e.width) * f64::from(e.height) * frames_out * bpp / 8.0;
    (bytes * 1.5) as u64
}

/// Conformed frame count estimate from probe data (used before analysis completes).
pub fn estimate_frames(info: &MediaInfo, conform: Rational) -> u64 {
    (info.duration_s * conform.as_f64()).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_types::media::*;
    use fs_types::pipeline::*;

    fn info() -> MediaInfo {
        let json = r#"{"streams":[{"index":0,"codec_type":"video","codec_name":"h264","width":1920,
          "height":1080,"pix_fmt":"yuv420p","r_frame_rate":"30/1","avg_frame_rate":"30/1",
          "start_time":"0.0","side_data_list":[{"side_data_type":"Display Matrix","rotation":90}]}],
          "format":{"format_name":"mov","duration":"10","start_time":"0"}}"#;
        crate::probe::parse_streams(json, "/in.mov").expect("parse")
    }

    #[test]
    fn plan_upscale_interpolate_with_deinterlace() {
        let m = ModelRef {
            id: "realesrgan-x4plus".into(),
            path: None,
        };
        let spec = JobSpec {
            input: "/in.mov".into(),
            pipeline: PipelineSpec {
                stages: vec![
                    StageConfig::Deinterlace { send_field: true },
                    StageConfig::Upscale {
                        model: m.clone(),
                        scale: 2,
                        denoise: None,
                    },
                    StageConfig::Interpolate {
                        model: m,
                        factor: 2,
                        scene_threshold: 10.0,
                    },
                ],
            },
            output: OutputSpec {
                path: "/out.mp4".into(),
                codec: VideoCodec::Av1,
                container: Container::Mp4,
                quality: 20,
                encoder: EncoderPreference::Auto,
            },
            device: None,
        };
        let analysis = AnalysisReport {
            conform_rate: Rational::new(30, 1),
            vfr_ratio: 0.0,
            is_vfr: false,
            field_order: FieldOrder::TopFirst,
            scene_cuts: vec![100],
            total_frames: 300,
        };
        let env = EncoderEnv {
            os: Os::Linux,
            vendor: None,
            usable: ["libsvtav1".to_owned()].into(),
            vaapi_device: None,
        };
        let info = info();
        let p = compile(&PlanInputs {
            info: &info,
            analysis: &analysis,
            spec: &spec,
            device: None,
            encoders: &env,
            ffmpeg: "ffmpeg",
            segment_dir: "/tmp/seg",
            models_dir: "/models",
        })
        .expect("plan")
        .plan;
        // Rotation 90° swaps dimensions; ×2 upscale.
        assert_eq!((p.decode.width, p.decode.height), (1080, 1920));
        assert_eq!((p.encode.width, p.encode.height), (2160, 3840));
        // send_field doubles the conform rate; interpolation doubles again.
        assert_eq!(p.decode.conform_rate, Rational::new(60, 1));
        assert_eq!(p.encode.rate, Rational::new(120, 1));
        assert_eq!(p.total_frames, 600);
        assert_eq!(p.engine.scene_cuts, vec![200]);
        assert_eq!(p.segment_frames, 240);
        assert_eq!(p.segments_total(), 3);
        assert!(
            p.engine
                .stages
                .iter()
                .all(|s| s.kind() != StageKind::Deinterlace)
        );
        assert!(estimate_output_bytes(&p) > 0);
    }
}
