//! Colour handling: tag inference for untagged sources, and the zscale
//! conversions into and out of the engine's RGB domain.
//!
//! Policy: the *transfer function is never changed*. PQ/HLG code values pass
//! through the AI untouched (no linearisation, no tone mapping), so the output
//! carries exactly the source's primaries, transfer and matrix tags. Only the
//! YUV⇄RGB matrix and the range are converted.

use fs_types::media::{
    ColorPrimaries as P, ColorProfile, ColorRange, MatrixCoefficients as M,
    TransferCharacteristic as T,
};

/// Raw (possibly missing) colour tags from ffprobe.
#[derive(Debug, Clone, Default)]
pub struct RawColorTags<'a> {
    pub primaries: Option<&'a str>,
    pub transfer: Option<&'a str>,
    pub matrix: Option<&'a str>,
    pub range: Option<&'a str>,
}

/// Resolve a complete profile. Missing tags are inferred the way broadcast
/// players do: HD and above → BT.709; 576-line → BT.601-625 (bt470bg);
/// other SD → BT.601-525 (smpte170m).
///
/// FFmpeg's swscale treats untagged HD as BT.601, which visibly shifts colours.
/// Explicit inference here avoids that.
pub fn resolve(
    tags: &RawColorTags<'_>,
    width: u32,
    height: u32,
    bit_depth: u8,
    rgb: bool,
) -> ColorProfile {
    let mut inferred = false;
    let hd = width >= 1280 || height >= 720;
    let pal = !hd && height == 576;

    let mut matrix = tags.matrix.map(M::from_ffmpeg).unwrap_or(M::Unknown);
    let mut primaries = tags.primaries.map(P::from_ffmpeg).unwrap_or(P::Unknown);
    let mut transfer = tags.transfer.map(T::from_ffmpeg).unwrap_or(T::Unknown);

    if rgb {
        matrix = M::Rgb;
    }
    if matrix == M::Unknown {
        inferred = true;
        matrix = match primaries {
            P::Bt2020 => M::Bt2020Nc,
            _ if hd => M::Bt709,
            _ if pal => M::Bt470bg,
            _ => M::Smpte170m,
        };
    }
    if primaries == P::Unknown {
        inferred = true;
        primaries = match matrix {
            M::Bt2020Nc | M::Bt2020C => P::Bt2020,
            M::Bt470bg => P::Bt470bg,
            M::Smpte170m => P::Smpte170m,
            _ => P::Bt709,
        };
    }
    if transfer == T::Unknown {
        inferred = true;
        // Without an explicit PQ/HLG tag we can't claim HDR. Assume SDR gamma.
        transfer = match primaries {
            P::Smpte170m | P::Bt470bg => T::Smpte170m,
            _ => T::Bt709,
        };
    }
    let range = match tags.range.and_then(ColorRange::from_ffmpeg) {
        Some(r) => r,
        None => {
            inferred = true;
            if rgb {
                ColorRange::Full
            } else {
                ColorRange::Limited
            }
        }
    };

    ColorProfile {
        primaries,
        transfer,
        matrix,
        range,
        bit_depth,
        mastering: None,
        content_light: None,
        inferred,
    }
}

fn name<E: Copy>(
    v: E,
    f: impl Fn(E) -> Option<&'static str>,
    fallback: &'static str,
) -> &'static str {
    f(v).unwrap_or(fallback)
}

fn tags_in(c: &ColorProfile) -> (&'static str, &'static str, &'static str) {
    (
        name(c.matrix, M::ffmpeg_name, "bt709"),
        name(c.primaries, P::ffmpeg_name, "bt709"),
        name(c.transfer, T::ffmpeg_name, "bt709"),
    )
}

// zimg only produces *planar* formats. Asking zscale for packed rgb24/rgb48le
// makes FFmpeg silently insert swscale with default BT.601 coefficients after
// it. So zscale always targets a planar format here, and the follow-up
// `format=` is a lossless planar⇄packed shuffle with no matrix involved.

/// Source YUV → full-range RGB with the *explicit* source matrix and range,
/// keeping the source primaries and transfer unchanged. Ends in planar
/// `gbrp`/`gbrp16le`; the caller appends the packed `format=`.
pub fn to_rgb_filter(c: &ColorProfile, high_precision: bool) -> String {
    let (m, p, t) = tags_in(c);
    let planar = if high_precision { "gbrp16le" } else { "gbrp" };
    let r = c.range.zscale_name();
    if c.matrix == M::Rgb {
        return format!("zscale=rangein={r}:range=full,format={planar}");
    }
    format!(
        "zscale=matrixin={m}:rangein={r}:primariesin={p}:transferin={t}:matrix=gbr:primaries={p}:transfer={t}:range=full,format={planar}"
    )
}

/// Planar YUV format zscale should produce for an encoder pixel format
/// (semi-planar targets like p010/nv12 are filled by a lossless shuffle).
pub fn planar_for(pix_fmt: &str) -> &'static str {
    match pix_fmt {
        "p010le" | "yuv420p10le" => "yuv420p10le",
        "nv12" | "yuv420p" => "yuv420p",
        "p210le" | "yuv422p10le" => "yuv422p10le",
        "p410le" | "yuv444p10le" => "yuv444p10le",
        "ayuv64le" | "yuv444p16le" => "yuv444p16le",
        _ => "yuv420p10le",
    }
}

/// Engine RGB (full range) → output YUV with the output matrix and range,
/// then stamp the frames' colour properties. FFmpeg ≥ 7 takes encoder colour
/// tags from the frames, so `-color_trc` alone is not enough.
pub fn from_rgb_filter(c: &ColorProfile, pix_fmt: &str) -> String {
    let (m, p, t) = tags_in(c);
    let r = c.range.zscale_name();
    format!(
        "zscale=rangein=full:primariesin={p}:transferin={t}:matrix={m}:primaries={p}:transfer={t}:range={r}:dither=error_diffusion,format={planar},setparams=color_primaries={p}:color_trc={t}:colorspace={m}:range={fr}",
        planar = planar_for(pix_fmt),
        fr = c.range.ffmpeg_name()
    )
}

/// Full-range RGB in the source colour space → JPEG-ready `yuvj444p` for
/// *display* (previews, thumbnails). JPEG decoders assume BT.601 full range.
/// PQ/HLG are tone-mapped (Hable) to SDR BT.709 first; the render itself is
/// never tone-mapped.
pub fn display_jpeg_filter(c: &ColorProfile) -> String {
    // zimg needs every stage fully specified (in *and* out), or it reports
    // "no path between colorspaces". Verified against the pinned FFmpeg 8.1.
    let (_, p, t) = tags_in(c);
    let rgb_in = format!("matrixin=gbr:rangein=full:primariesin={p}:transferin={t}");
    if c.is_hdr() {
        format!(
            "zscale={rgb_in}:matrix=gbr:range=full:primaries={p}:transfer=linear:npl=100,format=gbrpf32le,\
             zscale=primaries=bt709,tonemap=tonemap=hable:desat=0,zscale=transfer=bt709,format=gbrp,\
             zscale=matrix=470bg:range=full,format=yuvj444p"
        )
    } else {
        format!(
            "zscale={rgb_in}:matrix=470bg:range=full:primaries={p}:transfer={t},format=yuvj444p"
        )
    }
}

/// Container/bitstream colour tags for the encoder output.
pub fn output_tags(c: &ColorProfile) -> Vec<String> {
    let (m, p, t) = tags_in(c);
    [
        "-color_primaries",
        p,
        "-color_trc",
        t,
        "-colorspace",
        m,
        "-color_range",
        c.range.ffmpeg_name(),
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

/// Bit depth from an FFmpeg pixel format name (`yuv420p10le` → 10, `p010le` → 10).
pub fn bit_depth_of(pix_fmt: &str) -> u8 {
    if pix_fmt.starts_with("nv") {
        return 8; // nv12/nv16/nv21/nv24: the digits are the chroma layout, not the depth
    }
    let digits: String = pix_fmt
        .trim_end_matches("le")
        .trim_end_matches("be")
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    match (pix_fmt, digits.parse::<u8>().ok()) {
        (_, Some(d @ (9 | 10 | 12 | 14 | 16))) => d,
        (f, _) if f.starts_with("p01") || f.starts_with("p21") || f.starts_with("p41") => 10,
        (f, _) if f.starts_with("p016") || f.starts_with("p216") || f.starts_with("p416") => 16,
        (f, _) if f.contains("48") || f.contains("64") => 16,
        _ => 8,
    }
}

pub fn is_rgb_pix_fmt(pix_fmt: &str) -> bool {
    pix_fmt.starts_with("rgb") || pix_fmt.starts_with("bgr") || pix_fmt.starts_with("gbr")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untagged_hd_is_bt709_not_601() {
        let c = resolve(&RawColorTags::default(), 1920, 1080, 8, false);
        assert_eq!(
            (c.matrix, c.primaries, c.transfer),
            (M::Bt709, P::Bt709, T::Bt709)
        );
        assert_eq!(c.range, ColorRange::Limited);
        assert!(c.inferred);
    }

    #[test]
    fn untagged_sd_pal_vs_ntsc() {
        assert_eq!(
            resolve(&RawColorTags::default(), 720, 576, 8, false).matrix,
            M::Bt470bg
        );
        assert_eq!(
            resolve(&RawColorTags::default(), 720, 480, 8, false).matrix,
            M::Smpte170m
        );
    }

    #[test]
    fn hdr10_tags_pass_through() {
        let tags = RawColorTags {
            primaries: Some("bt2020"),
            transfer: Some("smpte2084"),
            matrix: Some("bt2020nc"),
            range: Some("tv"),
        };
        let c = resolve(&tags, 3840, 2160, 10, false);
        assert!(c.is_hdr() && !c.inferred);
        let f = to_rgb_filter(&c, true);
        assert!(
            f.contains("matrixin=bt2020nc") && f.contains("transferin=smpte2084:"),
            "{f}"
        );
        assert!(
            f.contains(":transfer=smpte2084:"),
            "transfer must not change: {f}"
        );
        assert!(
            f.contains("matrix=gbr") && f.ends_with("format=gbrp16le"),
            "{f}"
        );
        let e = from_rgb_filter(&c, "p010le");
        assert!(
            e.contains("matrix=bt2020nc") && e.contains("format=yuv420p10le,setparams="),
            "{e}"
        );
        assert!(
            e.ends_with("color_trc=smpte2084:colorspace=bt2020nc:range=tv"),
            "{e}"
        );
        assert_eq!(output_tags(&c)[3], "smpte2084");
    }

    #[test]
    fn bit_depths() {
        assert_eq!(bit_depth_of("yuv420p10le"), 10);
        assert_eq!(bit_depth_of("p010le"), 10);
        assert_eq!(bit_depth_of("yuv420p"), 8);
        assert_eq!(bit_depth_of("yuv422p12le"), 12);
        assert_eq!(bit_depth_of("rgb48le"), 16);
        assert_eq!(bit_depth_of("nv12"), 8);
    }
}
