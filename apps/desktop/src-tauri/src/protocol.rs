//! `fs://` protocol: serves preview frames, thumbnails and comparison frames
//! to the webview.
//!
//! Only two disk roots are reachable (`preview/…`, `thumbs/…`). Every path
//! segment is validated, so a crafted URL can't climb out with `..`, absolute
//! paths or encoded separators. Files are content-addressed (UUID / hash
//! names), so they are cached as immutable. `compare/<session>/<side>/[full/]<n>.jpg`
//! never touches the disk: frames come from the session's in-memory decoders.

use std::path::PathBuf;
use std::sync::Arc;

use fs_types::CompareSide;
use tauri::http::{Request, Response, StatusCode, header};

use crate::state::AppState;

fn safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

pub fn resolve(state: &AppState, uri_path: &str) -> Option<PathBuf> {
    let mut parts = uri_path.trim_start_matches('/').split('/');
    let root = match parts.next()? {
        "preview" => state.paths.previews.clone(),
        "thumbs" => state.paths.thumbs.clone(),
        _ => return None,
    };
    let rest: Vec<&str> = parts.collect();
    if rest.is_empty() || !rest.iter().all(|s| safe_segment(s)) || !rest.last()?.ends_with(".jpg") {
        return None;
    }
    Some(rest.iter().fold(root, |p, s| p.join(s)))
}

fn respond(status: StatusCode, body: Vec<u8>, cacheable: bool) -> Response<Vec<u8>> {
    let ok = status == StatusCode::OK;
    Response::builder()
        .status(status)
        .header(
            header::CONTENT_TYPE,
            if ok {
                "image/jpeg"
            } else {
                "text/plain; charset=utf-8"
            },
        )
        .header(
            header::CACHE_CONTROL,
            if ok && cacheable {
                "public, max-age=31536000, immutable"
            } else {
                "no-store"
            },
        )
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(body)
        .unwrap_or_else(|_| Response::new(Vec::new()))
}

pub async fn serve(state: Arc<AppState>, req: Request<Vec<u8>>) -> Response<Vec<u8>> {
    let path = req.uri().path().trim_start_matches('/').to_owned();
    if let Some(rest) = path.strip_prefix("compare/") {
        return compare_frame(&state, rest).await;
    }
    let file = {
        let st = Arc::clone(&state);
        let p = path.clone();
        tauri::async_runtime::spawn_blocking(move || resolve(&st, &p).map(std::fs::read)).await
    };
    match file {
        Ok(Some(Ok(bytes))) => respond(StatusCode::OK, bytes, true),
        Ok(Some(Err(_))) => respond(StatusCode::NOT_FOUND, Vec::new(), false),
        Ok(None) => {
            tracing::warn!(%path, "fs:// request outside the allowed roots");
            respond(StatusCode::FORBIDDEN, Vec::new(), false)
        }
        Err(_) => respond(StatusCode::INTERNAL_SERVER_ERROR, Vec::new(), false),
    }
}

/// `<session>/<before|after>/<n>.jpg` (playback size) or
/// `<session>/<before|after>/full/<n>.jpg` (native resolution).
fn parse_compare(rest: &str) -> Option<(&str, CompareSide, bool, u64)> {
    let parts: Vec<&str> = rest.split('/').collect();
    let (session, side, full, file) = match parts.as_slice() {
        [s, side, f] => (*s, *side, false, *f),
        [s, side, "full", f] => (*s, *side, true, *f),
        _ => return None,
    };
    let side = match side {
        "before" => CompareSide::Before,
        "after" => CompareSide::After,
        _ => return None,
    };
    let n = file.strip_suffix(".jpg")?;
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((session, side, full, n.parse().ok()?))
}

async fn compare_frame(state: &AppState, rest: &str) -> Response<Vec<u8>> {
    use fs_core::compare::FrameError;
    let Some((id, side, full, n)) = parse_compare(rest) else {
        return respond(StatusCode::FORBIDDEN, Vec::new(), false);
    };
    let session = state
        .compare
        .lock()
        .ok()
        .and_then(|open| open.iter().find(|s| s.info.session == id).cloned());
    let Some(session) = session else {
        return respond(StatusCode::NOT_FOUND, b"comparison closed".to_vec(), false);
    };
    let frame = if full {
        session.still(side, n).await
    } else {
        session.frame(side, n).await
    };
    match frame {
        // Frame content depends on the session's decoders: never cached.
        Ok(jpeg) => respond(StatusCode::OK, jpeg.to_vec(), false),
        Err(e) => {
            let status = match e {
                FrameError::Closed => StatusCode::NOT_FOUND,
                FrameError::PastEnd => StatusCode::RANGE_NOT_SATISFIABLE,
                FrameError::Superseded => StatusCode::CONFLICT,
                FrameError::TimedOut => StatusCode::GATEWAY_TIMEOUT,
                FrameError::Decode(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            if let FrameError::Decode(msg) = &e {
                tracing::warn!(%msg, "comparison decode failed");
            }
            respond(status, e.to_string().into_bytes(), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_compare, safe_segment};
    use fs_types::CompareSide;

    #[test]
    fn rejects_traversal() {
        for bad in ["..", ".", "", "a/b", "a\\b", "%2e%2e", "~root"] {
            assert!(!safe_segment(bad), "{bad}");
        }
        assert!(safe_segment("01a19ed6-c3be-74a6.jpg"));
    }

    #[test]
    fn parses_compare_routes() {
        assert_eq!(
            parse_compare("abc/before/12.jpg"),
            Some(("abc", CompareSide::Before, false, 12))
        );
        assert_eq!(
            parse_compare("abc/after/full/7.jpg"),
            Some(("abc", CompareSide::After, true, 7))
        );
        for bad in [
            "abc/side/1.jpg",
            "abc/after/x.jpg",
            "abc/after/-1.jpg",
            "abc/after/1.png",
            "abc/after/.jpg",
            "abc/after/half/1.jpg",
            "abc/after/full/1.jpg/x",
        ] {
            assert_eq!(parse_compare(bad), None, "{bad}");
        }
    }
}
