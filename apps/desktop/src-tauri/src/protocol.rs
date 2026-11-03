//! `fs://` protocol: serves preview frames and thumbnails to the webview.
//!
//! Only two roots are reachable (`preview/…`, `thumbs/…`). Every path segment
//! is validated, so a crafted URL can't climb out with `..`, absolute paths or
//! encoded separators. Files are content-addressed (UUID / hash names), so they
//! are cached as immutable.

use std::path::PathBuf;
use std::sync::Arc;

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

pub fn serve(state: &Arc<AppState>, req: &Request<Vec<u8>>) -> Response<Vec<u8>> {
    let respond = |status: StatusCode, body: Vec<u8>| {
        Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "image/jpeg")
            .header(header::CACHE_CONTROL, "public, max-age=31536000, immutable")
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .body(body)
            .unwrap_or_else(|_| Response::new(Vec::new()))
    };
    match resolve(state, req.uri().path()).map(std::fs::read) {
        Some(Ok(bytes)) => respond(StatusCode::OK, bytes),
        Some(Err(_)) => respond(StatusCode::NOT_FOUND, Vec::new()),
        None => {
            tracing::warn!(path = %req.uri().path(), "fs:// request outside the allowed roots");
            respond(StatusCode::FORBIDDEN, Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::safe_segment;

    #[test]
    fn rejects_traversal() {
        for bad in ["..", ".", "", "a/b", "a\\b", "%2e%2e", "~root"] {
            assert!(!safe_segment(bad), "{bad}");
        }
        assert!(safe_segment("01a19ed6-c3be-74a6.jpg"));
    }
}
