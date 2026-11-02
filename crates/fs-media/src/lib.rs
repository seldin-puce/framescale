//! ffprobe parsing, typed FFmpeg argv builder, progress and stderr parsing.
//!
//! FFmpeg is never linked; it always runs as a sidecar process. This keeps
//! the LGPL boundary clean. Command lines are built from typed plans and
//! never by concatenating strings, so user-controlled paths can't inject flags.
//!
//! Planned modules: `probe`, `color` (tag inference for untagged sources),
//! `argv` (decode / encode / analyze / concat / mux builders, per-vendor hwaccel),
//! `progress` (`-progress` key=value parser), `classify` (stderr → typed errors:
//! ENOSPC, encoder session limit, corrupt packets, …).

#![forbid(unsafe_code)]
