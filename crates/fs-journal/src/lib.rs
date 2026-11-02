//! Crash-consistent job and segment journal (SQLite, WAL).
//!
//! Source of truth for resume. A segment counts as committed only after its
//! file is fsync'd and renamed and this journal records it. Migrations live
//! in `migrations/` and are embedded at compile time.

#![forbid(unsafe_code)]
