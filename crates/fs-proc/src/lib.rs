//! Process supervision and the zombie reaper.
//!
//! - Windows: the app joins a `KILL_ON_JOB_CLOSE` Job Object at startup, and
//!   every descendant inherits it.
//! - Unix: workers run in their own process group with `PR_SET_PDEATHSIG`
//!   (Linux) / kqueue `NOTE_EXIT` (macOS) parent watchdogs.
//! - PID ledger `(pid, start_time, exe)` for safe kills of stale processes on startup.
//! - Graceful shutdown: SIGTERM → grace period → SIGKILL to the whole group.

#![deny(unsafe_code)]
