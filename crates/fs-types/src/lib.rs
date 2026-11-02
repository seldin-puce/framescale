//! Shared domain model and IPC contract types.
//!
//! Every type that crosses a process boundary lives here: UI ⇄ core (Tauri IPC)
//! and core ⇄ fs-worker (JSON-lines over stdio). Types derive `serde` and
//! `ts_rs::TS`; `cargo xtask gen-types` exports them to
//! `packages/ipc-types/src/generated`, and CI fails on drift.
//!
//! Planned modules (Phase 3):
//! - `media`    — `MediaInfo`, `ColorProfile` (primaries/trc/matrix/range, HDR side data)
//! - `pipeline` — `StageKind` (canonical order), `StageConfig`, `PipelinePlan`
//! - `job`      — `JobId`, `JobSpec`, `JobState`, `SegmentRecord`, `PauseMode`
//! - `device`   — `GpuDevice`, `DeviceUuid`, `BackendKind`, `Capabilities`
//! - `event`    — `JobEvent`, `Telemetry`, worker protocol messages
//! - `license`  — `Entitlements`, `Feature`

#![forbid(unsafe_code)]
