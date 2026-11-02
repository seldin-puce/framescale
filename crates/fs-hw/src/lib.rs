//! GPU discovery, ranking, and live telemetry.
//!
//! - Vulkan enumeration via `ash` (dlopen'd loader): device UUID, driver ID,
//!   heaps, FP16 / cooperative-matrix / subgroup features, `VK_EXT_memory_budget`.
//! - Ranking: discrete ≫ integrated, then VRAM and compute features.
//!   User overrides are persisted by device UUID, never by index.
//! - `TelemetrySource` backends: NVML (NVIDIA), ADLX / amdgpu sysfs (AMD),
//!   IGCL / i915-xe sysfs (Intel), IOKit + thermalState (Apple).
//!
//! `unsafe` is denied crate-wide and allowed only in the FFI modules.

#![deny(unsafe_code)]
