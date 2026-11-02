//! Inference engine: `InferenceBackend` trait, ncnn-Vulkan backend, tiler, color converters.
//!
//! Backends:
//! - `ncnn`: baseline on every platform (MoltenVK on macOS). Built from the
//!   pinned ncnn source (`third_party/manifest.toml`) via `cmake` + a `cxx`
//!   bridge in `cpp/`.
//! - `coreml`: native Apple backend (later).
//! - `tensorrt`: optional NVIDIA backend (later).
//!
//! Also owns: auto-tiling with overlap and feathered seams, OOM tile backoff,
//! per-dispatch time bounds (TDR safety), and u8/u16 ⇄ fp16 conversion shaders.

#![deny(unsafe_code)]
