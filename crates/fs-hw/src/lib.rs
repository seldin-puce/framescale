//! GPU discovery, ranking, live telemetry, and the thermal governor.
//!
//! - [`vulkan`]: enumerate devices via the dlopen'd Vulkan loader (MoltenVK
//!   on macOS). Devices are identified by UUID, never by index.
//! - [`rank`]: discrete ≫ integrated, then memory and compute features.
//! - [`telemetry`]: NVML (NVIDIA), sysfs (AMD/Intel on Linux), WDDM perf data
//!   and GPU Engine counters (any GPU on Windows), IOKit / IOReport / SMC
//!   (Apple Silicon), and `VK_EXT_memory_budget` (vendor-neutral VRAM),
//!   merged per field.
//! - [`governor`]: Eco mode duty-cycle controller.
//!
//! `unsafe` is denied crate-wide and allowed only in the FFI modules (Vulkan,
//! Windows and macOS telemetry, the battery query).
#![deny(unsafe_code)]

pub mod governor;
pub mod power;
pub mod rank;
pub mod telemetry;
#[cfg(target_os = "macos")]
mod telemetry_macos;
#[cfg(windows)]
mod telemetry_windows;
pub mod vulkan;

pub use rank::{rank, select_default};
pub use vulkan::{DiscoveryError, VulkanContext};

/// Free bytes on the volume containing `path` (for StorageGuard / HUD).
pub fn disk_free(path: &std::path::Path) -> std::io::Result<u64> {
    fs4::available_space(path)
}
