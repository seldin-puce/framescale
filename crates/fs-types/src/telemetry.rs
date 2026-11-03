//! Live hardware telemetry pushed to the HUD.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::device::DeviceUuid;

/// Every reading is optional: a vendor or OS may not expose it, and the HUD
/// shows "—" rather than a made-up number.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DeviceTelemetry {
    pub uuid: Option<DeviceUuid>,
    pub utilization_pct: Option<f32>,
    #[ts(type = "number | null")]
    pub vram_used: Option<u64>,
    #[ts(type = "number | null")]
    pub vram_total: Option<u64>,
    pub temperature_c: Option<f32>,
    pub power_w: Option<f32>,
    /// Power as a share of the board's limit, when watts aren't known
    /// (Windows reports this for every GPU).
    pub power_pct: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, TS)]
#[serde(
    tag = "mode",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
#[ts(export)]
pub enum EcoMode {
    #[default]
    Off,
    /// Temperature-targeting controller (PI on GPU temperature, or on battery).
    Auto { target_c: f32 },
    /// Fixed GPU duty cycle, 10–100 %.
    Fixed { duty_pct: u8 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct EcoState {
    pub mode: EcoMode,
    /// Current duty cycle applied by the governor (0–1].
    pub duty: f32,
    pub throttling: bool,
    pub on_battery: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Telemetry {
    #[ts(type = "number")]
    pub at_ms: u64,
    pub device: DeviceTelemetry,
    pub render_fps: Option<f32>,
    pub eco: EcoState,
    #[ts(type = "number | null")]
    pub disk_free_bytes: Option<u64>,
}
