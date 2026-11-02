//! GPU device description (from Vulkan discovery).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use ts_rs::TS;

/// `VkPhysicalDeviceIDProperties::deviceUUID`. It's stable across processes
/// and reboots, unlike enumeration indices. Serialized as 32 hex chars.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TS)]
#[ts(export, type = "string")]
pub struct DeviceUuid(pub [u8; 16]);

impl fmt::Display for DeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

impl fmt::Debug for DeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeviceUuid({self})")
    }
}

impl FromStr for DeviceUuid {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.len() != 32 || !s.is_ascii() {
            return Err("device uuid must be 32 hex characters");
        }
        let mut out = [0u8; 16];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|_| "invalid hex")?;
        }
        Ok(Self(out))
    }
}

impl Serialize for DeviceUuid {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for DeviceUuid {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

impl Vendor {
    /// From the PCI vendor id reported by Vulkan.
    pub fn from_pci_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 | 0x1022 => Self::Amd,
            0x8086 => Self::Intel,
            0x106B => Self::Apple,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum DeviceKind {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct GpuDevice {
    pub uuid: DeviceUuid,
    pub name: String,
    pub vendor: Vendor,
    /// Raw PCI ids; used to cross-check the inference runtime's device list.
    pub pci_vendor_id: u32,
    pub pci_device_id: u32,
    pub kind: DeviceKind,
    pub driver_name: String,
    pub driver_info: String,
    pub api_version: String,
    /// Total DEVICE_LOCAL heap size. On UMA devices this is shared system memory.
    #[ts(type = "number")]
    pub vram_bytes: u64,
    pub unified_memory: bool,
    pub fp16: bool,
    pub cooperative_matrix: bool,
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    pub memory_budget: bool,
    /// Linux DRM render node minor (`/dev/dri/renderD<minor>`), for VAAPI.
    pub drm_render_minor: Option<u32>,
    /// Ranking score (higher = preferred). Computed by `fs_hw::rank`.
    pub score: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_hex_roundtrip() {
        let u = DeviceUuid([0xab; 16]);
        let s = u.to_string();
        assert_eq!(s.len(), 32);
        assert_eq!(s.parse::<DeviceUuid>(), Ok(u));
        assert!("zz".parse::<DeviceUuid>().is_err());
    }
}
