//! Device ranking: pick the most capable GPU by default, bypassing weak iGPUs.

use fs_types::{DeviceKind, DeviceUuid, GpuDevice, Vendor};

/// Compute `score` for each device and sort best-first.
pub fn rank(devices: &mut [GpuDevice]) {
    for d in devices.iter_mut() {
        d.score = score(d);
    }
    devices.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.name.cmp(&b.name)));
}

fn score(d: &GpuDevice) -> u32 {
    let class = match (d.kind, d.vendor) {
        (DeviceKind::Discrete, _) => 4000,
        // Apple Silicon is "integrated" on paper but is the only, and a strong, GPU.
        (DeviceKind::Integrated, Vendor::Apple) => 3000,
        (DeviceKind::Integrated, _) => 1000,
        (DeviceKind::Virtual, _) => 300,
        (DeviceKind::Other, _) => 100,
        (DeviceKind::Cpu, _) => 10,
    };
    let gib = (d.vram_bytes >> 30).min(48) as u32;
    // UMA heaps are shared system RAM; count them at half weight.
    let mem = if d.unified_memory { gib * 25 } else { gib * 50 };
    class + mem + if d.fp16 { 200 } else { 0 } + if d.cooperative_matrix { 300 } else { 0 }
}

/// The user's override if it still exists, otherwise the best-ranked device.
/// `devices` must already be ranked.
pub fn select_default<'a>(
    devices: &'a [GpuDevice],
    preferred: Option<&DeviceUuid>,
) -> Option<&'a GpuDevice> {
    preferred
        .and_then(|p| devices.iter().find(|d| &d.uuid == p))
        .or_else(|| devices.first())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn dev(name: &str, vendor: Vendor, kind: DeviceKind, gib: u64, uma: bool) -> GpuDevice {
        GpuDevice {
            uuid: DeviceUuid([name.len() as u8; 16]),
            name: name.into(),
            vendor,
            pci_vendor_id: 0,
            pci_device_id: 0,
            kind,
            driver_name: String::new(),
            driver_info: String::new(),
            api_version: "1.3".into(),
            vram_bytes: gib << 30,
            unified_memory: uma,
            fp16: true,
            cooperative_matrix: false,
            subgroup_min: 32,
            subgroup_max: 32,
            memory_budget: true,
            drm_render_minor: None,
            score: 0,
        }
    }

    #[test]
    fn laptop_prefers_dgpu_over_igpu_and_llvmpipe() {
        let mut v = vec![
            dev(
                "Intel Iris Xe",
                Vendor::Intel,
                DeviceKind::Integrated,
                16,
                true,
            ),
            dev("llvmpipe", Vendor::Other, DeviceKind::Cpu, 32, true),
            dev(
                "NVIDIA RTX 4060 Laptop",
                Vendor::Nvidia,
                DeviceKind::Discrete,
                8,
                false,
            ),
        ];
        rank(&mut v);
        assert_eq!(v[0].name, "NVIDIA RTX 4060 Laptop");
        assert_eq!(v[2].name, "llvmpipe");
        assert_eq!(
            select_default(&v, None).map(|d| d.name.as_str()),
            Some("NVIDIA RTX 4060 Laptop")
        );
        let igpu = v[1].uuid;
        assert_eq!(
            select_default(&v, Some(&igpu)).map(|d| d.uuid),
            Some(igpu),
            "manual override"
        );
        assert_eq!(
            select_default(&v, Some(&DeviceUuid([9; 16]))).map(|d| d.name.as_str()),
            Some("NVIDIA RTX 4060 Laptop"),
            "stale override falls back"
        );
    }

    #[test]
    fn apple_silicon_beats_cpu_fallbacks() {
        let mut v = vec![
            dev("llvmpipe", Vendor::Other, DeviceKind::Cpu, 64, true),
            dev(
                "Apple M3 Max",
                Vendor::Apple,
                DeviceKind::Integrated,
                48,
                true,
            ),
        ];
        rank(&mut v);
        assert_eq!(v[0].name, "Apple M3 Max");
    }
}
