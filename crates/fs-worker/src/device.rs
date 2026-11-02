//! Map the planned device (Vulkan UUID) to an ncnn GPU index.
//!
//! ncnn exposes no device UUID, so this process runs its own Vulkan
//! enumeration: both instances use the same loader in the same process, so
//! their orders agree. The match is then cross-checked by PCI vendor/device id
//! (and name) before it's trusted.

use std::path::PathBuf;

use fs_engine::EngineError;
use fs_engine::InferenceBackend;
use fs_engine::ncnn::{self, DeviceSel, GpuEntry, NcnnBackend};
use fs_types::worker::EnginePlan;
use fs_types::{DeviceKind, DeviceUuid, GpuDevice};

const NCNN_KIND_CPU: i32 = 3;

/// macOS: the bundled MoltenVK in `Contents/Frameworks` (dyld expands
/// `@executable_path`). `FS_VULKAN_DRIVER` overrides on any platform.
fn driver_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FS_VULKAN_DRIVER") {
        return Some(p.into());
    }
    if cfg!(target_os = "macos") {
        let exe = std::env::current_exe().ok()?;
        let fw = exe.parent()?.join("../Frameworks/libMoltenVK.dylib");
        if fw.exists() {
            return Some(PathBuf::from(
                "@executable_path/../Frameworks/libMoltenVK.dylib",
            ));
        }
    }
    None
}

fn sel(g: &GpuEntry, vram: Option<u64>) -> DeviceSel {
    DeviceSel {
        index: g.index,
        name: g.name.clone(),
        vram_bytes: vram,
    }
}

fn cpu() -> DeviceSel {
    DeviceSel {
        index: -1,
        name: "CPU (ncnn)".into(),
        vram_bytes: None,
    }
}

pub fn pick(want: Option<DeviceUuid>, gpus: &[GpuEntry], ours: &[GpuDevice]) -> DeviceSel {
    // Target: the planned UUID, else the best-ranked device.
    let target = match want {
        Some(u) => ours.iter().position(|d| d.uuid == u),
        None => {
            let mut ranked = ours.to_vec();
            fs_hw::rank(&mut ranked);
            ranked
                .first()
                .and_then(|b| ours.iter().position(|d| d.uuid == b.uuid))
        }
    };
    let chosen = target.and_then(|i| {
        let d = &ours[i];
        let ids = |g: &&GpuEntry| g.vendor_id == d.pci_vendor_id && g.device_id == d.pci_device_id;
        gpus.get(i)
            .filter(ids)
            .or_else(|| gpus.iter().find(|g| ids(g) && g.name == d.name))
            .or_else(|| gpus.iter().find(ids))
            .map(|g| (g, d))
    });
    match chosen {
        // A CPU "GPU" (llvmpipe/SwiftShader) is slower than ncnn's native CPU path.
        Some((g, d)) if g.kind != NCNN_KIND_CPU && d.kind != DeviceKind::Cpu => {
            sel(g, Some(d.vram_bytes))
        }
        Some(_) => cpu(),
        None => gpus
            .iter()
            .find(|g| g.kind != NCNN_KIND_CPU)
            .map_or_else(cpu, |g| sel(g, None)),
    }
}

pub fn ncnn_backend(plan: &EnginePlan) -> Result<Box<dyn InferenceBackend>, EngineError> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() as i32);
    let n = ncnn::init(driver_path().as_deref());
    let device = if n > 0 {
        let ours = fs_hw::VulkanContext::discover()
            .map(|(_, d)| d)
            .unwrap_or_default();
        pick(plan.device, &ncnn::gpus(), &ours)
    } else {
        cpu()
    };
    Ok(Box::new(NcnnBackend::new(plan, device, threads)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_types::Vendor;

    fn ours(name: &str, vendor: u32, device: u32, kind: DeviceKind, uuid: u8) -> GpuDevice {
        GpuDevice {
            uuid: DeviceUuid([uuid; 16]),
            name: name.into(),
            vendor: Vendor::from_pci_id(vendor),
            pci_vendor_id: vendor,
            pci_device_id: device,
            kind,
            driver_name: String::new(),
            driver_info: String::new(),
            api_version: "1.3".into(),
            vram_bytes: 8 << 30,
            unified_memory: false,
            fp16: true,
            cooperative_matrix: false,
            subgroup_min: 32,
            subgroup_max: 32,
            memory_budget: true,
            drm_render_minor: None,
            score: 0,
        }
    }
    fn g(i: i32, name: &str, v: u32, d: u32, kind: i32) -> GpuEntry {
        GpuEntry {
            index: i,
            name: name.into(),
            vendor_id: v,
            device_id: d,
            kind,
        }
    }

    #[test]
    fn picks_planned_uuid_and_defaults_to_dgpu() {
        let ours = vec![
            ours("Iris Xe", 0x8086, 1, DeviceKind::Integrated, 1),
            ours("RTX 4060", 0x10de, 2, DeviceKind::Discrete, 2),
        ];
        let gpus = vec![
            g(0, "Iris Xe", 0x8086, 1, 1),
            g(1, "RTX 4060", 0x10de, 2, 0),
        ];
        assert_eq!(pick(None, &gpus, &ours).index, 1, "dGPU by default");
        assert_eq!(
            pick(Some(DeviceUuid([1; 16])), &gpus, &ours).index,
            0,
            "manual iGPU override"
        );
    }

    #[test]
    fn survives_order_mismatch_and_prefers_native_cpu_over_llvmpipe() {
        let ours = vec![ours("RTX 4060", 0x10de, 2, DeviceKind::Discrete, 2)];
        // ncnn listed devices in another order (e.g. one skipped on our side).
        let gpus = vec![
            g(0, "llvmpipe", 0x10005, 0, 3),
            g(1, "RTX 4060", 0x10de, 2, 0),
        ];
        assert_eq!(pick(None, &gpus, &ours).index, 1);
        let only_cpu = vec![self::ours("llvmpipe", 0x10005, 0, DeviceKind::Cpu, 9)];
        assert_eq!(
            pick(None, &[g(0, "llvmpipe", 0x10005, 0, 3)], &only_cpu).index,
            -1
        );
    }
}
