//! Live telemetry sources, merged field-by-field: vendor APIs (most accurate)
//! override generic ones. Every reading is optional. The HUD shows "—" rather
//! than inventing numbers.

use std::path::PathBuf;
use std::sync::Arc;

use fs_types::{DeviceTelemetry, GpuDevice, Vendor};

use crate::vulkan::VulkanContext;

pub trait TelemetrySource: Send {
    fn name(&self) -> &'static str;
    /// Fill in whatever this source knows. Fields already set are left alone
    /// (sources are queried highest-priority first).
    fn sample(&mut self, into: &mut DeviceTelemetry);
}

/// Composite sampler for one device.
pub struct Sampler {
    device: GpuDevice,
    sources: Vec<Box<dyn TelemetrySource>>,
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<_> = self.sources.iter().map(|s| s.name()).collect();
        f.debug_struct("Sampler")
            .field("device", &self.device.name)
            .field("sources", &names)
            .finish()
    }
}

impl Sampler {
    /// Build the best available source stack for `device`.
    pub fn for_device(device: GpuDevice, vulkan: Option<Arc<VulkanContext>>) -> Self {
        let mut sources: Vec<Box<dyn TelemetrySource>> = Vec::new();
        if device.vendor == Vendor::Nvidia {
            match NvmlSource::new(&device) {
                Some(s) => sources.push(Box::new(s)),
                None => tracing::info!("NVML unavailable; falling back to generic telemetry"),
            }
        }
        if let Some(s) = SysfsSource::new(&device) {
            sources.push(Box::new(s));
        }
        // Windows: load, temperature and power share for any vendor (what
        // Task Manager shows); NVML above still wins for NVIDIA.
        #[cfg(windows)]
        if let Some(s) = vulkan
            .as_ref()
            .and_then(|v| v.luid(&device.uuid))
            .and_then(crate::telemetry_windows::WindowsGpuSource::new)
        {
            sources.push(Box::new(s));
        }
        if let Some(vk) = vulkan {
            sources.push(Box::new(VulkanBudgetSource {
                ctx: vk,
                device: device.clone(),
            }));
        }
        Self { device, sources }
    }

    pub fn sample(&mut self) -> DeviceTelemetry {
        let mut t = DeviceTelemetry {
            uuid: Some(self.device.uuid),
            ..Default::default()
        };
        for s in &mut self.sources {
            s.sample(&mut t);
        }
        if t.vram_total.is_none() && self.device.vram_bytes > 0 {
            t.vram_total = Some(self.device.vram_bytes);
        }
        t
    }

    pub fn source_names(&self) -> Vec<&'static str> {
        self.sources.iter().map(|s| s.name()).collect()
    }
}

pub(crate) fn set<T>(slot: &mut Option<T>, v: Option<T>) {
    if slot.is_none() {
        *slot = v;
    }
}

// --- NVML --------------------------------------------------------------------

struct NvmlSource {
    nvml: nvml_wrapper::Nvml,
    /// NVML's UUID string ("GPU-xxxxxxxx-xxxx-…") matches Vulkan's deviceUUID.
    uuid: String,
}

impl NvmlSource {
    fn new(d: &GpuDevice) -> Option<Self> {
        let nvml = nvml_wrapper::Nvml::init().ok()?;
        let h = d.uuid.to_string();
        let uuid = format!(
            "GPU-{}-{}-{}-{}-{}",
            &h[0..8],
            &h[8..12],
            &h[12..16],
            &h[16..20],
            &h[20..32]
        );
        nvml.device_by_uuid(uuid.as_str()).ok()?;
        Some(Self { nvml, uuid })
    }
}

impl TelemetrySource for NvmlSource {
    fn name(&self) -> &'static str {
        "nvml"
    }
    fn sample(&mut self, t: &mut DeviceTelemetry) {
        use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
        let Ok(dev) = self.nvml.device_by_uuid(self.uuid.as_str()) else {
            return;
        };
        set(
            &mut t.utilization_pct,
            dev.utilization_rates().ok().map(|u| u.gpu as f32),
        );
        set(
            &mut t.temperature_c,
            dev.temperature(TemperatureSensor::Gpu)
                .ok()
                .map(|c| c as f32),
        );
        set(
            &mut t.power_w,
            dev.power_usage().ok().map(|mw| mw as f32 / 1000.0),
        );
        if let Ok(m) = dev.memory_info() {
            set(&mut t.vram_used, Some(m.used));
            set(&mut t.vram_total, Some(m.total));
        }
    }
}

// --- Linux sysfs (amdgpu, i915/xe) --------------------------------------------

struct SysfsSource {
    dev_dir: PathBuf,
    hwmon: Option<PathBuf>,
}

impl SysfsSource {
    fn new(d: &GpuDevice) -> Option<Self> {
        if !cfg!(target_os = "linux") {
            return None;
        }
        let minor = d.drm_render_minor?;
        let dev_dir = PathBuf::from(format!("/sys/class/drm/renderD{minor}/device"));
        if !dev_dir.exists() {
            return None;
        }
        let hwmon = std::fs::read_dir(dev_dir.join("hwmon"))
            .ok()
            .and_then(|mut it| it.find_map(|e| e.ok().map(|e| e.path())));
        Some(Self { dev_dir, hwmon })
    }

    fn read_u64(p: PathBuf) -> Option<u64> {
        std::fs::read_to_string(p).ok()?.trim().parse().ok()
    }
}

impl TelemetrySource for SysfsSource {
    fn name(&self) -> &'static str {
        "sysfs"
    }
    fn sample(&mut self, t: &mut DeviceTelemetry) {
        set(
            &mut t.utilization_pct,
            Self::read_u64(self.dev_dir.join("gpu_busy_percent")).map(|v| v as f32),
        );
        set(
            &mut t.vram_used,
            Self::read_u64(self.dev_dir.join("mem_info_vram_used")),
        );
        set(
            &mut t.vram_total,
            Self::read_u64(self.dev_dir.join("mem_info_vram_total")),
        );
        if let Some(h) = &self.hwmon {
            set(
                &mut t.temperature_c,
                Self::read_u64(h.join("temp1_input")).map(|m| m as f32 / 1000.0),
            );
            let power = Self::read_u64(h.join("power1_average"))
                .or_else(|| Self::read_u64(h.join("power1_input")));
            set(&mut t.power_w, power.map(|uw| uw as f32 / 1_000_000.0));
        }
    }
}

// --- Vulkan memory budget (vendor-neutral VRAM) -------------------------------

struct VulkanBudgetSource {
    ctx: Arc<VulkanContext>,
    device: GpuDevice,
}

impl TelemetrySource for VulkanBudgetSource {
    fn name(&self) -> &'static str {
        "vk-memory-budget"
    }
    fn sample(&mut self, t: &mut DeviceTelemetry) {
        if let Some((used, total)) = self.ctx.memory_usage(&self.device.uuid) {
            set(&mut t.vram_used, Some(used));
            set(&mut t.vram_total, Some(total));
        }
    }
}

// SAFETY-free note: `VulkanContext` holds only Vulkan handles that are safe to
// query from any thread (vkGetPhysicalDevice* are externally synchronised-free).
#[allow(unsafe_code)]
unsafe impl Send for VulkanBudgetSource {}
