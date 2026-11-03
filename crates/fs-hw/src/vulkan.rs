//! Vulkan device discovery and VRAM-budget queries.
//!
//! The loader is dlopen'd (`ash::Entry::load`), so a machine without Vulkan
//! still runs the app. Discovery then returns an error and the UI offers the
//! CPU path. On macOS the loader resolves to MoltenVK, which needs
//! `VK_KHR_portability_enumeration` before it will list its devices.
#![allow(unsafe_code)]

use std::ffi::CStr;

use ash::vk;
use fs_types::{DeviceKind, DeviceUuid, GpuDevice, Vendor};

#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("Vulkan loader not found: {0}")]
    NoLoader(#[from] ash::LoadingError),
    #[error("Vulkan call failed: {0}")]
    Vk(#[from] vk::Result),
    #[error("Vulkan 1.1+ instance required (found {0})")]
    TooOld(String),
}

/// Owns the Vulkan instance for the app's lifetime. Discovery runs once, and
/// telemetry reuses the instance for `VK_EXT_memory_budget` polling.
pub struct VulkanContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    devices: Vec<(DeviceUuid, vk::PhysicalDevice, bool)>, // (uuid, handle, has_memory_budget)
}

impl std::fmt::Debug for VulkanContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanContext")
            .field("devices", &self.devices.len())
            .finish()
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: no child objects (devices) were created from this instance.
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn version_string(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

fn cstr_field(bytes: &[std::ffi::c_char]) -> String {
    // Vulkan fixed-size strings are NUL-terminated within the array.
    let b: Vec<u8> = bytes
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8_lossy(&b).into_owned()
}

/// The Vulkan entry point: `FS_VULKAN_DRIVER` when set; on macOS the MoltenVK
/// bundled in `Contents/Frameworks` (Macs have no system Vulkan loader, and
/// the worker's ncnn loads that same library, so device orders agree);
/// otherwise the system loader.
fn load_entry() -> Result<ash::Entry, ash::LoadingError> {
    if let Some(p) = std::env::var_os("FS_VULKAN_DRIVER") {
        // SAFETY: dlopen of the Vulkan library the user pointed us at.
        return unsafe { ash::Entry::load_from(p) };
    }
    #[cfg(target_os = "macos")]
    if let Some(fw) = std::env::current_exe()
        .ok()
        .and_then(|e| Some(e.parent()?.join("../Frameworks/libMoltenVK.dylib")))
        .filter(|p| p.exists())
    {
        // SAFETY: the MoltenVK we ship; it exports vkGetInstanceProcAddr.
        return unsafe { ash::Entry::load_from(fw) };
    }
    // SAFETY: dlopen of the system Vulkan loader. The Entry keeps it loaded.
    unsafe { ash::Entry::load() }
}

impl VulkanContext {
    /// Load the loader, create an instance, and describe every physical device.
    pub fn discover() -> Result<(Self, Vec<GpuDevice>), DiscoveryError> {
        let entry = load_entry()?;
        // SAFETY: valid entry; returns None on 1.0-only loaders.
        let instance_version =
            unsafe { entry.try_enumerate_instance_version()? }.unwrap_or(vk::API_VERSION_1_0);
        if instance_version < vk::API_VERSION_1_1 {
            return Err(DiscoveryError::TooOld(version_string(instance_version)));
        }
        let api = instance_version.min(vk::API_VERSION_1_3);

        // SAFETY: valid entry.
        let inst_exts = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let has_ext = |name: &CStr| {
            inst_exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let mut ext_ptrs = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        if has_ext(vk::KHR_PORTABILITY_ENUMERATION_NAME) {
            ext_ptrs.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        let app = vk::ApplicationInfo::default()
            .application_name(c"FrameScale AI")
            .engine_name(c"fs-hw")
            .api_version(api);
        let ci = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&ext_ptrs)
            .flags(flags);
        // SAFETY: create info and the pointed-to strings outlive the call.
        let instance = unsafe { entry.create_instance(&ci, None)? };

        let mut ctx = Self {
            _entry: entry,
            instance,
            devices: Vec::new(),
        };
        // SAFETY: valid instance.
        let handles = unsafe { ctx.instance.enumerate_physical_devices()? };
        let mut out = Vec::with_capacity(handles.len());
        for pd in handles {
            match ctx.describe(pd) {
                Ok(d) => {
                    ctx.devices.push((d.uuid, pd, d.memory_budget));
                    out.push(d);
                }
                Err(e) => tracing::warn!(error = %e, "skipping Vulkan device"),
            }
        }
        Ok((ctx, out))
    }

    fn describe(&self, pd: vk::PhysicalDevice) -> Result<GpuDevice, DiscoveryError> {
        let inst = &self.instance;
        // SAFETY: `pd` came from this instance.
        let props = unsafe { inst.get_physical_device_properties(pd) };
        let dev_api = props.api_version;
        // SAFETY: as above.
        let exts = unsafe { inst.enumerate_device_extension_properties(pd)? };
        let has = |n: &CStr| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(n));

        let has_driver_props =
            dev_api >= vk::API_VERSION_1_2 || has(vk::KHR_DRIVER_PROPERTIES_NAME);
        let has_sg_control =
            dev_api >= vk::API_VERSION_1_3 || has(vk::EXT_SUBGROUP_SIZE_CONTROL_NAME);
        let has_drm = has(vk::EXT_PHYSICAL_DEVICE_DRM_NAME);
        let has_f16 = dev_api >= vk::API_VERSION_1_2 || has(vk::KHR_SHADER_FLOAT16_INT8_NAME);

        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut sg_ctl = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut id)
                .push_next(&mut subgroup);
            if has_driver_props {
                p2 = p2.push_next(&mut driver);
            }
            if has_sg_control {
                p2 = p2.push_next(&mut sg_ctl);
            }
            if has_drm {
                p2 = p2.push_next(&mut drm);
            }
            // SAFETY: every chained struct is supported by this device (checked above).
            unsafe { inst.get_physical_device_properties2(pd, &mut p2) };
        }

        let mut f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        {
            let mut f2 = vk::PhysicalDeviceFeatures2::default();
            if has_f16 {
                f2 = f2.push_next(&mut f16);
            }
            // SAFETY: as above.
            unsafe { inst.get_physical_device_features2(pd, &mut f2) };
        }

        // SAFETY: as above.
        let mem = unsafe { inst.get_physical_device_memory_properties(pd) };
        let heaps = &mem.memory_heaps[..mem.memory_heap_count as usize];
        let vram = heaps
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);

        let kind = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::Integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::Virtual,
            vk::PhysicalDeviceType::CPU => DeviceKind::Cpu,
            _ => DeviceKind::Other,
        };
        let vendor = Vendor::from_pci_id(props.vendor_id);
        let (sg_min, sg_max) = if has_sg_control && sg_ctl.min_subgroup_size > 0 {
            (sg_ctl.min_subgroup_size, sg_ctl.max_subgroup_size)
        } else {
            (subgroup.subgroup_size, subgroup.subgroup_size)
        };

        Ok(GpuDevice {
            uuid: DeviceUuid(id.device_uuid),
            name: cstr_field(&props.device_name),
            vendor,
            pci_vendor_id: props.vendor_id,
            pci_device_id: props.device_id,
            kind,
            driver_name: if has_driver_props {
                cstr_field(&driver.driver_name)
            } else {
                String::new()
            },
            driver_info: if has_driver_props {
                cstr_field(&driver.driver_info)
            } else {
                String::new()
            },
            api_version: version_string(dev_api),
            vram_bytes: vram,
            unified_memory: matches!(kind, DeviceKind::Integrated | DeviceKind::Cpu)
                || vendor == Vendor::Apple,
            fp16: f16.shader_float16 == vk::TRUE,
            cooperative_matrix: has(vk::KHR_COOPERATIVE_MATRIX_NAME)
                || has(vk::NV_COOPERATIVE_MATRIX_NAME),
            subgroup_min: sg_min,
            subgroup_max: sg_max,
            memory_budget: has(vk::EXT_MEMORY_BUDGET_NAME),
            drm_render_minor: (has_drm && drm.has_render == vk::TRUE)
                .then_some(drm.render_minor as u32),
            score: 0,
        })
    }

    /// The device's Windows adapter LUID (`VkPhysicalDeviceIDProperties`), for
    /// matching it in WDDM and performance-counter telemetry. `None` when the
    /// driver doesn't report one (always, off Windows).
    pub fn luid(&self, uuid: &DeviceUuid) -> Option<[u8; 8]> {
        let (_, pd, _) = self.devices.iter().find(|(u, _, _)| u == uuid)?;
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        // SAFETY: `pd` came from this instance, which is Vulkan 1.1+ (checked
        // during discovery, which queries the same struct).
        unsafe { self.instance.get_physical_device_properties2(*pd, &mut p2) };
        (id.device_luid_valid == vk::TRUE).then_some(id.device_luid)
    }

    /// System-wide VRAM `(used, total)` for a device, via `VK_EXT_memory_budget`.
    ///
    /// `heapBudget` is what *this* process may still allocate, i.e. heap size
    /// minus everyone else's usage. So `size − budget + own usage` estimates
    /// total usage across processes, including the fs-worker that does the
    /// rendering.
    pub fn memory_usage(&self, uuid: &DeviceUuid) -> Option<(u64, u64)> {
        let (_, pd, budget_ext) = self.devices.iter().find(|(u, _, _)| u == uuid)?;
        if !*budget_ext {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut m2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: extension support was checked during discovery.
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(*pd, &mut m2)
        };
        let props = m2.memory_properties;
        let mut used = 0u64;
        let mut total = 0u64;
        for i in 0..props.memory_heap_count as usize {
            if props.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            {
                let size = props.memory_heaps[i].size;
                total += size;
                used += size.saturating_sub(budget.heap_budget[i]) + budget.heap_usage[i];
            }
        }
        Some((used.min(total), total))
    }
}
