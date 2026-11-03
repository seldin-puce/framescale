//! Vendor-neutral GPU telemetry on Windows: the same sources Task Manager uses.
//!
//! - Load: the "GPU Engine" performance counters (PDH), summed per engine for
//!   the adapter's LUID; the busiest engine is the GPU's load.
//! - Temperature and power: WDDM's adapter performance data
//!   (`D3DKMTQueryAdapterInfo(KMTQAITYPE_ADAPTERPERFDATA)`, Windows 10 2004+
//!   with a WDDM 2.7 driver). Power comes as a share of the board's limit, not
//!   watts. Drivers that don't report a value leave it at 0, shown as "—".
#![allow(unsafe_code)] // Win32 FFI, each call documented below.

use std::collections::HashMap;

use fs_types::DeviceTelemetry;
use windows_sys::Wdk::Graphics::Direct3D::{
    D3DKMT_ADAPTER_PERFDATA, D3DKMT_CLOSEADAPTER, D3DKMT_OPENADAPTERFROMLUID,
    D3DKMT_QUERYADAPTERINFO, D3DKMTCloseAdapter, D3DKMTOpenAdapterFromLuid, D3DKMTQueryAdapterInfo,
    KMTQAITYPE_ADAPTERPERFDATA,
};
use windows_sys::Win32::Foundation::LUID;
use windows_sys::Win32::System::Performance::{
    PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY,
    PDH_MORE_DATA, PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
    PdhGetFormattedCounterArrayW, PdhOpenQueryW,
};

use crate::telemetry::{TelemetrySource, set};

/// Vulkan's 8-byte deviceLUID as the Win32 LUID it encodes.
fn luid(bytes: [u8; 8]) -> LUID {
    LUID {
        LowPart: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        HighPart: i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    }
}

/// The adapter's tag in "GPU Engine" instance names:
/// `pid_1234_luid_0x00000000_0x0000C0C5_phys_0_eng_3_engtype_3D`.
fn luid_tag(l: &LUID) -> String {
    format!("luid_0x{:08X}_0x{:08X}", l.HighPart as u32, l.LowPart)
}

/// Busiest engine for one adapter: instances are per process and engine, so
/// sum each engine across processes, then take the maximum.
fn busiest_engine<'a>(tag: &str, items: impl Iterator<Item = (&'a str, f64)>) -> Option<f32> {
    let mut per_engine: HashMap<&str, f64> = HashMap::new();
    let mut seen = false;
    for (name, v) in items {
        let Some(at) = name.find(tag) else { continue };
        seen = true;
        // Engine key: everything after the LUID ("_phys_0_eng_3_engtype_3D").
        *per_engine.entry(&name[at + tag.len()..]).or_default() += v;
    }
    seen.then(|| {
        per_engine
            .values()
            .copied()
            .fold(0.0, f64::max)
            .clamp(0.0, 100.0) as f32
    })
}

pub(crate) struct WindowsGpuSource {
    tag: String,
    adapter: Option<u32>,
    query: PDH_HQUERY,
    counter: PDH_HCOUNTER,
}

// SAFETY: the PDH query/counter handles and the kernel adapter handle are only
// used through `&mut self`, one thread at a time; Win32 allows that from any
// thread.
unsafe impl Send for WindowsGpuSource {}

impl WindowsGpuSource {
    pub(crate) fn new(luid_bytes: [u8; 8]) -> Option<Self> {
        let luid = luid(luid_bytes);
        let mut open = D3DKMT_OPENADAPTERFROMLUID {
            AdapterLuid: luid,
            hAdapter: 0,
        };
        // SAFETY: `open` is a valid, initialised in/out struct.
        let adapter =
            (unsafe { D3DKMTOpenAdapterFromLuid(&mut open) } == 0).then_some(open.hAdapter);

        let mut query: PDH_HQUERY = std::ptr::null_mut();
        let mut counter: PDH_HCOUNTER = std::ptr::null_mut();
        let path: Vec<u16> = "\\GPU Engine(*)\\Utilization Percentage\0"
            .encode_utf16()
            .collect();
        // SAFETY: out-pointers are valid; `path` is NUL-terminated UTF-16.
        let pdh_ok = unsafe {
            PdhOpenQueryW(std::ptr::null(), 0, &mut query) == 0
                && PdhAddEnglishCounterW(query, path.as_ptr(), 0, &mut counter) == 0
        };
        if pdh_ok {
            // Rate counters need a first sample to compare against.
            // SAFETY: valid query handle.
            unsafe { PdhCollectQueryData(query) };
        } else if !query.is_null() {
            // SAFETY: the query was opened above.
            unsafe { PdhCloseQuery(query) };
            query = std::ptr::null_mut();
        }
        if adapter.is_none() && query.is_null() {
            return None;
        }
        Some(Self {
            tag: luid_tag(&luid),
            adapter,
            query,
            counter,
        })
    }

    fn load(&mut self) -> Option<f32> {
        if self.query.is_null() {
            return None;
        }
        // SAFETY: valid query handle.
        if unsafe { PdhCollectQueryData(self.query) } != 0 {
            return None;
        }
        let (mut size, mut count) = (0u32, 0u32);
        // SAFETY: a null buffer asks for the required size.
        let r = unsafe {
            PdhGetFormattedCounterArrayW(
                self.counter,
                PDH_FMT_DOUBLE,
                &mut size,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if r != PDH_MORE_DATA || size == 0 {
            return None;
        }
        // u64 storage keeps the item array aligned.
        let mut buf = vec![0u64; (size as usize).div_ceil(8)];
        let items = buf.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>();
        // SAFETY: `buf` holds at least `size` bytes, as PDH asked for.
        if unsafe {
            PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, &mut size, &mut count, items)
        } != 0
        {
            return None;
        }
        // SAFETY: PDH filled `count` items; names point into `buf`.
        let items = unsafe { std::slice::from_raw_parts(items, count as usize) };
        let named: Vec<(String, f64)> = items
            .iter()
            .filter(|i| i.FmtValue.CStatus == PDH_CSTATUS_VALID_DATA && !i.szName.is_null())
            .map(|i| {
                // SAFETY: szName is a NUL-terminated UTF-16 string inside `buf`.
                let name = unsafe {
                    let len = (0..).take_while(|&k| *i.szName.add(k) != 0).count();
                    String::from_utf16_lossy(std::slice::from_raw_parts(i.szName, len))
                };
                // SAFETY: PDH_FMT_DOUBLE fills the double member.
                (name, unsafe { i.FmtValue.Anonymous.doubleValue })
            })
            .collect();
        busiest_engine(&self.tag, named.iter().map(|(n, v)| (n.as_str(), *v)))
    }

    /// (temperature °C, power % of the board limit), when the driver reports them.
    fn perf(&self) -> (Option<f32>, Option<f32>) {
        let Some(h) = self.adapter else {
            return (None, None);
        };
        let mut data = D3DKMT_ADAPTER_PERFDATA::default();
        let mut q = D3DKMT_QUERYADAPTERINFO {
            hAdapter: h,
            Type: KMTQAITYPE_ADAPTERPERFDATA,
            pPrivateDriverData: (&raw mut data).cast(),
            PrivateDriverDataSize: size_of::<D3DKMT_ADAPTER_PERFDATA>() as u32,
        };
        // SAFETY: `q` points at a correctly sized, writable perf-data struct.
        if unsafe { D3DKMTQueryAdapterInfo(&mut q) } != 0 {
            return (None, None);
        }
        // Tenths of a degree Celsius / tenths of a percent; 0 = not reported.
        let temp = (data.Temperature > 0).then(|| data.Temperature as f32 / 10.0);
        let power = (data.Power > 0).then(|| (data.Power as f32 / 10.0).min(100.0));
        (temp, power)
    }
}

impl TelemetrySource for WindowsGpuSource {
    fn name(&self) -> &'static str {
        "windows-gpu"
    }
    fn sample(&mut self, t: &mut DeviceTelemetry) {
        set(&mut t.utilization_pct, self.load());
        let (temp, power) = self.perf();
        set(&mut t.temperature_c, temp);
        set(&mut t.power_pct, power);
    }
}

impl Drop for WindowsGpuSource {
    fn drop(&mut self) {
        if !self.query.is_null() {
            // SAFETY: opened in `new`, closed once.
            unsafe { PdhCloseQuery(self.query) };
        }
        if let Some(h) = self.adapter {
            let close = D3DKMT_CLOSEADAPTER { hAdapter: h };
            // SAFETY: opened in `new`, closed once.
            unsafe { D3DKMTCloseAdapter(&close) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luid_tag_matches_counter_names() {
        let l = luid([0xC5, 0xC0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(luid_tag(&l), "luid_0x00000000_0x0000C0C5");
    }

    #[test]
    fn busiest_engine_sums_processes() {
        let tag = "luid_0x00000000_0x0000C0C5";
        let items = [
            (
                "pid_1_luid_0x00000000_0x0000C0C5_phys_0_eng_0_engtype_3D",
                20.0,
            ),
            (
                "pid_2_luid_0x00000000_0x0000C0C5_phys_0_eng_0_engtype_3D",
                30.0,
            ),
            (
                "pid_2_luid_0x00000000_0x0000C0C5_phys_0_eng_3_engtype_Compute",
                45.0,
            ),
            (
                "pid_3_luid_0x00000000_0x0000AAAA_phys_0_eng_0_engtype_3D",
                99.0,
            ),
        ];
        assert_eq!(busiest_engine(tag, items.into_iter()), Some(50.0));
        assert_eq!(
            busiest_engine("luid_0x00000000_0x00000001", items.into_iter()),
            None
        );
    }
}
