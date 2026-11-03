//! GPU telemetry on Apple Silicon, without root: the sources Activity Monitor,
//! `powermetrics`-free monitors (macmon, Stats) and Xcode's gauges use.
//!
//! - Load: the GPU's IOAccelerator (`AGXAccelerator`) registry entry,
//!   `PerformanceStatistics` → "Device Utilization %".
//! - Power: IOReport's "Energy Model" group, channel "GPU Energy": energy
//!   since the previous sample (mJ/µJ/nJ) over the elapsed time, in watts.
//! - Temperature: the SMC's GPU sensors (`Tg*` keys, 32-bit floats),
//!   averaged; when a chip exposes none, the SoC die sensors ("PMU tdie*")
//!   from the HID event system.
//!
//! IOReport and the HID event-system client are private but stable APIs
//! (unchanged M1–M4); every call is checked, and a missing value stays `None`
//! (shown as "—").
#![allow(unsafe_code)] // CoreFoundation / IOKit / IOReport FFI, documented per call.

use std::ffi::{CStr, c_char, c_void};
use std::time::Instant;

use fs_types::DeviceTelemetry;

use crate::telemetry::{TelemetrySource, set};

// --- CoreFoundation / IOKit FFI ----------------------------------------------

type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFArrayRef = *const c_void;
type CFIndex = isize;
type IoObject = u32;
type KernReturn = i32;

const UTF8: u32 = 0x0800_0100; // kCFStringEncodingUTF8
const CF_NUMBER_DOUBLE: i32 = 13;
const CF_NUMBER_SINT32: i32 = 3;

#[repr(C)]
struct DictCallbacks([usize; 6]);

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFTypeDictionaryKeyCallBacks: DictCallbacks;
    static kCFTypeDictionaryValueCallBacks: DictCallbacks;
    fn CFStringCreateWithCString(alloc: CFTypeRef, s: *const c_char, enc: u32) -> CFStringRef;
    fn CFStringGetCString(s: CFStringRef, buf: *mut c_char, len: CFIndex, enc: u32) -> u8;
    fn CFEqual(a: CFTypeRef, b: CFTypeRef) -> u8;
    fn CFRelease(cf: CFTypeRef);
    fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
    fn CFGetTypeID(cf: CFTypeRef) -> usize;
    fn CFNumberGetTypeID() -> usize;
    fn CFDictionaryGetTypeID() -> usize;
    fn CFNumberGetValue(n: CFTypeRef, kind: i32, out: *mut c_void) -> u8;
    fn CFNumberCreate(alloc: CFTypeRef, kind: i32, value: *const c_void) -> CFTypeRef;
    fn CFDictionaryGetValue(d: CFDictionaryRef, key: CFTypeRef) -> CFTypeRef;
    fn CFDictionaryGetCount(d: CFDictionaryRef) -> CFIndex;
    fn CFDictionaryCreate(
        alloc: CFTypeRef,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        n: CFIndex,
        kcb: *const DictCallbacks,
        vcb: *const DictCallbacks,
    ) -> CFDictionaryRef;
    fn CFDictionaryCreateMutableCopy(
        alloc: CFTypeRef,
        cap: CFIndex,
        d: CFDictionaryRef,
    ) -> CFMutableDictionaryRef;
    fn CFArrayGetCount(a: CFArrayRef) -> CFIndex;
    fn CFArrayGetValueAtIndex(a: CFArrayRef, i: CFIndex) -> CFTypeRef;
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> CFMutableDictionaryRef;
    fn IOServiceGetMatchingServices(main: u32, m: CFDictionaryRef, it: *mut IoObject)
    -> KernReturn;
    fn IOServiceGetMatchingService(main: u32, m: CFDictionaryRef) -> IoObject;
    fn IOIteratorNext(it: IoObject) -> IoObject;
    fn IOObjectRelease(o: IoObject) -> KernReturn;
    fn IORegistryEntryCreateCFProperty(
        e: IoObject,
        key: CFStringRef,
        alloc: CFTypeRef,
        opts: u32,
    ) -> CFTypeRef;
    fn IOServiceOpen(s: IoObject, task: u32, kind: u32, conn: *mut IoObject) -> KernReturn;
    fn IOServiceClose(conn: IoObject) -> KernReturn;
    fn IOConnectCallStructMethod(
        conn: IoObject,
        selector: u32,
        input: *const c_void,
        input_size: usize,
        output: *mut c_void,
        output_size: *mut usize,
    ) -> KernReturn;
    // HID event system (private, exported by IOKit).
    fn IOHIDEventSystemClientCreate(alloc: CFTypeRef) -> CFTypeRef;
    fn IOHIDEventSystemClientSetMatching(client: CFTypeRef, m: CFDictionaryRef) -> i32;
    fn IOHIDEventSystemClientCopyServices(client: CFTypeRef) -> CFArrayRef;
    fn IOHIDServiceClientCopyProperty(svc: CFTypeRef, key: CFStringRef) -> CFTypeRef;
    fn IOHIDServiceClientCopyEvent(svc: CFTypeRef, kind: i64, opts: i32, ts: i64) -> CFTypeRef;
    fn IOHIDEventGetFloatValue(event: CFTypeRef, field: i32) -> f64;
}

#[link(name = "IOReport", kind = "dylib")]
unsafe extern "C" {
    fn IOReportCopyChannelsInGroup(
        group: CFStringRef,
        subgroup: CFStringRef,
        a: u64,
        b: u64,
        c: u64,
    ) -> CFDictionaryRef;
    fn IOReportCreateSubscription(
        a: CFTypeRef,
        channels: CFMutableDictionaryRef,
        subscribed: *mut CFMutableDictionaryRef,
        b: u64,
        c: CFTypeRef,
    ) -> CFTypeRef;
    fn IOReportCreateSamples(
        sub: CFTypeRef,
        channels: CFMutableDictionaryRef,
        a: CFTypeRef,
    ) -> CFDictionaryRef;
    fn IOReportCreateSamplesDelta(
        a: CFDictionaryRef,
        b: CFDictionaryRef,
        c: CFTypeRef,
    ) -> CFDictionaryRef;
    fn IOReportChannelGetChannelName(item: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetUnitLabel(item: CFDictionaryRef) -> CFStringRef;
    fn IOReportSimpleGetIntegerValue(item: CFDictionaryRef, index: i32) -> i64;
}

unsafe extern "C" {
    static mach_task_self_: u32;
}

/// An owned CFString (released on drop).
struct CfStr(CFStringRef);

impl CfStr {
    fn new(s: &CStr) -> Option<Self> {
        // SAFETY: NUL-terminated UTF-8 in; returns a +1 reference or NULL.
        let r = unsafe { CFStringCreateWithCString(std::ptr::null(), s.as_ptr(), UTF8) };
        (!r.is_null()).then_some(Self(r))
    }
}

impl Drop for CfStr {
    fn drop(&mut self) {
        // SAFETY: we own this +1 reference.
        unsafe { CFRelease(self.0) };
    }
}

/// A CFString's contents (`s` is borrowed, may be NULL).
fn cf_string(s: CFStringRef) -> Option<String> {
    if s.is_null() {
        return None;
    }
    let mut buf = [0 as c_char; 128];
    // SAFETY: valid CFString, buffer and its length.
    let ok = unsafe { CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as CFIndex, UTF8) };
    // SAFETY: on success the buffer holds a NUL-terminated string.
    (ok != 0).then(|| {
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    })
}

/// A CFNumber as f64 (`n` is borrowed, may be NULL or another type).
fn cf_number(n: CFTypeRef) -> Option<f64> {
    // SAFETY: type-checked before reading; out-pointer is a local.
    unsafe {
        if n.is_null() || CFGetTypeID(n) != CFNumberGetTypeID() {
            return None;
        }
        let mut v = 0f64;
        (CFNumberGetValue(n, CF_NUMBER_DOUBLE, (&raw mut v).cast()) != 0).then_some(v)
    }
}

// --- Load: IOAccelerator performance statistics ------------------------------

struct Accelerator {
    entry: IoObject,
    stats: CfStr,
    util: CfStr,
}

impl Accelerator {
    fn find() -> Option<Self> {
        let stats = CfStr::new(c"PerformanceStatistics")?;
        let util = CfStr::new(c"Device Utilization %")?;
        let mut it = 0;
        // SAFETY: IOServiceMatching returns a dictionary that
        // IOServiceGetMatchingServices consumes; the iterator is released below.
        unsafe {
            let m = IOServiceMatching(c"IOAccelerator".as_ptr());
            if m.is_null() || IOServiceGetMatchingServices(0, m, &mut it) != 0 {
                return None;
            }
        }
        let mut found = None;
        loop {
            // SAFETY: valid iterator; each returned entry is ours to release.
            let e = unsafe { IOIteratorNext(it) };
            if e == 0 {
                break;
            }
            if utilization(e, &stats, &util).is_some() {
                found = Some(e);
                break;
            }
            // SAFETY: entry from IOIteratorNext, not kept.
            unsafe { IOObjectRelease(e) };
        }
        // SAFETY: iterator from IOServiceGetMatchingServices.
        unsafe { IOObjectRelease(it) };
        found.map(|entry| Self { entry, stats, util })
    }

    fn read(&self) -> Option<f32> {
        utilization(self.entry, &self.stats, &self.util)
    }
}

/// `entry`'s PerformanceStatistics → "Device Utilization %".
fn utilization(entry: IoObject, stats: &CfStr, util: &CfStr) -> Option<f32> {
    // SAFETY: valid entry and key; the property is a +1 reference (or NULL),
    // type-checked before use and released.
    unsafe {
        let d = IORegistryEntryCreateCFProperty(entry, stats.0, std::ptr::null(), 0);
        if d.is_null() {
            return None;
        }
        let v = if CFGetTypeID(d) == CFDictionaryGetTypeID() {
            cf_number(CFDictionaryGetValue(d, util.0))
        } else {
            None
        };
        CFRelease(d);
        v.map(|v| v.clamp(0.0, 100.0) as f32)
    }
}

impl Drop for Accelerator {
    fn drop(&mut self) {
        // SAFETY: entry kept from IOIteratorNext.
        unsafe { IOObjectRelease(self.entry) };
    }
}

// --- Power: IOReport "Energy Model" ------------------------------------------

struct Energy {
    sub: CFTypeRef,
    channels: CFMutableDictionaryRef,
    gpu: CfStr,
    key: CfStr,
    prev: Option<(CFDictionaryRef, Instant)>,
}

impl Energy {
    fn new() -> Option<Self> {
        let group = CfStr::new(c"Energy Model")?;
        let gpu = CfStr::new(c"GPU Energy")?;
        let key = CfStr::new(c"IOReportChannels")?;
        // SAFETY: documented IOReport sequence (as used by macmon): copy the
        // group's channels, make a mutable copy, subscribe. Every +1
        // reference is released here or in Drop.
        unsafe {
            let chans = IOReportCopyChannelsInGroup(group.0, std::ptr::null(), 0, 0, 0);
            if chans.is_null() {
                return None;
            }
            let all =
                CFDictionaryCreateMutableCopy(std::ptr::null(), CFDictionaryGetCount(chans), chans);
            CFRelease(chans);
            if all.is_null() {
                return None;
            }
            let mut subscribed: CFMutableDictionaryRef = std::ptr::null_mut();
            let sub = IOReportCreateSubscription(
                std::ptr::null(),
                all,
                &mut subscribed,
                0,
                std::ptr::null(),
            );
            CFRelease(all);
            if sub.is_null() || subscribed.is_null() {
                if !sub.is_null() {
                    CFRelease(sub);
                }
                return None;
            }
            Some(Self {
                sub,
                channels: subscribed,
                gpu,
                key,
                prev: None,
            })
        }
    }

    /// GPU watts since the previous call (None on the first).
    fn read(&mut self) -> Option<f32> {
        // SAFETY: valid subscription; samples are +1 references kept in
        // `prev` or released.
        let now = unsafe { IOReportCreateSamples(self.sub, self.channels, std::ptr::null()) };
        if now.is_null() {
            return None;
        }
        let at = Instant::now();
        let watts = self.prev.take().and_then(|(prev, t)| {
            let dt = at.duration_since(t).as_secs_f64();
            // SAFETY: both samples valid; the delta is released below.
            let delta = unsafe { IOReportCreateSamplesDelta(prev, now, std::ptr::null()) };
            // SAFETY: `prev` was ours.
            unsafe { CFRelease(prev) };
            if delta.is_null() || dt <= 0.0 {
                return None;
            }
            let w = self.gpu_joules(delta).map(|j| (j / dt) as f32);
            // SAFETY: ours.
            unsafe { CFRelease(delta) };
            w
        });
        self.prev = Some((now, at));
        watts
    }

    fn gpu_joules(&self, delta: CFDictionaryRef) -> Option<f64> {
        // SAFETY: the delta's "IOReportChannels" array and its items are
        // borrowed from `delta`, which outlives this call.
        unsafe {
            let arr = CFDictionaryGetValue(delta, self.key.0);
            if arr.is_null() {
                return None;
            }
            for i in 0..CFArrayGetCount(arr) {
                let item = CFArrayGetValueAtIndex(arr, i);
                let name = IOReportChannelGetChannelName(item);
                if name.is_null() || CFEqual(name, self.gpu.0) == 0 {
                    continue;
                }
                let scale = match cf_string(IOReportChannelGetUnitLabel(item))?.trim() {
                    "mJ" => 1e-3,
                    "uJ" | "µJ" => 1e-6,
                    "nJ" => 1e-9,
                    _ => return None,
                };
                return Some(IOReportSimpleGetIntegerValue(item, 0) as f64 * scale);
            }
            None
        }
    }
}

impl Drop for Energy {
    fn drop(&mut self) {
        // SAFETY: all ours.
        unsafe {
            if let Some((p, _)) = self.prev.take() {
                CFRelease(p);
            }
            CFRelease(self.channels);
            CFRelease(self.sub);
        }
    }
}

// --- Temperature: SMC `Tg*` keys, HID die sensors as a fallback ---------------

/// `SMCKeyData_t` from AppleSMC's user client (80 bytes).
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SmcKeyData {
    key: u32,
    vers: [u8; 6],
    p_limit: [u32; 4],
    data_size: u32,
    data_type: u32,
    data_attributes: u8,
    /// `keyInfo` is a 12-byte struct in C.
    _pad: [u8; 3],
    result: u8,
    status: u8,
    data8: u8,
    data32: u32,
    bytes: [u8; 32],
}
const _: () = assert!(std::mem::size_of::<SmcKeyData>() == 80);

const SMC_HANDLE_EVENT: u32 = 2;
const SMC_READ_KEY: u8 = 5;
const SMC_KEY_AT_INDEX: u8 = 8;
const SMC_KEY_INFO: u8 = 9;

fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

struct Smc {
    conn: IoObject,
    /// GPU sensors: every `Tg*` key holding a 32-bit float.
    gpu: Vec<u32>,
}

impl Smc {
    fn open() -> Option<Self> {
        // SAFETY: the matching dictionary is consumed by the call; the
        // service is released after opening a connection to it.
        let conn = unsafe {
            let svc = IOServiceGetMatchingService(0, IOServiceMatching(c"AppleSMC".as_ptr()));
            if svc == 0 {
                return None;
            }
            let mut conn = 0;
            let r = IOServiceOpen(svc, mach_task_self_, 0, &mut conn);
            IOObjectRelease(svc);
            if r != 0 {
                return None;
            }
            conn
        };
        let mut smc = Self {
            conn,
            gpu: Vec::new(),
        };
        let count = smc
            .read(fourcc(b"#KEY"), 4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))?;
        let flt = fourcc(b"flt ");
        for i in 0..count.min(4096) {
            let Some(key) = smc.key_at(i) else { continue };
            if key >> 16 != u32::from(u16::from_be_bytes(*b"Tg")) {
                continue;
            }
            if smc.info(key) == Some((4, flt)) {
                smc.gpu.push(key);
            }
        }
        Some(smc)
    }

    fn call(&self, input: &SmcKeyData) -> Option<SmcKeyData> {
        let mut out = SmcKeyData::default();
        let mut size = std::mem::size_of::<SmcKeyData>();
        // SAFETY: both buffers are 80-byte SMCKeyData_t, sizes passed.
        let r = unsafe {
            IOConnectCallStructMethod(
                self.conn,
                SMC_HANDLE_EVENT,
                (&raw const *input).cast(),
                std::mem::size_of::<SmcKeyData>(),
                (&raw mut out).cast(),
                &mut size,
            )
        };
        (r == 0 && out.result == 0).then_some(out)
    }

    fn key_at(&self, index: u32) -> Option<u32> {
        self.call(&SmcKeyData {
            data8: SMC_KEY_AT_INDEX,
            data32: index,
            ..Default::default()
        })
        .map(|o| o.key)
    }

    fn info(&self, key: u32) -> Option<(u32, u32)> {
        self.call(&SmcKeyData {
            key,
            data8: SMC_KEY_INFO,
            ..Default::default()
        })
        .map(|o| (o.data_size, o.data_type))
    }

    fn read(&self, key: u32, size: u32) -> Option<[u8; 32]> {
        self.call(&SmcKeyData {
            key,
            data_size: size,
            data8: SMC_READ_KEY,
            ..Default::default()
        })
        .map(|o| o.bytes)
    }

    /// Mean of the GPU sensors that report a plausible value.
    fn gpu_temp(&self) -> Option<f32> {
        let vals: Vec<f32> = self
            .gpu
            .iter()
            .filter_map(|k| self.read(*k, 4))
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .filter(|t| (5.0..130.0).contains(t))
            .collect();
        (!vals.is_empty()).then(|| vals.iter().sum::<f32>() / vals.len() as f32)
    }
}

impl Drop for Smc {
    fn drop(&mut self) {
        // SAFETY: connection from IOServiceOpen.
        unsafe { IOServiceClose(self.conn) };
    }
}

/// SoC die temperature sensors from the HID event system ("PMU tdie*"; GPU
/// sensors where a chip names them).
struct HidTemps {
    client: CFTypeRef,
    sensors: Vec<CFTypeRef>,
}

impl HidTemps {
    fn new() -> Option<Self> {
        const TEMPERATURE_EVENT: i64 = 15;
        let page_key = CfStr::new(c"PrimaryUsagePage")?;
        let usage_key = CfStr::new(c"PrimaryUsage")?;
        let product = CfStr::new(c"Product")?;
        // SAFETY: Apple's vendor page 0xff00, usage 5 = temperature sensors.
        // Created objects are released; kept services are retained by the
        // client's services array, which we keep alive via `client`.
        unsafe {
            let (page, usage) = (0xff00i32, 5i32);
            let vals = [
                CFNumberCreate(std::ptr::null(), CF_NUMBER_SINT32, (&raw const page).cast()),
                CFNumberCreate(
                    std::ptr::null(),
                    CF_NUMBER_SINT32,
                    (&raw const usage).cast(),
                ),
            ];
            let keys = [page_key.0, usage_key.0];
            let matching = CFDictionaryCreate(
                std::ptr::null(),
                keys.as_ptr(),
                vals.as_ptr(),
                2,
                &raw const kCFTypeDictionaryKeyCallBacks,
                &raw const kCFTypeDictionaryValueCallBacks,
            );
            for v in vals {
                if !v.is_null() {
                    CFRelease(v);
                }
            }
            if matching.is_null() {
                return None;
            }
            let client = IOHIDEventSystemClientCreate(std::ptr::null());
            if client.is_null() {
                CFRelease(matching);
                return None;
            }
            IOHIDEventSystemClientSetMatching(client, matching);
            CFRelease(matching);
            let services = IOHIDEventSystemClientCopyServices(client);
            if services.is_null() {
                CFRelease(client);
                return None;
            }
            let mut gpu = Vec::new();
            let mut die = Vec::new();
            for i in 0..CFArrayGetCount(services) {
                let svc = CFArrayGetValueAtIndex(services, i);
                let name = IOHIDServiceClientCopyProperty(svc, product.0);
                let n = cf_string(name).unwrap_or_default();
                if !name.is_null() {
                    CFRelease(name);
                }
                // Only sensors that answer a temperature query.
                let keep = || {
                    let ev = IOHIDServiceClientCopyEvent(svc, TEMPERATURE_EVENT, 0, 0);
                    if ev.is_null() {
                        return false;
                    }
                    CFRelease(ev);
                    true
                };
                if n.contains("GPU") && keep() {
                    gpu.push(svc);
                } else if n.starts_with("PMU tdie") && keep() {
                    die.push(svc);
                }
            }
            let sensors = if gpu.is_empty() { die } else { gpu };
            // Keep the services alive past the array.
            for s in &sensors {
                CFRetain(*s);
            }
            CFRelease(services);
            if sensors.is_empty() {
                CFRelease(client);
                return None;
            }
            Some(Self { client, sensors })
        }
    }

    fn read(&self) -> Option<f32> {
        const TEMPERATURE_EVENT: i64 = 15;
        const TEMPERATURE_LEVEL: i32 = 15 << 16;
        let vals: Vec<f32> = self
            .sensors
            .iter()
            .filter_map(|s| {
                // SAFETY: retained service; the event is +1 and released.
                unsafe {
                    let ev = IOHIDServiceClientCopyEvent(*s, TEMPERATURE_EVENT, 0, 0);
                    if ev.is_null() {
                        return None;
                    }
                    let t = IOHIDEventGetFloatValue(ev, TEMPERATURE_LEVEL) as f32;
                    CFRelease(ev);
                    Some(t)
                }
            })
            .filter(|t| (5.0..130.0).contains(t))
            .collect();
        (!vals.is_empty()).then(|| vals.iter().sum::<f32>() / vals.len() as f32)
    }
}

impl Drop for HidTemps {
    fn drop(&mut self) {
        // SAFETY: retained in `new`.
        unsafe {
            for s in &self.sensors {
                CFRelease(*s);
            }
            CFRelease(self.client);
        }
    }
}

// --- The source --------------------------------------------------------------

pub struct AppleGpuSource {
    accel: Option<Accelerator>,
    energy: Option<Energy>,
    smc: Option<Smc>,
    hid: Option<HidTemps>,
}

// SAFETY: the CF/IOKit handles are only used from the sampler's thread, one
// call at a time (`&mut self`); moving them between threads is allowed.
unsafe impl Send for AppleGpuSource {}

impl AppleGpuSource {
    pub fn new() -> Option<Self> {
        let accel = Accelerator::find();
        let energy = Energy::new();
        let smc = Smc::open().filter(|s| !s.gpu.is_empty());
        let hid = if smc.is_none() { HidTemps::new() } else { None };
        tracing::info!(
            load = accel.is_some(),
            power = energy.is_some(),
            smc_gpu_sensors = smc.as_ref().map_or(0, |s| s.gpu.len()),
            hid_sensors = hid.as_ref().map_or(0, |h| h.sensors.len()),
            "macOS GPU telemetry"
        );
        (accel.is_some() || energy.is_some() || smc.is_some() || hid.is_some()).then_some(Self {
            accel,
            energy,
            smc,
            hid,
        })
    }
}

impl TelemetrySource for AppleGpuSource {
    fn name(&self) -> &'static str {
        "apple-gpu"
    }

    fn sample(&mut self, t: &mut DeviceTelemetry) {
        set(
            &mut t.utilization_pct,
            self.accel.as_ref().and_then(Accelerator::read),
        );
        set(&mut t.power_w, self.energy.as_mut().and_then(Energy::read));
        let temp = self
            .smc
            .as_ref()
            .and_then(Smc::gpu_temp)
            .or_else(|| self.hid.as_ref().and_then(HidTemps::read));
        set(&mut t.temperature_c, temp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smc_layout() {
        assert_eq!(fourcc(b"#KEY"), 0x234B_4559);
        assert_eq!(std::mem::offset_of!(SmcKeyData, data_size), 28);
        assert_eq!(std::mem::offset_of!(SmcKeyData, result), 40);
        assert_eq!(std::mem::offset_of!(SmcKeyData, data8), 42);
        assert_eq!(std::mem::offset_of!(SmcKeyData, data32), 44);
        assert_eq!(std::mem::offset_of!(SmcKeyData, bytes), 48);
    }
}
