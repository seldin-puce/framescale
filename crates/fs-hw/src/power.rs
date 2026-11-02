//! On-battery detection (Eco Auto mode lowers duty on battery).

/// `Some(true)` on battery, `Some(false)` on AC, `None` if unknown.
pub fn on_battery() -> Option<bool> {
    imp::on_battery()
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn on_battery() -> Option<bool> {
        let dir = std::fs::read_dir("/sys/class/power_supply").ok()?;
        let mut saw_mains = false;
        for e in dir.flatten() {
            let p = e.path();
            let kind = std::fs::read_to_string(p.join("type")).unwrap_or_default();
            if kind.trim() == "Mains" {
                saw_mains = true;
                if std::fs::read_to_string(p.join("online")).is_ok_and(|s| s.trim() == "1") {
                    return Some(false);
                }
            }
        }
        saw_mains.then_some(true)
    }
}

#[cfg(windows)]
mod imp {
    #[allow(unsafe_code)]
    pub fn on_battery() -> Option<bool> {
        use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
        // SAFETY: plain out-parameter struct, zero-initialised, owned by us.
        let mut s: SYSTEM_POWER_STATUS = unsafe { std::mem::zeroed() };
        if unsafe { GetSystemPowerStatus(&mut s) } == 0 {
            return None;
        }
        match s.ACLineStatus {
            0 => Some(true),
            1 => Some(false),
            _ => None,
        }
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod imp {
    // macOS: IOPowerSources via IOKit lands with the CoreML backend work.
    pub fn on_battery() -> Option<bool> {
        None
    }
}
