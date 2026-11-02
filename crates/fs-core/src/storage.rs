//! StorageGuard: refuse to start renders that can't fit, and pause (not fail)
//! when the disk fills mid-render.

use std::path::Path;

pub const SAFETY_MARGIN: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shortfall {
    pub required: u64,
    pub available: u64,
}

/// Peak usage = segments (≈ output) + concatenated video (≈ output) + final
/// mux (≈ output) briefly coexisting, so 3× the estimate, plus a fixed margin.
/// Assumes the temp and output volumes are the same, which is the worst case.
pub fn required_bytes(estimated_output: u64) -> u64 {
    estimated_output
        .saturating_mul(3)
        .saturating_add(SAFETY_MARGIN)
}

pub fn preflight(estimated_output: u64, available: u64) -> Result<(), Shortfall> {
    let required = required_bytes(estimated_output);
    if available >= required {
        Ok(())
    } else {
        Err(Shortfall {
            required,
            available,
        })
    }
}

/// Mid-render check, run after each committed segment.
pub fn should_pause(available: u64, avg_segment_bytes: u64) -> bool {
    available
        < avg_segment_bytes
            .saturating_mul(4)
            .saturating_add(SAFETY_MARGIN / 2)
}

pub fn free_space(path: &Path) -> Option<u64> {
    // Walk up to an existing ancestor (the output file doesn't exist yet).
    let mut p = Some(path);
    while let Some(dir) = p {
        if dir.exists() {
            return fs_hw::disk_free(dir).ok();
        }
        p = dir.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    const GB: u64 = 1 << 30;

    #[test]
    fn preflight_and_pause_thresholds() {
        assert!(preflight(10 * GB, 40 * GB).is_ok());
        assert_eq!(
            preflight(10 * GB, 20 * GB),
            Err(Shortfall {
                required: 32 * GB,
                available: 20 * GB
            })
        );
        assert!(!should_pause(10 * GB, 100 << 20));
        assert!(should_pause(GB, 100 << 20));
    }

    #[test]
    fn free_space_of_nonexistent_file_uses_parent() {
        assert!(free_space(&std::env::temp_dir().join("does/not/exist.mkv")).is_some());
    }
}
