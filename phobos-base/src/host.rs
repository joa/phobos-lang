//! What the host machine has to spare.

use anyhow::{Result, ensure};

/// Physical memory the host has, and how much of it the process could still
/// take before the system or its container has to evict something it cannot
/// drop for free. Reclaimable file pages count as available. A container's
/// limit caps both. `None` where the platform does not say.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub fn memory() -> Option<Memory> {
    imp::memory()
}

/// Refuses a budget of `bytes` the host does not have available. Passes
/// where the platform does not say.
pub fn ensure_available(bytes: u64) -> Result<()> {
    let Some(memory) = memory() else {
        return Ok(());
    };
    ensure!(
        bytes <= memory.available_bytes,
        "{} MiB of host memory asked for, but only {} MiB of the host's {} MiB is available",
        bytes >> 20,
        memory.available_bytes >> 20,
        memory.total_bytes >> 20
    );
    Ok(())
}

/// A `MemTotal:` or `MemAvailable:` line of `/proc/meminfo`, in bytes.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn meminfo(meminfo: &str, key: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.strip_prefix(key).is_some_and(|rest| rest.starts_with(':')))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// A cgroup's limit, and the room left under it: the limit less what is
/// charged, not counting the file pages a reclaim would drop. `None` for no
/// limit.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cgroup(limit: &str, charged: &str, stat: &str, file_key: &str) -> Option<Memory> {
    let limit: u64 = limit.trim().parse().ok()?;
    let charged: u64 = charged.trim().parse().ok()?;
    let file = stat
        .lines()
        .find_map(|l| l.strip_prefix(file_key)?.strip_prefix(' ')?.trim().parse::<u64>().ok())
        .unwrap_or(0);
    Some(Memory { total_bytes: limit, available_bytes: limit.saturating_sub(charged.saturating_sub(file)) })
}

#[cfg(target_os = "linux")]
mod imp {
    use std::fs::read_to_string;

    use super::Memory;

    /// A v1 limit this large is the kernel's way of saying none.
    const V1_UNLIMITED: u64 = 1 << 60;

    pub(super) fn memory() -> Option<Memory> {
        let text = read_to_string("/proc/meminfo").ok()?;
        let system = Memory { total_bytes: super::meminfo(&text, "MemTotal")?, available_bytes: super::meminfo(&text, "MemAvailable")? };
        let Some(group) = cgroup_v2().or_else(cgroup_v1).filter(|g| g.total_bytes < V1_UNLIMITED) else {
            return Some(system);
        };
        Some(Memory {
            total_bytes: system.total_bytes.min(group.total_bytes),
            available_bytes: system.available_bytes.min(group.available_bytes),
        })
    }

    /// The cgroup v2 limit at the namespace's root, which inside a container
    /// is the container's own.
    fn cgroup_v2() -> Option<Memory> {
        let dir = "/sys/fs/cgroup";
        let read = |f: &str| read_to_string(format!("{dir}/{f}")).ok();
        super::cgroup(&read("memory.max")?, &read("memory.current")?, &read("memory.stat").unwrap_or_default(), "file")
    }

    fn cgroup_v1() -> Option<Memory> {
        let dir = "/sys/fs/cgroup/memory";
        let read = |f: &str| read_to_string(format!("{dir}/{f}")).ok();
        super::cgroup(
            &read("memory.limit_in_bytes")?,
            &read("memory.usage_in_bytes")?,
            &read("memory.stat").unwrap_or_default(),
            "total_cache",
        )
    }
}

#[cfg(windows)]
mod imp {
    use super::Memory;

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }

    pub(super) fn memory() -> Option<Memory> {
        let mut status = MemoryStatusEx {
            length: size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: the struct is laid out as the call expects, its length set.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
        ok.then_some(Memory { total_bytes: status.total_phys, available_bytes: status.avail_phys })
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod imp {
    pub(super) fn memory() -> Option<super::Memory> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_reads_a_key_in_bytes() {
        let text = "MemTotal:       32768000 kB\nMemFree:         1000000 kB\nMemAvailable:   20000000 kB\n";
        assert_eq!(meminfo(text, "MemAvailable"), Some(20_000_000 * 1024));
        assert_eq!(meminfo(text, "MemTotal"), Some(32_768_000 * 1024));
        assert_eq!(meminfo("MemTotal: 1 kB\n", "MemAvailable"), None);
    }

    #[test]
    fn a_cgroup_counts_its_file_pages_as_room() {
        let stat = "anon 1000\nfile 3000\nfile_mapped 10\n";
        let room = cgroup("10000\n", "5000\n", stat, "file").unwrap();
        assert_eq!((room.total_bytes, room.available_bytes), (10000, 8000));
        // No limit reads as none, a charge past it as nothing left.
        assert!(cgroup("max\n", "5000\n", stat, "file").is_none());
        assert_eq!(cgroup("4000\n", "9000\n", "", "file").unwrap().available_bytes, 0);
    }

    #[test]
    fn this_host_reports_its_memory() {
        if cfg!(any(target_os = "linux", windows)) {
            let memory = memory().expect("a supported platform says");
            assert!(memory.total_bytes > 0 && memory.available_bytes <= memory.total_bytes);
            assert!(ensure_available(u64::MAX).is_err());
            assert!(ensure_available(0).is_ok());
        }
    }
}
