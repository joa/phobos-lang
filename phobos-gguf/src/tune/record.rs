//! A [`Tuner`]'s picks on disk, so the next process starts where this one
//! settled. One file per model, card and compiler, under the kernel cache's
//! chip directory, in `tuning/`, which `phobos-cache` leaves alone.
//!
//! One line per depth bucket, `key=value` pairs:
//! `bucket=1 pick=1024 settled=1 lost=512,2048 us=512:130.2,1024:121.0`.
//! The medians are for a reader; a load restores the pick, the losses and
//! whether the bucket settled, and an unsettled bucket measures again.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{Bucket, MIN_BATCH, Tuner};

/// Folded into every record's key. Bumped when a change to the runtime moves
/// what a pass of a given size costs without changing the compiler, so the
/// records measured before it are not read.
const RECORD_VERSION: u32 = 1;

/// Where the record for this model on this card lives, or `None` when the
/// kernel cache is turned off.
pub(crate) fn record_path(header: &[u8], device: &phobos_inference::DeviceInfo) -> Option<PathBuf> {
    let root = phobos_kernels::util::kernel_cache_dir()?;
    let (major, minor) = device.capability;
    let chip = std::env::var("PHOBOS_CHIP")
        .unwrap_or_else(|_| phobos_base::context::best_supported_chip(major * 10 + minor).to_string());
    let mut hash = Sha256::new();
    hash.update(header);
    hash.update(device.name.as_bytes());
    hash.update(device.multiprocessors.to_le_bytes());
    hash.update(phobos_kernels::COMPILER_FINGERPRINT.as_bytes());
    hash.update(RECORD_VERSION.to_le_bytes());
    let digest = hash.finalize();
    let key: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
    Some(root.join(chip).join("tuning").join(format!("prompt_batch_{key}.txt")))
}

impl Tuner {
    pub(crate) fn to_text(&self) -> String {
        let mut out = String::new();
        for (index, bucket) in self.buckets.iter().enumerate() {
            let join = |sizes: &mut dyn Iterator<Item = String>| sizes.collect::<Vec<_>>().join(",");
            let lost = join(&mut bucket.lost.iter().map(usize::to_string));
            let mut us = join(&mut bucket.adjusted().iter().map(|(size, secs)| format!("{size}:{:.2}", secs * 1e6)));
            if us.is_empty() {
                us.clone_from(&bucket.recorded_us);
            }
            out += &format!(
                "bucket={index} pick={} settled={} lost={lost} us={us}\n",
                bucket.pick, bucket.settled as u8
            );
        }
        out
    }

    /// Restores what [`Tuner::to_text`] wrote. A line that does not parse is
    /// skipped, and a size past this tuner's limit is dropped.
    pub(crate) fn load_text(&mut self, text: &str) {
        let limit = self.limit;
        for line in text.lines() {
            let field = |name: &str| {
                line.split_whitespace()
                    .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))
            };
            let size = |text: &str| text.parse::<usize>().ok().filter(|s| (MIN_BATCH..=limit).contains(s));
            let (Some(index), Some(pick)) = (
                field("bucket").and_then(|i| i.parse::<usize>().ok()),
                field("pick").and_then(size),
            ) else {
                continue;
            };
            // A bucket past the deepest a context can reach is a corrupt line.
            if index > 32 {
                continue;
            }
            self.bucket_mut(index);
            let mut bucket = Bucket::new(pick);
            bucket.lost = field("lost").unwrap_or("").split(',').filter_map(size).collect();
            bucket.settled = field("settled") == Some("1");
            bucket.recorded_us = field("us").unwrap_or("").to_string();
            self.buckets[index] = bucket;
        }
    }

    pub(crate) fn load(&mut self, path: &Path) {
        if let Ok(text) = std::fs::read_to_string(path) {
            self.load_text(&text);
        }
    }

    /// Writes the record through a temporary file and a rename, so a reader
    /// never sees half of one.
    pub(crate) fn save(&self, path: &Path) -> std::io::Result<()> {
        let dir = path.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join(format!(".tmp-{}", std::process::id()));
        std::fs::write(&tmp, self.to_text())?;
        std::fs::rename(&tmp, path)
    }
}
