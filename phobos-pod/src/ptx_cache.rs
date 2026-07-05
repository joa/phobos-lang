use std::collections::HashMap;

use anyhow::{Context, Result};
use cust::module::Module;
use phobos_cluster::tile::NodeId;

struct Entry {
    hash: String,
    name: String,
}

pub struct PtxCache {
    modules: HashMap<String, Module>,
    text: HashMap<String, String>,
    /// kernel-id -> (content hash, entry-point name)
    table: HashMap<u32, Entry>,
    /// content hash -> a peer that holds the PTX
    holders: HashMap<String, NodeId>,
}

impl PtxCache {
    pub fn new() -> PtxCache {
        PtxCache {
            modules: HashMap::new(),
            text: HashMap::new(),
            table: HashMap::new(),
            holders: HashMap::new(),
        }
    }

    pub fn set_table(&mut self, entries: impl IntoIterator<Item = (u32, String, String)>) {
        for (idx, hash, name) in entries {
            self.table.insert(idx, Entry { hash, name });
        }
    }

    pub fn set_holders(&mut self, holders: impl IntoIterator<Item = (String, NodeId)>) {
        self.holders.extend(holders);
    }

    pub fn insert(&mut self, hash: &str, ptx: &str) -> Result<()> {
        if self.modules.contains_key(hash) {
            return Ok(());
        }
        let module = Module::from_ptx(ptx, &[])
            .with_context(|| format!("loading PTX for kernel hash {hash}"))?;
        self.modules.insert(hash.to_string(), module);
        self.text.insert(hash.to_string(), ptx.to_string());
        Ok(())
    }

    pub fn text_of(&self, hash: &str) -> Option<String> {
        self.text.get(hash).cloned()
    }

    pub fn hash_of(&self, kernel: u32) -> Result<&str> {
        Ok(self
            .table
            .get(&kernel)
            .with_context(|| format!("kernel id {kernel} not in table"))?
            .hash
            .as_str())
    }

    /// The peer (if any) that can serve a missing kernel's PTX
    pub fn holder_of(&self, hash: &str) -> Option<NodeId> {
        self.holders.get(hash).copied()
    }

    /// The CUDA function for a given kernel; None if its PTX is not cached.
    /// Call GetKernel first to fill the cache.
    pub fn function(&self, kernel: u32) -> Result<Option<cust::function::Function<'_>>> {
        let entry = self
            .table
            .get(&kernel)
            .with_context(|| format!("kernel id {kernel} not in table"))?;
        match self.modules.get(&entry.hash) {
            Some(m) => Ok(Some(m.get_function(&entry.name)?)),
            None => Ok(None),
        }
    }
}

impl Default for PtxCache {
    fn default() -> Self {
        PtxCache::new()
    }
}
