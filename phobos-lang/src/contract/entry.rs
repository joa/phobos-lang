// A contract checked against the PTX it was compiled beside: the `.entry`
// parameter list, the CTA size and the shared memory the module declares.

use super::{Class, Contract};

/// One `.param` of a PTX `.entry`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct EntryParam {
    bytes: u32,
    ptr: bool,
    float: bool,
}

/// What the PTX says about one kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    params: Vec<EntryParam>,
    max_threads: Option<i64>,
    shared_bytes: i64,
}

/// Every mismatch between `contract` and the `.entry` of its kernel in
/// `ptx`, one line each; empty when they agree.
pub fn check_entry(ptx: &str, contract: &Contract) -> Vec<String> {
    let entry = match parse_entry(ptx, &contract.kernel) {
        Ok(entry) => entry,
        Err(e) => return vec![e],
    };
    let mut found = Vec::new();
    if entry.params.len() != contract.slots.len() {
        found.push(format!(
            "{} .entry parameters, {} slots in the contract",
            entry.params.len(),
            contract.slots.len()
        ));
    }
    for (i, (p, slot)) in entry.params.iter().zip(&contract.slots).enumerate() {
        let want = EntryParam {
            bytes: slot.bytes,
            ptr: slot.class == Class::Ptr,
            float: slot.class == Class::Float && slot.bytes >= 4,
        };
        if *p != want {
            let name = &contract.params[slot.param].name;
            found.push(format!("slot {i} ({name} {:?}): .entry has {p:?}, the contract {want:?}", slot.part));
        }
    }
    match entry.max_threads {
        Some(n) if n != contract.cta_threads => {
            found.push(format!(".maxntid {n}, the contract's CTA {}", contract.cta_threads));
        }
        None if contract.cta_threads != crate::ast::DEFAULT_CTA_THREADS => {
            found.push(format!("no .maxntid, the contract's CTA {}", contract.cta_threads));
        }
        _ => {}
    }
    if entry.shared_bytes != contract.static_shared_bytes {
        found.push(format!(
            "{} bytes of .shared declared, the contract's {}",
            entry.shared_bytes, contract.static_shared_bytes
        ));
    }
    if contract.dynamic_shared_bytes > 0 && !ptx.contains(".extern .shared") {
        found.push(format!("{} dynamic shared bytes but no .extern .shared", contract.dynamic_shared_bytes));
    }
    found
}

fn parse_entry(ptx: &str, kernel: &str) -> Result<Entry, String> {
    let head = format!(".entry {kernel}(");
    let at = ptx.find(&head).ok_or_else(|| format!("no `.entry {kernel}` in the PTX"))?;
    let rest = &ptx[at + head.len()..];
    let close = rest.find(')').ok_or("an unterminated .entry parameter list")?;
    let params = rest[..close]
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(parse_param)
        .collect::<Result<Vec<_>, _>>()?;
    let open = rest.find('{').ok_or("an .entry with no body")?;
    let max_threads = rest[close + 1..open]
        .split_whitespace()
        .skip_while(|w| *w != ".maxntid")
        .nth(1)
        .map(|n| n.trim_end_matches(',').parse::<i64>().map_err(|e| format!(".maxntid {n}: {e}")))
        .transpose()?;
    // A buffer only this kernel uses is demoted into its body; one it shares
    // stays at module scope. Both are named for it.
    let prefix = format!("__{kernel}_");
    let shared_bytes = ptx
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with(".shared ") || l.starts_with(".global .shared"))
        .filter_map(|l| {
            let decl = l.trim_end_matches(';');
            let name = decl.split_whitespace().last()?;
            let (name, count) = name.strip_suffix(']')?.split_once('[')?;
            let width = decl.split_whitespace().find_map(|w| match w {
                ".b8" | ".u8" | ".s8" => Some(1),
                ".b16" | ".u16" | ".s16" | ".f16" => Some(2),
                ".b32" | ".u32" | ".s32" | ".f32" => Some(4),
                ".b64" | ".u64" | ".s64" | ".f64" => Some(8),
                _ => None,
            })?;
            name.starts_with(&prefix).then(|| count.parse::<i64>().ok().map(|n| n * width))?
        })
        .sum();
    Ok(Entry { params, max_threads, shared_bytes })
}

fn parse_param(p: &str) -> Result<EntryParam, String> {
    let words: Vec<&str> = p.split_whitespace().collect();
    if words.first() != Some(&".param") {
        return Err(format!("`{p}` is not a .param"));
    }
    // An `f16` is passed as two bytes, `.param .align 2 .b8 NAME[2]`.
    let ty = words.iter().skip(1).find(|w| w.starts_with(".") && !matches!(**w, ".align" | ".ptr" | ".global"));
    let ty = ty.ok_or_else(|| format!("`{p}` has no type"))?;
    let width = match &ty[2..] {
        "8" => 1,
        "16" => 2,
        "32" => 4,
        "64" => 8,
        _ => return Err(format!("`{p}`: unknown .param type {ty}")),
    };
    let count = match words.last().and_then(|n| n.strip_suffix(']')).and_then(|n| n.split_once('[')) {
        Some((_, count)) => count.parse::<u32>().map_err(|e| format!("`{p}`: {e}"))?,
        None => 1,
    };
    Ok(EntryParam { bytes: width * count, ptr: words.contains(&".ptr"), float: ty.starts_with(".f") })
}
