// What phobos-lang.dev needs from the compiler:
//
//     cargo run -p phobos-lang --features trace --example site -- traces
//     cargo run -p phobos-lang --features trace --example site -- check
//     cargo run -p phobos-lang --features trace --example site -- attributes
//
// `traces` traces every kernel `website/kernels.json` lists and every docs
// fence with a `trace` meta into `website/src/data/traces/<name>.<chip>.json`.
// `check` compiles every `phobos` fence in the docs not marked `fragment`,
// expects the ones marked `expect-error` to fail and records their messages
// in `website/src/data/expected-errors.json`, keyed `<page>#<n>`: the page's
// repo-relative path and the fence's index, from 0, among that page's
// `phobos` fences, fragments included. A fence with `diagnostic=NAME` also
// has what `phobos-compile` prints for it, as a file `NAME.ph`, written to
// `website/src/data/diagnostics/NAME.txt`: its errors, or with `races`, what
// `--check-races` finds in a fence that compiles. `attributes` writes the
// attributes the compiler accepts to `website/src/data/attributes.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow, bail};
use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};
use phobos_lang::diag::{Colors, Diagnostics, SourceFile};
use phobos_lang::race;
use phobos_lang::trace::{self, Origin, Request, Variant};
use serde_json::Value;

const DEFAULT_CHIP: &str = "sm_75";

fn main() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("phobos-lang sits in the workspace")
        .to_path_buf();
    match std::env::args().nth(1).as_deref() {
        Some("traces") => traces(&root),
        Some("check") => check(&root),
        Some("attributes") => attributes(&root),
        _ => bail!("usage: site traces | check | attributes"),
    }
}

/// A `phobos` fence in a docs page.
struct Fence {
    /// Repo-relative path of the page, with forward slashes.
    page: String,
    /// The fence's position among the page's `phobos` fences, from 0.
    index: usize,
    /// The line of the opening fence, from 1.
    line: usize,
    meta: BTreeMap<String, String>,
    text: String,
}

impl Fence {
    fn flag(&self, name: &str) -> bool {
        self.meta.contains_key(name)
    }

    fn at(&self) -> String {
        format!("{}:{}", self.page, self.line)
    }
}

fn traces(root: &Path) -> Result<()> {
    let commit = commit(root);
    let out_dir = root.join("website/src/data/traces");
    std::fs::create_dir_all(&out_dir)?;
    let mut requests = Vec::new();

    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("website/kernels.json"))?)?;
    let entries = manifest["kernels"]
        .as_array()
        .ok_or_else(|| anyhow!("kernels.json: no `kernels` list"))?;
    for entry in entries {
        let field = |k: &str| {
            entry[k]
                .as_str()
                .ok_or_else(|| anyhow!("kernels.json: an entry has no `{k}`: {entry}"))
        };
        let path = field("path")?.to_string();
        let variant = entry.get("variant").map(|v| Variant {
            add: v["add"].as_str().map(str::to_string),
            drop: v["drop"].as_str().map(str::to_string),
        });
        let chips: Vec<String> = match entry["chips"].as_array() {
            Some(chips) => chips
                .iter()
                .filter_map(|c| c.as_str().map(str::to_string))
                .collect(),
            None => vec![DEFAULT_CHIP.to_string()],
        };
        for chip in chips {
            requests.push(Request {
                name: field("name")?.to_string(),
                origin: Origin::File,
                text: std::fs::read_to_string(root.join(&path)).with_context(|| path.clone())?,
                path: path.clone(),
                fence: None,
                variant: variant.clone(),
                chip,
                params: ints(&entry["params"])?,
                dims: ints(&entry["dims"])?,
                args: ints(&entry["args"])?,
                grid: grid(&entry["grid"])?,
                commit: commit.clone(),
            });
        }
    }

    for fence in fences(root)? {
        let Some(name) = fence.meta.get("trace") else {
            continue;
        };
        let pairs_of = |key: &str| -> Result<BTreeMap<String, i64>> {
            fence
                .meta
                .get(key)
                .map_or(Ok(BTreeMap::new()), |v| pairs(v))
                .with_context(|| fence.at())
        };
        let Some(g) = fence.meta.get("grid") else {
            bail!("{}: a traced fence needs `grid=`", fence.at());
        };
        let g: Vec<i64> = g
            .split(',')
            .map(|s| s.trim().parse())
            .collect::<Result<_, _>>()
            .with_context(|| fence.at())?;
        requests.push(Request {
            name: name.clone(),
            origin: Origin::Fence,
            path: fence.page.clone(),
            fence: Some(name.clone()),
            text: fence.text.clone(),
            variant: None,
            chip: fence
                .meta
                .get("chip")
                .cloned()
                .unwrap_or_else(|| DEFAULT_CHIP.to_string()),
            params: pairs_of("params")?,
            dims: pairs_of("dims")?,
            args: pairs_of("args")?,
            grid: extend(&g),
            commit: commit.clone(),
        });
    }

    for req in &requests {
        let t =
            trace::trace(req).with_context(|| format!("tracing {} for {}", req.name, req.chip))?;
        let file = out_dir.join(format!("{}.{}.json", req.name, req.chip));
        let json = serde_json::to_string(&t)?;
        std::fs::write(&file, &json)?;
        println!(
            "{}: {} KB, {} steps",
            rel(root, &file),
            json.len() / 1024,
            t.steps.len()
        );
    }
    Ok(())
}

fn check(root: &Path) -> Result<()> {
    let mut failures = Vec::new();
    let mut expected = BTreeMap::new();
    let mut compiled = 0;
    let diagnostics_dir = root.join("website/src/data/diagnostics");
    let mut rendered = 0;
    for fence in fences(root)? {
        if fence.flag("fragment") {
            continue;
        }
        compiled += 1;
        let chip = fence.meta.get("chip").map_or(DEFAULT_CHIP, String::as_str);
        let context = Context {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
            ..Context::default()
        };
        let outcome = phobos_lang::compile_shared(&context, &fence.text);
        if let Some(name) = fence.meta.get("diagnostic") {
            match diagnostic(&context, &fence, name, outcome.as_ref().err()) {
                Ok(text) => {
                    std::fs::create_dir_all(&diagnostics_dir)?;
                    std::fs::write(diagnostics_dir.join(format!("{name}.txt")), text)?;
                    rendered += 1;
                }
                Err(e) => failures.push(format!("{}: {e:#}", fence.at())),
            }
        }
        match (outcome, fence.flag("expect-error")) {
            (Ok(_), false) => {}
            (Ok(_), true) => failures.push(format!(
                "{}: expected an error, but the fence compiles",
                fence.at()
            )),
            (Err(e), true) => {
                expected.insert(
                    format!("{}#{}", fence.page, fence.index),
                    Value::String(message(&e)),
                );
            }
            (Err(e), false) => failures.push(format!(
                "{}:{}: {}",
                fence.page,
                source_line(&fence, &e),
                message(&e)
            )),
        }
    }
    let file = root.join("website/src/data/expected-errors.json");
    std::fs::write(&file, serde_json::to_string_pretty(&expected)? + "\n")?;
    println!(
        "{compiled} fences compiled, {} errors recorded in {}",
        expected.len(),
        rel(root, &file)
    );
    if rendered > 0 {
        println!("{rendered} diagnostics rendered in {}", rel(root, &diagnostics_dir));
    }
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("{f}");
        }
        bail!("{} fence(s) failed", failures.len());
    }
    Ok(())
}

fn attributes(root: &Path) -> Result<()> {
    let file = root.join("website/src/data/attributes.json");
    std::fs::write(
        &file,
        serde_json::to_string_pretty(&phobos_lang::ast::ATTRIBUTES)? + "\n",
    )?;
    println!("{}", rel(root, &file));
    Ok(())
}

/// What `phobos-compile` prints for `fence` as `NAME.ph`, uncolored: the
/// error `failed` with, or with `races`, what the race check finds.
fn diagnostic(
    context: &Context,
    fence: &Fence,
    name: &str,
    failed: Option<&anyhow::Error>,
) -> Result<String> {
    let file = SourceFile::new(format!("{name}.ph"), fence.text.clone());
    if fence.flag("races") {
        if let Some(e) = failed {
            bail!("a `races` fence has to compile: {}", message(e));
        }
        let kernels = phobos_lang::parse_source(&file)?;
        let reports = race::check_kernels(context, &kernels, &race::Config::default())?;
        let found = race::diagnostics(&kernels, &reports).context("the race check found nothing")?;
        return Ok(found.render(Colors::Off));
    }
    let e = failed.context("a `diagnostic` fence fails to compile, or is marked `races`")?;
    let ds = e
        .downcast_ref::<Diagnostics>()
        .with_context(|| format!("the error is not a diagnostic: {e:#}"))?;
    Ok(Diagnostics::new(file, ds.list().to_vec()).render(Colors::Off))
}

/// What an error says, without where: the figures show the message alone.
fn message(e: &anyhow::Error) -> String {
    match e.downcast_ref::<Diagnostics>() {
        Some(ds) => ds.list().iter().map(|d| d.message.as_str()).collect::<Vec<_>>().join("\n"),
        None => format!("{e:#}"),
    }
}

/// The page line an error points at: the line in the fence it is placed
/// on, else the opening fence.
fn source_line(fence: &Fence, e: &anyhow::Error) -> usize {
    let at = e
        .downcast_ref::<Diagnostics>()
        .and_then(|ds| Some(ds.source().line_of(ds.list().first()?.span()?.lo)));
    at.map_or(fence.line, |n| fence.line + n as usize)
}

/// Every `phobos` fence under the docs, pages in path order.
fn fences(root: &Path) -> Result<Vec<Fence>> {
    let mut pages = Vec::new();
    collect_pages(&root.join("website/src/content/docs"), &mut pages)?;
    pages.sort();
    let mut out = Vec::new();
    for page in pages {
        let text = std::fs::read_to_string(&page)?.replace("\r\n", "\n");
        let page = rel(root, &page);
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        let mut index = 0;
        while i < lines.len() {
            let Some((ticks, info)) = opening(lines[i]) else {
                i += 1;
                continue;
            };
            let indent = lines[i].len() - lines[i].trim_start().len();
            let Some(close) = (i + 1..lines.len()).find(|&j| lines[j].trim() == ticks) else {
                bail!("{page}:{}: a fence that never closes", i + 1);
            };
            if let Some(rest) = info.strip_prefix("phobos")
                && (rest.is_empty() || rest.starts_with(char::is_whitespace))
            {
                let body: Vec<&str> = lines[i + 1..close]
                    .iter()
                    .map(|l| l.get(indent..).unwrap_or(l.trim_start()))
                    .collect();
                out.push(Fence {
                    page: page.clone(),
                    index,
                    line: i + 1,
                    meta: meta(rest),
                    text: body.join("\n") + "\n",
                });
                index += 1;
            }
            i = close + 1;
        }
    }
    Ok(out)
}

fn collect_pages(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_pages(&path, out)?;
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("md" | "mdx")
        ) {
            out.push(path);
        }
    }
    Ok(())
}

/// The backtick run and the info string of an opening fence line.
fn opening(line: &str) -> Option<(&str, &str)> {
    let t = line.trim_start();
    let n = t.chars().take_while(|&c| c == '`').count();
    (n >= 3).then(|| (&t[..n], t[n..].trim()))
}

/// `key="a b" flag key=v` as a map, a bare word mapping to "".
fn meta(info: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut rest = info.trim();
    while !rest.is_empty() {
        let end = rest
            .find(|c: char| c == '=' || c.is_whitespace())
            .unwrap_or(rest.len());
        let key = rest[..end].to_string();
        rest = &rest[end..];
        let value = match rest.strip_prefix('=') {
            Some(after) => match after.strip_prefix('"') {
                Some(quoted) => {
                    let close = quoted.find('"').unwrap_or(quoted.len());
                    rest = quoted.get(close + 1..).unwrap_or("");
                    quoted[..close].to_string()
                }
                None => {
                    let close = after.find(char::is_whitespace).unwrap_or(after.len());
                    rest = &after[close..];
                    after[..close].to_string()
                }
            },
            None => String::new(),
        };
        out.insert(key, value);
        rest = rest.trim_start();
    }
    out
}

/// `"K=256, N=64"` as a map.
fn pairs(text: &str) -> Result<BTreeMap<String, i64>> {
    text.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let (k, v) = p
                .split_once('=')
                .ok_or_else(|| anyhow!("`{p}` is not NAME=VALUE"))?;
            Ok((k.trim().to_string(), v.trim().parse()?))
        })
        .collect()
}

fn ints(v: &Value) -> Result<BTreeMap<String, i64>> {
    let Some(obj) = v.as_object() else {
        return Ok(BTreeMap::new());
    };
    obj.iter()
        .map(|(k, n)| {
            Ok((
                k.clone(),
                n.as_i64()
                    .ok_or_else(|| anyhow!("`{k}` is not an integer"))?,
            ))
        })
        .collect()
}

fn grid(v: &Value) -> Result<[i64; 3]> {
    let n: Vec<i64> = v
        .as_array()
        .ok_or_else(|| anyhow!("kernels.json: an entry has no `grid`"))?
        .iter()
        .map(|x| {
            x.as_i64()
                .ok_or_else(|| anyhow!("a grid extent is not an integer"))
        })
        .collect::<Result<_>>()?;
    Ok(extend(&n))
}

/// A grid of one to three extents, the missing ones 1.
fn extend(n: &[i64]) -> [i64; 3] {
    [0, 1, 2].map(|i| n.get(i).copied().unwrap_or(1))
}

fn commit(root: &Path) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
